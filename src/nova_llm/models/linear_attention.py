"""Gated DeltaNet-2 recurrent linear attention for 3:1 hybrid attention models.

Implements the decoupled decay + erase + write gated delta rule (ICLR 2025 Gated DeltaNet
& NVIDIA 2026 Gated DeltaNet-2, arXiv:2605.22791):

    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + omega_t v_t k_t^T
    o_t = S_t q_t

Features:
- Causal depthwise short convolutions on q, k, v with exact document-boundary isolation
- L2 normalization on q, k for training stability
- SiLU output gate + per-head RMSNorm
- O(1)-memory recurrent state cache matching full-sequence forward identically!
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from nova_llm.config import ModelConfig
from nova_llm.models.norm import RMSNorm

if TYPE_CHECKING:
    from nova_llm.inference.kv_cache import LayerLinearAttnState


class GatedDeltaNetAttention(nn.Module):
    """Hardware-friendly PyTorch Gated DeltaNet-2 token mixer with O(1) decode cache."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.inner_dim = self.n_heads * self.head_dim
        self.kernel_size = max(2, cfg.linear_conv_kernel)

        self.wq = nn.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.inner_dim, bias=False)

        # Depthwise causal short convolutions
        self.conv_q = nn.Conv1d(
            self.inner_dim, self.inner_dim, kernel_size=self.kernel_size, groups=self.inner_dim, bias=False
        )
        self.conv_k = nn.Conv1d(
            self.inner_dim, self.inner_dim, kernel_size=self.kernel_size, groups=self.inner_dim, bias=False
        )
        self.conv_v = nn.Conv1d(
            self.inner_dim, self.inner_dim, kernel_size=self.kernel_size, groups=self.inner_dim, bias=False
        )

        # Decoupled decay (alpha), erase gate (beta), and write gate (omega) per head
        self.w_alpha = nn.Linear(cfg.d_model, self.n_heads, bias=True)
        self.w_beta = nn.Linear(cfg.d_model, self.n_heads, bias=True)
        self.w_omega = nn.Linear(cfg.d_model, self.n_heads, bias=True)

        # Output gating & projection
        self.w_gate = nn.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.out_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
        self.wo = nn.Linear(self.inner_dim, cfg.d_model, bias=False)

        # Initialize alpha bias slightly positive so initial decay stays close to ~0.9
        nn.init.constant_(self.w_alpha.bias, 2.0)
        nn.init.constant_(self.w_beta.bias, -1.0)
        nn.init.constant_(self.w_omega.bias, -1.0)

    def _causal_short_conv(
        self,
        x: torch.Tensor,
        conv: nn.Conv1d,
        hist_buf: torch.Tensor | None,
        doc_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply strictly causal depthwise conv1d; handles both cached decode and packed doc_ids."""
        # x: (B, L, D) -> (B, D, L)
        bsz, seq_len, dim = x.shape
        x_t = x.transpose(1, 2)
        pad = self.kernel_size - 1

        if hist_buf is not None:
            # Concatenate cached history with new inputs
            padded = torch.cat([hist_buf, x_t], dim=-1)  # (B, D, pad + L)
            out = conv(padded)  # (B, D, L)
            new_hist = padded[:, :, -pad:].detach()
            return out.transpose(1, 2), new_hist

        if doc_ids is None:
            padded = F.pad(x_t, (pad, 0))
            out = conv(padded)
            return out.transpose(1, 2), None

        # Document-isolated causal convolution: ensure tokens from earlier documents never enter kernel
        weight = conv.weight.squeeze(1)  # (D, K)
        # Compute sliding windows explicitly over K taps with document equality masking
        padded = F.pad(x_t, (pad, 0))  # (B, D, pad + L)
        padded_docs = F.pad(doc_ids, (pad, 0), value=-1)  # (B, pad + L)
        out = torch.zeros_like(x_t)
        for tap in range(self.kernel_size):
            # tap 0 corresponds to oldest element (t - pad), tap K-1 to current token t
            slice_x = padded[:, :, tap : tap + seq_len]  # (B, D, L)
            slice_doc = padded_docs[:, tap : tap + seq_len]  # (B, L)
            valid = (slice_doc == doc_ids).unsqueeze(1).to(x_t.dtype)  # (B, 1, L)
            out = out + slice_x * valid * weight[:, tap].view(1, dim, 1)
        return out.transpose(1, 2), None

    def forward(
        self,
        x: torch.Tensor,
        rope=None,
        position_ids: torch.Tensor | None = None,
        kv_cache: LayerLinearAttnState | None = None,
        doc_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        track_max_logits: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q_raw = self.wq(x)
        k_raw = self.wk(x)
        v_raw = self.wv(x)

        cq_in = kv_cache.conv_q[:bsz] if kv_cache is not None else None
        ck_in = kv_cache.conv_k[:bsz] if kv_cache is not None else None
        cv_in = kv_cache.conv_v[:bsz] if kv_cache is not None else None

        q_conv, new_cq = self._causal_short_conv(q_raw, self.conv_q, cq_in, doc_ids)
        k_conv, new_ck = self._causal_short_conv(k_raw, self.conv_k, ck_in, doc_ids)
        v_conv, new_cv = self._causal_short_conv(v_raw, self.conv_v, cv_in, doc_ids)

        if kv_cache is not None and new_cq is not None:
            kv_cache.conv_q[:bsz].copy_(new_cq)
            kv_cache.conv_k[:bsz].copy_(new_ck)
            kv_cache.conv_v[:bsz].copy_(new_cv)

        q = F.silu(q_conv).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = F.silu(k_conv).view(bsz, seq_len, self.n_heads, self.head_dim)
        v = F.silu(v_conv).view(bsz, seq_len, self.n_heads, self.head_dim)

        # L2 normalization on q and k (essential per Gated DeltaNet ablations)
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
        v_f = v.float()

        # Decoupled decay alpha in (0, 1), erase beta in (0, 1), write omega in (0, 1)
        alpha = torch.sigmoid(self.w_alpha(x).float())  # (B, L, H)
        beta = torch.sigmoid(self.w_beta(x).float())  # (B, L, H)
        omega = torch.sigmoid(self.w_omega(x).float())  # (B, L, H)

        if kv_cache is not None:
            state = kv_cache.state[:bsz].clone()
        else:
            state = torch.zeros(
                bsz, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32
            )

        outputs = []
        prev_doc = None
        for t in range(seq_len):
            if doc_ids is not None:
                cur_doc = doc_ids[:, t]
                if prev_doc is not None:
                    same = (cur_doc == prev_doc).float().view(bsz, 1, 1, 1)
                    state = state * same
                prev_doc = cur_doc

            q_t = q[:, t]  # (B, H, D)
            k_t = k[:, t]  # (B, H, D)
            v_t = v_f[:, t]  # (B, H, D)
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            b_t = beta[:, t].unsqueeze(-1)  # (B, H, 1)
            w_t = omega[:, t].unsqueeze(-1)  # (B, H, 1)

            # S_t = alpha_t * S_{t-1} - alpha_t * beta_t * (S_{t-1} k_t) k_t^T + omega_t * v_t k_t^T
            decayed = a_t * state
            retrieved = torch.einsum("bhij,bhj->bhi", decayed, k_t)  # (B, H, D)
            delta_vec = (w_t * v_t) - (b_t * retrieved)  # (B, H, D)
            state = decayed + torch.einsum("bhi,bhj->bhij", delta_vec, k_t)

            o_t = torch.einsum("bhij,bhj->bhi", state, q_t)  # (B, H, D)
            outputs.append(o_t)

        if kv_cache is not None:
            kv_cache.state[:bsz].copy_(state.detach())
            kv_cache.seq_len += seq_len

        out = torch.stack(outputs, dim=1).to(x.dtype)  # (B, L, H, D)
        out = self.out_norm(out).reshape(bsz, seq_len, self.inner_dim)
        gate = F.silu(self.w_gate(x))
        return self.wo(out * gate)
