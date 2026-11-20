"""Grouped-Query Attention (GQA + QK-Norm + SWA) and Multi-Head Latent Attention (MLA).

Supports:
- Document-boundary attention isolation for packed sequences (zero cross-doc leakage)
- Sliding-window attention (Gemma 3/4 & OLMo 3)
- Attention logit soft-capping (Gemma 2/3)
- Per-head max attention logit telemetry for MuonClip QK-Clip (Kimi K2)
- Offset-aware causal masking for chunked prefill & speculative verification
- Pre-allocated Static KV-Cache & MLA Compressed Latent Cache
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from nova_llm.config import ModelConfig
from nova_llm.models.norm import RMSNorm
from nova_llm.models.rope import RotaryEmbedding, apply_rotary_emb

if TYPE_CHECKING:
    from nova_llm.inference.kv_cache import LayerKVCache, LayerMLACache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads along head axis: (B, n_kv, L, D) -> (B, n_kv * n_rep, L, D)."""
    if n_rep == 1:
        return x
    bsz, n_kv, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bsz, n_kv, n_rep, seq_len, head_dim)
        .reshape(bsz, n_kv * n_rep, seq_len, head_dim)
    )


def build_attention_mask(
    q_len: int,
    k_len: int,
    start_pos: int,
    device: torch.device,
    sliding_window: int | None = None,
    doc_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a boolean allow-mask of shape (B or 1, 1, q_len, k_len) where True = attend.

    Handles causal offset (start_pos), sliding window, packed document isolation,
    and padding masks without NaN rows.
    """
    q_pos = torch.arange(start_pos, start_pos + q_len, device=device)
    k_pos = torch.arange(k_len, device=device)
    # Causal: query at q_pos can attend to key at k_pos <= q_pos
    allowed = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # (q_len, k_len)
    if sliding_window is not None:
        in_window = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)) < sliding_window
        allowed = allowed & in_window

    mask = allowed.unsqueeze(0).unsqueeze(0)  # (1, 1, q_len, k_len)

    if doc_ids is not None:
        # doc_ids: (B, q_len) for training/prefill where q_len == k_len
        if doc_ids.shape[1] != k_len:
            raise ValueError(
                f"doc_ids length ({doc_ids.shape[1]}) must match k_len ({k_len})"
            )
        q_docs = doc_ids[:, -q_len:]  # (B, q_len)
        same_doc = q_docs.unsqueeze(2) == doc_ids.unsqueeze(1)  # (B, q_len, k_len)
        mask = mask & same_doc.unsqueeze(1)

    if attention_mask is not None:
        # attention_mask: (B, k_len) with 1/True = valid, 0/False = pad
        key_valid = attention_mask.bool()[:, None, None, :]  # (B, 1, 1, k_len)
        mask = mask & key_valid
        # Guard against all-False rows (padding queries) which cause softmax NaN:
        # allow pad query to attend to position 0 (its output is ignored downstream)
        all_masked = ~mask.any(dim=-1, keepdim=True)
        if all_masked.any():
            fallback = torch.zeros_like(mask)
            fallback[..., 0] = True
            mask = torch.where(all_masked, fallback, mask)

    return mask


def _scaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    start_pos: int,
    dropout_p: float,
    sliding_window: int | None = None,
    doc_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    softcap: float | None = None,
    track_max_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute scaled dot-product attention via fast SDPA or exact FP32 manual path."""
    q_len = q.shape[2]
    k_len = k.shape[2]

    needs_custom_mask = (
        sliding_window is not None
        or doc_ids is not None
        or attention_mask is not None
        or (q_len > 1 and q_len != k_len)
    )
    needs_manual = softcap is not None or track_max_logits

    if not needs_manual:
        if not needs_custom_mask:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=(q_len > 1),
                scale=scale,
            )
            return out, None
        bool_mask = build_attention_mask(
            q_len=q_len,
            k_len=k_len,
            start_pos=start_pos,
            device=q.device,
            sliding_window=sliding_window,
            doc_ids=doc_ids,
            attention_mask=attention_mask,
        )
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bool_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )
        return out, None

    # Exact FP32 attention logits path (used when softcap or MuonClip QK-Clip tracking is active)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)

    bool_mask = build_attention_mask(
        q_len=q_len,
        k_len=k_len,
        start_pos=start_pos,
        device=q.device,
        sliding_window=sliding_window,
        doc_ids=doc_ids,
        attention_mask=attention_mask,
    )
    masked_scores = scores.masked_fill(~bool_mask, float("-inf"))

    max_logits: torch.Tensor | None = None
    if track_max_logits:
        with torch.no_grad():
            finite_scores = scores.masked_fill(~bool_mask, float("-inf"))
            # Per-head max over (B, q_len, k_len) -> (n_heads,)
            max_logits = finite_scores.amax(dim=(0, 2, 3))

    probs = F.softmax(masked_scores, dim=-1).to(v.dtype)
    if dropout_p > 0.0:
        probs = F.dropout(probs, p=dropout_p, training=True)
    out = torch.matmul(probs, v)
    return out, max_logits


class GroupedQueryAttention(nn.Module):
    """Modern GQA with optional QK-Norm, RoPE/NoPE, Sliding Window, and Static KV-Cache."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.use_rope = cfg.layer_uses_rope(layer_idx)
        self.sliding_window = cfg.layer_sliding_window(layer_idx)
        self.softcap = cfg.attn_logit_softcap
        self.dropout = cfg.dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=cfg.attention_bias)
        self.wk = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.wv = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps) if cfg.qk_norm else None

        # Telemetry buffer for MuonClip QK-Clip
        self.last_max_logits: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: LayerKVCache | None = None,
        doc_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        track_max_logits: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        start_pos = kv_cache.seq_len if kv_cache is not None else 0

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        if self.use_rope and rope is not None:
            cos, sin = rope(seq_len=seq_len, start_pos=start_pos, position_ids=position_ids)
            q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        dropout_p = self.dropout if self.training else 0.0
        out, max_logits = _scaled_attention(
            q=q,
            k=k_rep,
            v=v_rep,
            scale=self.scale,
            start_pos=start_pos,
            dropout_p=dropout_p,
            sliding_window=self.sliding_window,
            doc_ids=doc_ids,
            attention_mask=attention_mask,
            softcap=self.softcap,
            track_max_logits=track_max_logits,
        )
        self.last_max_logits = max_logits

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.wo(out)


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V3/V4 & Kimi K2 Multi-Head Latent Attention (MLA).

    Compresses KV into a compact low-rank latent vector `c_kv` + shared decoupled rotary key `k_rope`,
    dramatically shrinking KV-cache memory while preserving high-rank multi-head expressivity.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.q_lora_rank = cfg.mla_q_lora_rank
        self.kv_lora_rank = cfg.mla_kv_lora_rank
        self.qk_nope_head_dim = cfg.mla_qk_nope_head_dim
        self.qk_rope_head_dim = cfg.mla_qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = cfg.mla_v_head_dim
        self.sliding_window = cfg.layer_sliding_window(layer_idx)
        self.softcap = cfg.attn_logit_softcap
        self.dropout = cfg.dropout
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        # Query low-rank compression
        self.wq_a = nn.Linear(cfg.d_model, self.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(self.q_lora_rank, eps=cfg.norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        # KV low-rank compression + shared rotary key projection
        self.wkv_a = nn.Linear(cfg.d_model, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_lora_rank, eps=cfg.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_seq_len=cfg.max_seq_len,
            theta=cfg.rope_theta,
            scaling_type=cfg.rope_scaling_type,
            scaling_factor=cfg.rope_scaling_factor,
            orig_max_seq_len=cfg.rope_orig_max_seq_len,
        )
        self.last_max_logits: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: LayerMLACache | None = None,
        doc_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        track_max_logits: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        start_pos = kv_cache.seq_len if kv_cache is not None else 0

        # Query projection -> split into content (nope) and rotary (rope)
        q_compressed = self.q_a_norm(self.wq_a(x))
        q = self.wq_b(q_compressed).view(bsz, seq_len, self.n_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV compression -> c_kv (normalized) + shared k_rope
        kv_raw = self.wkv_a(x)
        c_kv, k_rope = torch.split(kv_raw, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_norm(c_kv)  # (B, L, kv_lora_rank)
        k_rope = k_rope.unsqueeze(1)  # (B, 1, L, qk_rope_head_dim)

        cos, sin = self.rope(seq_len=seq_len, start_pos=start_pos, position_ids=position_ids)
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, cos, sin)

        if kv_cache is not None:
            c_kv, k_rope = kv_cache.update(c_kv, k_rope)

        total_len = c_kv.shape[1]
        kv_up = self.wkv_b(c_kv).view(
            bsz, total_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, v = torch.split(kv_up, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rope_expanded = k_rope.expand(bsz, self.n_heads, total_len, self.qk_rope_head_dim)
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope_expanded], dim=-1)

        dropout_p = self.dropout if self.training else 0.0
        out, max_logits = _scaled_attention(
            q=q_full,
            k=k_full,
            v=v,
            scale=self.scale,
            start_pos=start_pos,
            dropout_p=dropout_p,
            sliding_window=self.sliding_window,
            doc_ids=doc_ids,
            attention_mask=attention_mask,
            softcap=self.softcap,
            track_max_logits=track_max_logits,
        )
        self.last_max_logits = max_logits

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.v_head_dim)
        return self.wo(out)
