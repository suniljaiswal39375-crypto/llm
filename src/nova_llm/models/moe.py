"""DeepSeek-V3/V4 Fine-Grained Sparse Mixture-of-Experts with Auxiliary-Loss-Free Load Balancing.

Key frontier innovations:
- Always-active shared expert(s) + top-k fine-grained routed experts
- FP32 router weights and affinity computation (prevents low-precision routing collapse)
- Auxiliary-loss-free load balancing (DeepSeek-V3): maintains per-expert dynamic bias `expert_bias`
  used ONLY for top-k selection, while combining outputs with unbiased affinity weights; updated
  outside autograd after each step with zero interference to the language modeling gradient!
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nova_llm.config import ModelConfig
from nova_llm.models.ffn import SwiGLUFFN


class DeepSeekMoE(nn.Module):
    """Sparse MoE layer with shared experts and auxiliary-loss-free dynamic bias balancing."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_routed_experts = cfg.n_routed_experts
        self.top_k = cfg.moe_top_k
        self.n_shared_experts = cfg.n_shared_experts
        self.aux_free_gamma = cfg.moe_aux_free_gamma
        expert_d_ff = cfg.moe_expert_d_ff or cfg.d_ff

        # Router linear (always evaluated in FP32)
        self.router = nn.Linear(cfg.d_model, self.n_routed_experts, bias=False)

        # Dynamic load-balancing bias (non-trainable buffer, updated outside backprop)
        self.register_buffer("expert_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))
        self.register_buffer("_last_expert_loads", torch.zeros(self.n_routed_experts, dtype=torch.float32), persistent=False)

        self.experts = nn.ModuleList(
            [SwiGLUFFN(cfg.d_model, expert_d_ff, dropout=cfg.dropout) for _ in range(self.n_routed_experts)]
        )
        self.shared_experts = (
            SwiGLUFFN(cfg.d_model, expert_d_ff * self.n_shared_experts, dropout=cfg.dropout)
            if self.n_shared_experts > 0
            else None
        )
        self.last_load_entropy: float = 0.0
        self.last_max_load_ratio: float = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        bsz_seq = orig_shape[0] * orig_shape[1]
        flat_x = x.reshape(bsz_seq, orig_shape[-1])

        # Compute affinities in FP32 (sigmoid routing per DeepSeek-V3)
        router_logits = self.router(flat_x.float())  # (N, E)
        affinities = torch.sigmoid(router_logits)  # (N, E)

        # Selection uses biased scores; combination weights use unbiased affinities!
        biased_scores = affinities + self.expert_bias.unsqueeze(0)
        _, topk_idx = torch.topk(biased_scores, k=self.top_k, dim=-1)  # (N, K)
        topk_weights = torch.gather(affinities, dim=-1, index=topk_idx)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = topk_weights.to(flat_x.dtype)

        # Track per-expert load statistics (detached)
        with torch.no_grad():
            counts = torch.bincount(topk_idx.reshape(-1), minlength=self.n_routed_experts).float()
            total_assign = max(1.0, float(counts.sum().item()))
            fracs = counts / total_assign
            self._last_expert_loads.copy_(fracs)
            probs = fracs.clamp(min=1e-12)
            entropy = -(probs * probs.log()).sum().item()
            self.last_load_entropy = entropy / max(1e-9, math.log(self.n_routed_experts))
            mean_load = total_assign / self.n_routed_experts
            self.last_max_load_ratio = float(counts.max().item() / max(1e-9, mean_load))

        # Sparse expert dispatch
        out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_mask, slot_idx = torch.where(topk_idx == expert_id)
            if token_mask.numel() == 0:
                continue
            expert_in = flat_x[token_mask]
            expert_out = expert(expert_in)
            w = topk_weights[token_mask, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_mask, expert_out * w)

        if self.shared_experts is not None:
            out = out + self.shared_experts(flat_x)

        return out.view(orig_shape)

    @torch.no_grad()
    def update_load_balance_bias(self, gamma: float | None = None) -> None:
        """Update dynamic router bias after an optimizer step (DeepSeek-V3 aux-loss-free)."""
        step = self.aux_free_gamma if gamma is None else gamma
        if step <= 0.0:
            return
        target = 1.0 / self.n_routed_experts
        error = target - self._last_expert_loads
        self.expert_bias.add_(step * torch.sign(error))
