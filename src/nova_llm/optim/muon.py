"""Muon & MuonClip (QK-Clip) Optimizer — Moonlight & Kimi K2 (arXiv:2502.16982, 2507.20534).

Key mechanisms:
1. 5-step Newton-Schulz orthogonalization of momentum matrices for 2D hidden weights
2. Consistent RMS update scaling: 0.2 * sqrt(max(rows, cols)) (matches AdamW update RMS)
3. Decoupled weight decay
4. Per-head QK-Clip weight rescaling whenever max attention logit exceeds threshold `tau`
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from nova_llm.models.attention import GroupedQueryAttention, MultiHeadLatentAttention


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Compute approximate polar factor U V^T of a 2D matrix G via 5 Newton-Schulz iterations."""
    if g.ndim != 2:
        raise ValueError(f"Newton-Schulz expects a 2D tensor, got shape {tuple(g.shape)}")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D hidden weight matrices with RMS matching and decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim != 2:
                    raise ValueError(
                        f"Muon only optimizes 2D weight matrices; got shape {tuple(grad.shape)}"
                    )
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(grad)
                update_in = grad.add(buf, alpha=mu) if nesterov else buf

                ortho = zeropower_via_newtonschulz5(update_in, steps=ns_steps)
                # Moonlight / Kimi K2 consistent RMS scaling matching AdamW
                rms_scale = 0.2 * math.sqrt(max(p.shape[0], p.shape[1]))

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho, alpha=-lr * rms_scale)

        return loss


@torch.no_grad()
def apply_qk_clip_(model: nn.Module, tau: float = 100.0) -> dict[str, float]:
    """Apply Kimi K2 QK-Clip per-head weight rescaling when max attention logit > tau.

    Handles both GQA (rescaling per-head W_q and corresponding KV-group W_k by sqrt(gamma))
    and MLA (rescaling q_nope/k_nope by sqrt(gamma), q_rope by gamma, leaving shared k_rope intact).
    """
    clipped_heads = 0
    global_max_logit = 0.0

    for module in model.modules():
        if isinstance(module, GroupedQueryAttention):
            ml = module.last_max_logits
            if ml is None:
                continue
            max_val = float(ml.max().item())
            global_max_logit = max(global_max_logit, max_val)
            if max_val <= tau:
                continue

            # Per-head scaling gamma_h = min(1, tau / S_max^h)
            gammas = torch.clamp(tau / ml.float().clamp(min=1e-6), max=1.0)
            n_heads = module.n_heads
            n_kv = module.n_kv_heads
            n_rep = module.n_rep
            hd = module.head_dim

            # For GQA, group heads sharing a KV head take the strictest gamma for W_k,
            # while each query head scales W_q by gamma_h / sqrt(gamma_group) so q^T k scales by gamma_h.
            wq_view = module.wq.weight.view(n_heads, hd, -1)
            wk_view = module.wk.weight.view(n_kv, hd, -1)

            for g in range(n_kv):
                group_gammas = gammas[g * n_rep : (g + 1) * n_rep]
                if float(group_gammas.min().item()) >= 1.0:
                    continue
                g_min = float(group_gammas.min().item())
                k_scale = math.sqrt(g_min)
                wk_view[g].mul_(k_scale)
                for r in range(n_rep):
                    h_idx = g * n_rep + r
                    gh = float(gammas[h_idx].item())
                    if gh < 1.0:
                        q_scale = gh / k_scale
                        wq_view[h_idx].mul_(q_scale)
                        clipped_heads += 1

        elif isinstance(module, MultiHeadLatentAttention):
            ml = module.last_max_logits
            if ml is None:
                continue
            max_val = float(ml.max().item())
            global_max_logit = max(global_max_logit, max_val)
            if max_val <= tau:
                continue

            gammas = torch.clamp(tau / ml.float().clamp(min=1e-6), max=1.0)
            n_heads = module.n_heads
            d_nope = module.qk_nope_head_dim
            d_rope = module.qk_rope_head_dim
            d_v = module.v_head_dim

            wq_b_view = module.wq_b.weight.view(n_heads, d_nope + d_rope, -1)
            wkv_b_view = module.wkv_b.weight.view(n_heads, d_nope + d_v, -1)

            for h in range(n_heads):
                gh = float(gammas[h].item())
                if gh < 1.0:
                    sqrt_g = math.sqrt(gh)
                    # Kimi K2 Algorithm 1: qC & kC scaled by sqrt(gamma), qR by gamma, shared kR untouched
                    wq_b_view[h, :d_nope, :].mul_(sqrt_g)
                    wkv_b_view[h, :d_nope, :].mul_(sqrt_g)
                    wq_b_view[h, d_nope:, :].mul_(gh)
                    clipped_heads += 1

    return {"qk_clipped_heads": float(clipped_heads), "max_attn_logit_pre_clip": global_max_logit}
