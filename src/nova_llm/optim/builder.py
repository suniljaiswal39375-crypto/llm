"""Optimizer builder enforcing strict zero weight decay on norms, biases, and embeddings."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nova_llm.config import OptimConfig
from nova_llm.models.norm import RMSNorm, SingleScaleRMSNorm
from nova_llm.optim.muon import Muon


class HybridOptimizer:
    """Composes Muon (for hidden 2D matrices) + AdamW (for embeddings/heads/norms/1D params)."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        states = state_dict["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError(
                f"Optimizer count mismatch: checkpoint has {len(states)}, live has {len(self.optimizers)}"
            )
        for opt, st in zip(self.optimizers, states):
            opt.load_state_dict(st)


def _classify_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Split trainable parameters into (muon_2d_hidden, adam_decay_2d, adam_no_decay).

    Rules (OLMo 2 / Kimi K2 / Moonlight best practices):
    - NEVER apply weight decay to:
      * 1D tensors (biases, RMSNorm / SingleScaleRMSNorm weights)
      * Token embeddings (`nn.Embedding`)
      * Tied or untied `lm_head` projection (routed to AdamW with no decay for embeddings,
        decay only if untied non-embedding — safer: keep embeddings strictly no-decay!)
    - Muon applies ONLY to internal hidden 2D Linear matrices (attention/FFN/MoE/MTP projections,
      excluding embeddings, lm_head, and small routers).
    """
    norm_param_ids: set[int] = set()
    embed_param_ids: set[int] = set()
    lm_head_param_ids: set[int] = set()

    for mod_name, module in model.named_modules():
        if isinstance(module, (RMSNorm, SingleScaleRMSNorm, nn.LayerNorm)):
            for p in module.parameters(recurse=False):
                norm_param_ids.add(id(p))
        elif isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                embed_param_ids.add(id(p))
        elif mod_name == "lm_head":
            for p in module.parameters(recurse=False):
                lm_head_param_ids.add(id(p))

    seen: set[int] = set()
    muon_params: list[nn.Parameter] = []
    adam_decay: list[nn.Parameter] = []
    adam_no_decay: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        pid = id(p)

        # Strictly zero weight decay on norms, biases, 1D tensors, and embeddings
        if pid in norm_param_ids or pid in embed_param_ids or p.ndim < 2 or name.endswith(".bias"):
            adam_no_decay.append(p)
        elif pid in lm_head_param_ids:
            # lm_head (when untied) goes to AdamW with no weight decay (matches embedding stability)
            adam_no_decay.append(p)
        elif p.ndim == 2 and not name.endswith(".router.weight"):
            muon_params.append(p)
        else:
            # e.g., Conv1d weights or FP32 MoE router -> AdamW with decay
            adam_decay.append(p)

    return muon_params, adam_decay, adam_no_decay


def build_optimizer(
    model: nn.Module,
    cfg: OptimConfig,
    device: torch.device | None = None,
) -> HybridOptimizer | torch.optim.AdamW:
    """Build AdamW or Hybrid Muon/MuonClip + AdamW with verified weight-decay hygiene."""
    muon_params, adam_decay, adam_no_decay = _classify_parameters(model)
    use_fused = device is not None and device.type == "cuda"

    if cfg.optimizer == "adamw":
        decay_group = muon_params + adam_decay
        param_groups = []
        if decay_group:
            param_groups.append(
                {
                    "params": decay_group,
                    "lr": cfg.lr,
                    "base_lr": cfg.lr,
                    "weight_decay": cfg.weight_decay,
                    "name": "adamw_decay",
                }
            )
        if adam_no_decay:
            param_groups.append(
                {
                    "params": adam_no_decay,
                    "lr": cfg.lr,
                    "base_lr": cfg.lr,
                    "weight_decay": 0.0,
                    "name": "adamw_no_decay",
                }
            )
        return torch.optim.AdamW(
            param_groups,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            fused=use_fused,
        )

    if cfg.optimizer in ("muon", "muonclip"):
        opts: list[torch.optim.Optimizer] = []
        if muon_params:
            opts.append(
                Muon(
                    [
                        {
                            "params": muon_params,
                            "lr": cfg.muon_lr,
                            "base_lr": cfg.muon_lr,
                            "weight_decay": cfg.weight_decay,
                            "name": "muon_hidden_2d",
                        }
                    ],
                    lr=cfg.muon_lr,
                    momentum=cfg.muon_momentum,
                    weight_decay=cfg.weight_decay,
                    ns_steps=cfg.muon_ns_steps,
                )
            )
        adam_groups = []
        if adam_decay:
            adam_groups.append(
                {
                    "params": adam_decay,
                    "lr": cfg.lr,
                    "base_lr": cfg.lr,
                    "weight_decay": cfg.weight_decay,
                    "name": "adamw_decay",
                }
            )
        if adam_no_decay:
            adam_groups.append(
                {
                    "params": adam_no_decay,
                    "lr": cfg.lr,
                    "base_lr": cfg.lr,
                    "weight_decay": 0.0,
                    "name": "adamw_no_decay",
                }
            )
        if adam_groups:
            opts.append(
                torch.optim.AdamW(
                    adam_groups,
                    betas=(cfg.beta1, cfg.beta2),
                    eps=cfg.eps,
                    fused=use_fused,
                )
            )
        return HybridOptimizer(opts)

    raise ValueError(f"Unsupported optimizer: {cfg.optimizer!r}")
