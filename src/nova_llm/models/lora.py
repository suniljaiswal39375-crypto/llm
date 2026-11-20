"""LoRA and DoRA (Weight-Decomposed Low-Rank Adaptation) with lossless merge/unmerge."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps an nn.Linear with LoRA or DoRA low-rank adapters."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_dora: bool = False,
    ) -> None:
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(1, rank)
        self.dropout = dropout
        self.use_dora = use_dora
        self.merged = False

        # Freeze base weights
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base_layer.in_features
        out_f = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_a = nn.Parameter(torch.empty(rank, in_f, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        if use_dora:
            with torch.no_grad():
                mag = base_layer.weight.float().norm(p=2, dim=1)
            self.dora_magnitude = nn.Parameter(mag.to(dtype=dtype))
        else:
            self.dora_magnitude = None

    def _delta_weight(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling

    def _effective_weight(self) -> torch.Tensor:
        w = self.base.weight + self._delta_weight()
        if self.use_dora and self.dora_magnitude is not None:
            col_norm = w.float().norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            w = (w.float() / col_norm * self.dora_magnitude.float().unsqueeze(1)).to(w.dtype)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)
        if self.use_dora:
            return F.linear(x, self._effective_weight(), self.base.bias)
        base_out = self.base(x)
        x_in = F.dropout(x, p=self.dropout, training=self.training) if self.dropout > 0.0 else x
        lora_out = F.linear(F.linear(x_in, self.lora_a), self.lora_b) * self.scaling
        return base_out + lora_out

    @torch.no_grad()
    def merge(self) -> None:
        """Fold adapter into base weights for zero inference overhead."""
        if self.merged:
            return
        self._orig_weight = self.base.weight.detach().clone()
        self.base.weight.copy_(self._effective_weight())
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Restore original base weights."""
        if not self.merged:
            return
        self.base.weight.copy_(self._orig_weight)
        del self._orig_weight
        self.merged = False


DEFAULT_LORA_TARGETS = ("wq", "wk", "wv", "wo", "wq_a", "wq_b", "wkv_a", "wkv_b", "w_gate", "w_up", "w_down")


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    use_dora: bool = False,
    target_names: tuple[str, ...] = DEFAULT_LORA_TARGETS,
) -> int:
    """Freeze base model and replace matching nn.Linear modules with LoRALinear."""
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(
                    module,
                    child_name,
                    LoRALinear(
                        child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        use_dora=use_dora,
                    ),
                )
                replaced += 1
    return replaced


def merge_lora(model: nn.Module) -> None:
    """Merge all LoRALinear adapters in-place."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora(model: nn.Module) -> None:
    """Unmerge all LoRALinear adapters in-place."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


@torch.no_grad()
def merge_and_unload_lora(model: nn.Module) -> int:
    """Permanently merge all LoRALinear adapters into base nn.Linear modules and unwrap them.

    Restores vanilla state_dict keys (e.g. `layers.0.attn.wq.weight`) for clean checkpointing,
    further full fine-tuning, and SafeTensors / vLLM / SGLang export.
    """
    unloaded = 0
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                if not child.merged:
                    child.merge()
                base = child.base
                for p in base.parameters():
                    p.requires_grad = True
                setattr(module, child_name, base)
                unloaded += 1
    return unloaded
