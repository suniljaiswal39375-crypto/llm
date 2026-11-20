"""Group-wise INT8 / INT4 weight-only quantization and QAT (Quantization-Aware Training) STE hooks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """Group-wise symmetric INT8 or INT4 weight-only quantized Linear layer."""

    def __init__(self, linear: nn.Linear, bits: int = 8, group_size: int = 64) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.group_size = min(group_size, self.in_features)
        self.qmax = (1 << (bits - 1)) - 1  # 127 for int8, 7 for int4

        w = linear.weight.detach().float()
        pad = (self.group_size - (self.in_features % self.group_size)) % self.group_size
        self.pad = pad
        if pad > 0:
            w = F.pad(w, (0, pad))
        n_groups = w.shape[1] // self.group_size
        w_grouped = w.view(self.out_features, n_groups, self.group_size)
        scales = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.qmax
        q_w = torch.clamp(torch.round(w_grouped / scales), -self.qmax, self.qmax).to(torch.int8)

        if bits == 4:
            # Pack two signed int4 values (offset by +8 -> uint4 in [0..15]) per uint8 byte!
            u4 = (q_w + 8).to(torch.uint8).reshape(self.out_features, -1)
            if u4.shape[1] % 2 != 0:
                u4 = F.pad(u4, (0, 1))
            packed = (u4[:, 0::2] & 0x0F) | ((u4[:, 1::2] & 0x0F) << 4)
            self.register_buffer("qweight", packed)
        else:
            self.register_buffer("qweight", q_w.reshape(self.out_features, -1))

        self.register_buffer("scales", scales.squeeze(-1).to(torch.float16))
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        n_groups = self.scales.shape[1]
        if self.bits == 4:
            lo = (self.qweight & 0x0F).to(torch.int8) - 8
            hi = ((self.qweight >> 4) & 0x0F).to(torch.int8) - 8
            unpacked = torch.stack([lo, hi], dim=-1).reshape(self.out_features, -1)
            unpacked = unpacked[:, : n_groups * self.group_size]
            q_grouped = unpacked.view(self.out_features, n_groups, self.group_size).float()
        else:
            q_grouped = self.qweight.view(self.out_features, n_groups, self.group_size).float()
        w = (q_grouped * self.scales.float().unsqueeze(-1)).reshape(self.out_features, -1)
        if self.pad > 0:
            w = w[:, : self.in_features]
        return w.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight(x.dtype)
        return F.linear(x, w, self.bias)


def quantize_model_weight_only(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 64,
    skip_names: tuple[str, ...] = ("lm_head", "router"),
) -> dict[str, float]:
    """Replace internal Linear layers with group-wise INT8/INT4 QuantizedLinear."""
    orig_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    replaced = 0
    for mod_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name not in skip_names:
                setattr(module, child_name, QuantizedLinear(child, bits=bits, group_size=group_size))
                replaced += 1
    new_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    return {
        "layers_quantized": float(replaced),
        "orig_mb": orig_bytes / (1024 * 1024),
        "quant_mb": new_bytes / (1024 * 1024),
        "compression_ratio": orig_bytes / max(1, new_bytes),
    }


class QATLinear(nn.Module):
    """Straight-Through Estimator (STE) fake-quantized Linear for Quantization-Aware Training."""

    def __init__(self, linear: nn.Linear, bits: int = 4, group_size: int = 64) -> None:
        super().__init__()
        self.linear = linear
        self.bits = bits
        self.group_size = min(group_size, linear.in_features)
        self.qmax = (1 << (bits - 1)) - 1

    def _fake_quant(self, w: torch.Tensor) -> torch.Tensor:
        out_f, in_f = w.shape
        pad = (self.group_size - (in_f % self.group_size)) % self.group_size
        w_p = F.pad(w.float(), (0, pad)) if pad > 0 else w.float()
        w_g = w_p.view(out_f, -1, self.group_size)
        scale = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.qmax
        q = torch.clamp(torch.round(w_g / scale), -self.qmax, self.qmax) * scale
        q_flat = q.reshape(out_f, -1)[:, :in_f].to(w.dtype)
        # Straight-through estimator: forward uses fake-quantized w, backward flows through exact w
        return w + (q_flat - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self._fake_quant(self.linear.weight)
        return F.linear(x, w_q, self.linear.bias)


def apply_qat(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
    skip_names: tuple[str, ...] = ("lm_head", "router"),
) -> int:
    """Wrap linear layers with STE fake-quantization for native INT4/INT8 QAT (Kimi K2.6 style)."""
    count = 0
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name not in skip_names:
                setattr(module, child_name, QATLinear(child, bits=bits, group_size=group_size))
                count += 1
    return count
