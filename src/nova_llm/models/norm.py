"""FP32-safe RMSNorm and Single-Scale RMSNorm (Outlier-Safe Pre-Training, ACL 2025)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with mandatory FP32 variance accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = (x_fp32 * inv_rms).type_as(x)
        return normed * self.weight.type_as(x)


class SingleScaleRMSNorm(nn.Module):
    """Single-Scale RMSNorm from Outlier-Safe Pre-Training (OSP, ACL 2025).

    Uses a single scalar gain instead of per-channel weights to prevent channel-wise
    outlier amplification during pre-training, improving post-training 4-bit quantization.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = (x_fp32 * inv_rms).type_as(x)
        return normed * self.weight.type_as(x)


def build_norm(dim: int, norm_type: str = "rmsnorm", eps: float = 1e-6) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim=dim, eps=eps)
    if norm_type == "single_scale_rmsnorm":
        return SingleScaleRMSNorm(dim=dim, eps=eps)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")
