"""Device detection and mixed-precision autocast context utilities."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


def resolve_device(device: str | None = None) -> torch.device:
    """Pick explicit device or auto-detect CUDA > MPS > CPU."""
    if device is not None and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    """Resolve safe autocast dtype for the given precision string and device."""
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    if precision == "fp16":
        if device.type == "cpu":
            # CPU autocast does not support fp16 robustly; fall back to bf16 or fp32
            return torch.bfloat16
        return torch.float16
    raise ValueError(f"Unsupported precision: {precision!r}")


def autocast_context(precision: str, device: torch.device) -> ContextManager:
    """Return an appropriate autocast context manager (nullcontext for fp32)."""
    if precision == "fp32":
        return nullcontext()
    dtype = resolve_dtype(precision, device)
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated GPU memory in MB (0.0 on CPU)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0
