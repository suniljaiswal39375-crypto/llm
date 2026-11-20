"""FP32 Rotary Position Embeddings (RoPE) with linear & YaRN long-context scaling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int, device: torch.device) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (
        max_val - min_val
    )
    return torch.clamp(linear_func, 0.0, 1.0)


class RotaryEmbedding(nn.Module):
    """Split-half FP32 RoPE supporting standard, linear, and YaRN context extension."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 500000.0,
        scaling_type: str = "none",
        scaling_factor: float = 1.0,
        orig_max_seq_len: int = 2048,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")
        self.dim = dim
        self.theta = float(theta)
        self.scaling_type = scaling_type
        self.scaling_factor = float(scaling_factor)
        self.orig_max_seq_len = int(orig_max_seq_len)
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        inv_freq, self.attention_factor = self._compute_inv_freq(device=torch.device("cpu"))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_max_len = 0
        self.register_buffer("_cos_cached", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0, dtype=torch.float32), persistent=False)
        self._build_cache(max_seq_len)

    def _compute_inv_freq(self, device: torch.device) -> tuple[torch.Tensor, float]:
        pos_freqs = self.theta ** (
            torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim
        )
        inv_freq_extrap = 1.0 / pos_freqs
        if self.scaling_type == "none" or self.scaling_factor <= 1.0:
            return inv_freq_extrap, 1.0
        if self.scaling_type == "linear":
            return inv_freq_extrap / self.scaling_factor, 1.0
        if self.scaling_type == "yarn":
            inv_freq_interp = inv_freq_extrap / self.scaling_factor
            low, high = _yarn_find_correction_range(
                self.beta_fast, self.beta_slow, self.dim, self.theta, self.orig_max_seq_len
            )
            ramp = 1.0 - _yarn_linear_ramp_mask(low, high, self.dim // 2, device=device)
            inv_freq = inv_freq_interp * (1.0 - ramp) + inv_freq_extrap * ramp
            mscale = 0.1 * math.log(self.scaling_factor) + 1.0
            return inv_freq, mscale
        raise ValueError(f"Unknown rope_scaling_type: {self.scaling_type!r}")

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self._cached_max_len and self._cos_cached.numel() > 0:
            return
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.float())
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self._cos_cached = emb.cos() * self.attention_factor
        self._sin_cached = emb.sin() * self.attention_factor
        self._cached_max_len = seq_len

    def forward(
        self,
        seq_len: int,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FP32 (cos, sin) broadcastable to (B, 1, L, dim)."""
        if position_ids is not None:
            # position_ids: (B, L) -> compute exact FP32 freqs on the fly
            freqs = torch.einsum("bl,d->bld", position_ids.float(), self.inv_freq.float())
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = (emb.cos() * self.attention_factor).unsqueeze(1)  # (B, 1, L, dim)
            sin = (emb.sin() * self.attention_factor).unsqueeze(1)
            return cos, sin

        needed = start_pos + seq_len
        if needed > self._cached_max_len:
            self._build_cache(needed)
        cos = self._cos_cached[start_pos:needed].unsqueeze(0).unsqueeze(0)  # (1, 1, L, dim)
        sin = self._sin_cached[start_pos:needed].unsqueeze(0).unsqueeze(0)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply split-half RoPE in FP32 and cast back to input dtype.

    q, k shape: (B, n_heads, L, dim)
    cos, sin shape: (1 or B, 1, L, dim)
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    q_fp32, k_fp32 = q.float(), k.float()
    cos_f, sin_f = cos.float(), sin.float()
    q_rot = (q_fp32 * cos_f) + (_rotate_half(q_fp32) * sin_f)
    k_rot = (k_fp32 * cos_f) + (_rotate_half(k_fp32) * sin_f)
    return q_rot.to(q_dtype), k_rot.to(k_dtype)
