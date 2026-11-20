"""Pre-allocated static KV-cache, MLA latent cache, and linear-attention recurrent state.

Eliminates O(T^2) torch.cat reallocations in the hot decode path and maintains
torch.compile / CUDA-Graph compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LayerKVCache:
    """Pre-allocated static K/V buffers for a single GQA/MHA layer."""

    k: torch.Tensor  # (B, n_kv_heads, max_seq_len, head_dim)
    v: torch.Tensor  # (B, n_kv_heads, max_seq_len, head_dim)
    seq_len: int = 0

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V tokens in-place at [seq_len : seq_len + L] and return valid prefix view."""
        bsz, n_kv, new_len, head_dim = k_new.shape
        end = self.seq_len + new_len
        if end > self.k.shape[2]:
            raise ValueError(
                f"KV-cache overflow: trying to store {end} tokens in cache of size {self.k.shape[2]}"
            )
        self.k[:bsz, :n_kv, self.seq_len : end, :head_dim].copy_(k_new.detach())
        self.v[:bsz, :n_kv, self.seq_len : end, :head_dim].copy_(v_new.detach())
        self.seq_len = end
        return self.k[:bsz, :n_kv, :end, :head_dim], self.v[:bsz, :n_kv, :end, :head_dim]

    def reset(self) -> None:
        self.seq_len = 0


@dataclass
class LayerMLACache:
    """Pre-allocated compressed latent KV cache for DeepSeek-style MLA layers."""

    c_kv: torch.Tensor  # (B, max_seq_len, kv_lora_rank)
    k_rope: torch.Tensor  # (B, 1, max_seq_len, qk_rope_head_dim)
    seq_len: int = 0

    def update(
        self, c_kv_new: torch.Tensor, k_rope_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, new_len, rank = c_kv_new.shape
        end = self.seq_len + new_len
        if end > self.c_kv.shape[1]:
            raise ValueError(
                f"MLA-cache overflow: trying to store {end} tokens in cache of size {self.c_kv.shape[1]}"
            )
        self.c_kv[:bsz, self.seq_len : end, :rank].copy_(c_kv_new.detach())
        self.k_rope[:bsz, :, self.seq_len : end, :].copy_(k_rope_new.detach())
        self.seq_len = end
        return self.c_kv[:bsz, :end, :rank], self.k_rope[:bsz, :, :end, :]

    def reset(self) -> None:
        self.seq_len = 0


@dataclass
class LayerLinearAttnState:
    """O(1)-memory recurrent matrix state + short-conv ring buffer for Gated DeltaNet-2."""

    state: torch.Tensor  # (B, n_heads, head_dim, head_dim)
    conv_q: torch.Tensor  # (B, conv_dim, kernel_size - 1)
    conv_k: torch.Tensor  # (B, conv_dim, kernel_size - 1)
    conv_v: torch.Tensor  # (B, conv_dim, kernel_size - 1)
    seq_len: int = 0

    def reset(self) -> None:
        self.state.zero_()
        self.conv_q.zero_()
        self.conv_k.zero_()
        self.conv_v.zero_()
        self.seq_len = 0


class InferenceCache:
    """Unified multi-layer cache container supporting GQA, MLA, and Hybrid Linear layers."""

    def __init__(self, layers: list[Any], max_seq_len: int) -> None:
        self.layers = layers
        self.max_seq_len = max_seq_len

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def truncate(self, new_len: int) -> None:
        """Roll back KV/MLA caches to new_len (used after speculative decoding rejection)."""
        for layer in self.layers:
            if isinstance(layer, LayerLinearAttnState):
                raise RuntimeError(
                    "Linear attention recurrent states cannot be truncated in-place; snapshot instead."
                )
            if new_len > layer.seq_len:
                raise ValueError(f"Cannot truncate cache from {layer.seq_len} to {new_len}")
            layer.seq_len = new_len

    def snapshot_linear_states(self) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """Clone recurrent states if speculative rollback is needed on hybrid models."""
        snaps = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, LayerLinearAttnState):
                snaps[idx] = (
                    layer.state.clone(),
                    layer.conv_q.clone(),
                    layer.conv_k.clone(),
                    layer.conv_v.clone(),
                    layer.seq_len,
                )
        return snaps

    def restore_linear_states(
        self,
        snaps: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]],
    ) -> None:
        for idx, (s, cq, ck, cv, sl) in snaps.items():
            layer = self.layers[idx]
            layer.state.copy_(s)
            layer.conv_q.copy_(cq)
            layer.conv_k.copy_(ck)
            layer.conv_v.copy_(cv)
            layer.seq_len = sl

    @classmethod
    def allocate(
        cls,
        model_cfg: Any,
        batch_size: int,
        max_seq_len: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "InferenceCache":
        """Pre-allocate static cache buffers matching every layer of the model."""
        cap = max_seq_len or model_cfg.max_seq_len
        layers: list[Any] = []
        for idx in range(model_cfg.n_layers):
            if model_cfg.is_linear_attn_layer(idx):
                head_dim = model_cfg.head_dim
                conv_dim = model_cfg.n_heads * head_dim
                k_hist = max(1, model_cfg.linear_conv_kernel - 1)
                layers.append(
                    LayerLinearAttnState(
                        state=torch.zeros(
                            batch_size, model_cfg.n_heads, head_dim, head_dim, device=device, dtype=torch.float32
                        ),
                        conv_q=torch.zeros(batch_size, conv_dim, k_hist, device=device, dtype=dtype),
                        conv_k=torch.zeros(batch_size, conv_dim, k_hist, device=device, dtype=dtype),
                        conv_v=torch.zeros(batch_size, conv_dim, k_hist, device=device, dtype=dtype),
                    )
                )
            elif model_cfg.attention_type == "mla":
                layers.append(
                    LayerMLACache(
                        c_kv=torch.zeros(
                            batch_size, cap, model_cfg.mla_kv_lora_rank, device=device, dtype=dtype
                        ),
                        k_rope=torch.zeros(
                            batch_size, 1, cap, model_cfg.mla_qk_rope_head_dim, device=device, dtype=dtype
                        ),
                    )
                )
            else:
                layers.append(
                    LayerKVCache(
                        k=torch.zeros(
                            batch_size, model_cfg.n_kv_heads, cap, model_cfg.head_dim, device=device, dtype=dtype
                        ),
                        v=torch.zeros(
                            batch_size, model_cfg.n_kv_heads, cap, model_cfg.head_dim, device=device, dtype=dtype
                        ),
                    )
                )
        return cls(layers=layers, max_seq_len=cap)
