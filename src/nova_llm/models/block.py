"""Composable Transformer Block supporting GQA/MLA/Gated-DeltaNet, Dense/MoE, and Sandwich Norm."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nova_llm.config import ModelConfig
from nova_llm.models.attention import GroupedQueryAttention, MultiHeadLatentAttention
from nova_llm.models.ffn import SwiGLUFFN
from nova_llm.models.linear_attention import GatedDeltaNetAttention
from nova_llm.models.moe import DeepSeekMoE
from nova_llm.models.norm import build_norm
from nova_llm.models.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    """Single decoder layer composing pluggable token mixer + FFN/MoE + Pre/Sandwich Norm."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.sandwich_norm = cfg.sandwich_norm

        self.attn_norm = build_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)
        self.post_attn_norm = (
            build_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps) if cfg.sandwich_norm else None
        )

        if cfg.is_linear_attn_layer(layer_idx):
            self.attn: nn.Module = GatedDeltaNetAttention(cfg, layer_idx)
        elif cfg.attention_type == "mla":
            self.attn = MultiHeadLatentAttention(cfg, layer_idx)
        else:
            self.attn = GroupedQueryAttention(cfg, layer_idx)

        self.ffn_norm = build_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)
        self.post_ffn_norm = (
            build_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps) if cfg.sandwich_norm else None
        )

        if cfg.is_moe_layer(layer_idx):
            self.ffn: nn.Module = DeepSeekMoE(cfg)
        else:
            self.ffn = SwiGLUFFN(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: Any = None,
        doc_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        track_max_logits: bool = False,
    ) -> torch.Tensor:
        h = self.attn(
            self.attn_norm(x),
            rope=rope,
            position_ids=position_ids,
            kv_cache=kv_cache,
            doc_ids=doc_ids,
            attention_mask=attention_mask,
            track_max_logits=track_max_logits,
        )
        if self.post_attn_norm is not None:
            h = self.post_attn_norm(h)
        x = x + h

        f = self.ffn(self.ffn_norm(x))
        if self.post_ffn_norm is not None:
            f = self.post_ffn_norm(f)
        x = x + f
        return x
