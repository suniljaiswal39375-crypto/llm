"""NovaLM: Frontier 2026 Composable Decoder-Only / Hybrid / MoE Language Model.

Includes:
- GQA + QK-Norm, MLA (DeepSeek-V3/V4), Sliding Window (Gemma 3/4), Gated DeltaNet-2 Hybrid (Kimi Linear)
- Pre-RMSNorm / Gemma Sandwich Norm / Single-Scale RMSNorm (OSP)
- DeepSeek-style Auxiliary-Loss-Free Sparse MoE + Shared Experts
- FP32 Cross-Entropy + Z-Loss + Optional Output Logit Soft-Capping
- Multi-Token Prediction (MTP) auxiliary heads (training boost + native speculative drafting)
- Optional Outlier-Safe Pre-Training (OSP) learnable embedding projection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nova_llm.config import ModelConfig
from nova_llm.models.block import TransformerBlock
from nova_llm.models.moe import DeepSeekMoE
from nova_llm.models.norm import build_norm
from nova_llm.models.rope import RotaryEmbedding

if TYPE_CHECKING:
    from nova_llm.inference.kv_cache import InferenceCache


@dataclass
class LMOutput:
    """Structured forward output from NovaLM."""

    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: torch.Tensor | None = None
    ce_loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    num_valid_tokens: int = 0


class MTPModule(nn.Module):
    """DeepSeek-V3 Multi-Token Prediction module for depth k >= 1."""

    def __init__(self, cfg: ModelConfig, depth_idx: int) -> None:
        super().__init__()
        self.depth_idx = depth_idx
        self.prev_norm = build_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)
        self.emb_norm = build_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)
        self.proj = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False)
        # Lightweight dense GQA block for MTP head
        mtp_block_cfg = ModelConfig(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=1,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
            attention_type="gqa",
            qk_norm=cfg.qk_norm,
            rope_theta=cfg.rope_theta,
            norm_type=cfg.norm_type,
            norm_eps=cfg.norm_eps,
            sandwich_norm=cfg.sandwich_norm,
            use_moe=False,
            dropout=cfg.dropout,
            pad_vocab_multiple=1,
        )
        self.block = TransformerBlock(mtp_block_cfg, layer_idx=0)
        self.final_norm = build_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        next_token_emb: torch.Tensor,
        rope: RotaryEmbedding,
        doc_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = torch.cat([self.prev_norm(prev_hidden), self.emb_norm(next_token_emb)], dim=-1)
        h = self.proj(fused)
        h = self.block(h, rope=rope, doc_ids=doc_ids, attention_mask=attention_mask)
        return self.final_norm(h)


class NovaLM(nn.Module):
    """Composable frontier decoder-only / hybrid / MoE language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.activation_checkpointing = False

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.osp_embed_proj = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False) if cfg.use_osp_embed_proj else None
        )

        self.rope = RotaryEmbedding(
            dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            theta=cfg.rope_theta,
            scaling_type=cfg.rope_scaling_type,
            scaling_factor=cfg.rope_scaling_factor,
            orig_max_seq_len=cfg.rope_orig_max_seq_len,
        )

        self.layers = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = build_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.mtp_modules = nn.ModuleList(
            [MTPModule(cfg, k + 1) for k in range(cfg.mtp_depth)]
        )

        # Initialize weights BEFORE tying so shared weights are never double-initialized
        self._init_weights()
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self) -> None:
        std = self.cfg.init_std
        residual_std = std / math.sqrt(2.0 * max(1, self.cfg.n_layers))

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_resid_out = name.endswith(".wo") or name.endswith(".w_down")
                s = residual_std if is_resid_out else std
                nn.init.normal_(module.weight, mean=0.0, std=s)
                if module.bias is not None and not name.endswith(("w_alpha", "w_beta", "w_omega")):
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        if self.osp_embed_proj is not None:
            nn.init.orthogonal_(self.osp_embed_proj.weight)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        if self.osp_embed_proj is not None:
            x = self.osp_embed_proj(x)
        return x

    def _project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        if self.cfg.final_logit_softcap is not None:
            cap = self.cfg.final_logit_softcap
            logits = cap * torch.tanh(logits / cap)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: InferenceCache | None = None,
        doc_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        track_max_logits: bool = False,
        loss_reduction: str = "mean",
    ) -> LMOutput:
        """Run forward pass and optionally compute FP32 CE + Z-loss + MTP loss.

        Args:
            input_ids: (B, L) token indices.
            targets: Optional (B, L) next-token targets with ignore_index=-100 for masked positions.
            position_ids: Optional (B, L) explicit positions.
            cache: Optional pre-allocated InferenceCache for fast autoregressive decoding.
            doc_ids: Optional (B, L) integer document IDs for packed-sequence attention isolation.
            attention_mask: Optional (B, L_k) padding mask (1=valid, 0=pad).
            track_max_logits: Track per-head max attention logits (needed for MuonClip QK-Clip).
            loss_reduction: 'mean' or 'sum' (use 'sum' with num_valid_tokens for exact masked grad accum).
        """
        x = self._embed(input_ids)

        for idx, layer in enumerate(self.layers):
            layer_cache = cache.layers[idx] if cache is not None else None
            if self.activation_checkpointing and self.training and cache is None:
                x = checkpoint(
                    layer,
                    x,
                    self.rope,
                    position_ids,
                    None,
                    doc_ids,
                    attention_mask,
                    track_max_logits,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x,
                    rope=self.rope,
                    position_ids=position_ids,
                    kv_cache=layer_cache,
                    doc_ids=doc_ids,
                    attention_mask=attention_mask,
                    track_max_logits=track_max_logits,
                )

        hidden = self.final_norm(x)
        logits = self._project_logits(hidden)

        if targets is None:
            return LMOutput(logits=logits, hidden_states=hidden)

        # Mandatory FP32 cross-entropy + Z-loss computation for numerical stability
        flat_logits = logits.float().reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        valid_mask = flat_targets != -100
        num_valid = int(valid_mask.sum().item())

        if num_valid == 0:
            zero = logits.sum() * 0.0
            return LMOutput(
                logits=logits,
                hidden_states=hidden,
                loss=zero,
                ce_loss=zero,
                z_loss=zero,
                mtp_loss=zero,
                num_valid_tokens=0,
            )

        valid_logits = flat_logits[valid_mask]
        valid_targets = flat_targets[valid_mask]

        if loss_reduction == "sum":
            ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction="sum")
        else:
            ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction="mean")

        if self.cfg.z_loss_weight > 0.0:
            log_z = torch.logsumexp(valid_logits, dim=-1)
            z_raw = log_z.pow(2).sum() if loss_reduction == "sum" else log_z.pow(2).mean()
            z_loss = self.cfg.z_loss_weight * z_raw
        else:
            z_loss = ce_loss.new_zeros(())

        total_loss = ce_loss + z_loss
        mtp_loss_total = ce_loss.new_zeros(())

        # Multi-Token Prediction (MTP) auxiliary heads
        if len(self.mtp_modules) > 0 and targets.shape[1] > 1:
            prev_h = hidden
            mtp_terms = []
            for depth_idx, mtp_mod in enumerate(self.mtp_modules):
                shift = depth_idx + 1
                if targets.shape[1] <= shift:
                    break
                # Align: hidden[:, :-shift] + emb(targets[:, :-shift]) predicts targets[:, shift:]
                src_h = prev_h[:, :-1, :]
                tok_in = targets[:, depth_idx : targets.shape[1] - 1].clamp(min=0)
                tok_target = targets[:, shift:]
                # Mask out positions where either the bridge token or future target was masked (-100)
                bridge_invalid = targets[:, depth_idx : targets.shape[1] - 1] == -100
                tok_target = tok_target.masked_fill(bridge_invalid, -100)
                if doc_ids is not None:
                    # Never predict across packed document boundaries!
                    cross_doc = doc_ids[:, :-shift] != doc_ids[:, shift:]
                    tok_target = tok_target.masked_fill(cross_doc, -100)
                    mtp_doc = doc_ids[:, :-shift]
                else:
                    mtp_doc = None
                mtp_attn_mask = attention_mask[:, :-shift] if attention_mask is not None else None

                mtp_h = mtp_mod(
                    prev_hidden=src_h,
                    next_token_emb=self._embed(tok_in),
                    rope=self.rope,
                    doc_ids=mtp_doc,
                    attention_mask=mtp_attn_mask,
                )
                prev_h = mtp_h
                mtp_logits = self._project_logits(mtp_h).float()
                mtp_flat_t = tok_target.reshape(-1)
                mtp_valid = mtp_flat_t != -100
                if mtp_valid.any():
                    mtp_flat_l = mtp_logits.reshape(-1, mtp_logits.shape[-1])[mtp_valid]
                    mtp_flat_y = mtp_flat_t[mtp_valid]
                    if loss_reduction == "sum":
                        mtp_terms.append(F.cross_entropy(mtp_flat_l, mtp_flat_y, reduction="sum"))
                    else:
                        mtp_terms.append(F.cross_entropy(mtp_flat_l, mtp_flat_y, reduction="mean"))
            if mtp_terms:
                mtp_loss_total = (self.cfg.mtp_loss_weight / len(mtp_terms)) * sum(mtp_terms)
                total_loss = total_loss + mtp_loss_total

        return LMOutput(
            logits=logits,
            hidden_states=hidden,
            loss=total_loss,
            ce_loss=ce_loss,
            z_loss=z_loss,
            mtp_loss=mtp_loss_total,
            num_valid_tokens=num_valid,
        )

    @torch.inference_mode()
    def predict_mtp_draft(
        self, last_hidden: torch.Tensor, first_token: torch.Tensor
    ) -> list[torch.Tensor]:
        """Generate greedy draft tokens from MTP heads for built-in speculative decoding.

        Args:
            last_hidden: (B, 1, d_model) final hidden state from target forward pass.
            first_token: (B, 1) token already sampled from target logits.
        Returns:
            List of up to `mtp_depth` draft token tensors of shape (B, 1).
        """
        drafts: list[torch.Tensor] = []
        prev_h = last_hidden
        cur_tok = first_token
        for mtp_mod in self.mtp_modules:
            h = mtp_mod(
                prev_hidden=prev_h,
                next_token_emb=self._embed(cur_tok),
                rope=self.rope,
            )
            logits = self._project_logits(h)
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            drafts.append(next_tok)
            prev_h = h
            cur_tok = next_tok
        return drafts

    @torch.no_grad()
    def update_moe_biases(self) -> None:
        """Update auxiliary-loss-free dynamic router biases across all MoE layers."""
        for layer in self.layers:
            if isinstance(layer.ffn, DeepSeekMoE):
                layer.ffn.update_load_balance_bias()

    def collect_telemetry(self) -> dict[str, float]:
        """Collect max attention logit and MoE load balance stats from the latest forward pass."""
        stats: dict[str, float] = {}
        max_logits = []
        moe_entropies = []
        moe_ratios = []
        for layer in self.layers:
            ml = getattr(layer.attn, "last_max_logits", None)
            if ml is not None:
                max_logits.append(float(ml.max().item()))
            if isinstance(layer.ffn, DeepSeekMoE):
                moe_entropies.append(layer.ffn.last_load_entropy)
                moe_ratios.append(layer.ffn.last_max_load_ratio)
        if max_logits:
            stats["max_attn_logit"] = max(max_logits)
        if moe_entropies:
            stats["moe_load_entropy"] = sum(moe_entropies) / len(moe_entropies)
            stats["moe_max_load_ratio"] = max(moe_ratios)
        return stats

    def count_parameters(self) -> dict[str, int]:
        """Return total, active (accounting for MoE top-k sparsity), and non-embedding param counts."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()

        embed_params = self.embed_tokens.weight.numel()
        if not self.cfg.tie_word_embeddings:
            embed_params += self.lm_head.weight.numel()

        inactive = 0
        for layer in self.layers:
            if isinstance(layer.ffn, DeepSeekMoE):
                n_inactive = layer.ffn.n_routed_experts - layer.ffn.top_k
                if n_inactive > 0 and len(layer.ffn.experts) > 0:
                    one_expert = sum(p.numel() for p in layer.ffn.experts[0].parameters())
                    inactive += n_inactive * one_expert

        return {
            "total": total,
            "active": total - inactive,
            "non_embedding": total - embed_params,
        }


def build_model(cfg: ModelConfig) -> NovaLM:
    """Factory entrypoint to build NovaLM from a ModelConfig."""
    return NovaLM(cfg)
