"""Strongly-typed configuration schemas with YAML loading and CLI dot-override support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml


def round_up_multiple(value: int, multiple: int = 64) -> int:
    """Round an integer up to the nearest multiple for Tensor Core alignment."""
    return ((value + multiple - 1) // multiple) * multiple


@dataclass
class ModelConfig:
    """Composable 2026 Transformer / Hybrid / MoE model configuration."""

    # Core dimensions
    vocab_size: int = 32000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int | None = None
    d_ff: int | None = None
    max_seq_len: int = 1024

    # Attention mode: "gqa", "mla", or "hybrid_linear" (3:1 Gated DeltaNet + global)
    attention_type: Literal["gqa", "mla", "hybrid_linear"] = "gqa"
    qk_norm: bool = True
    # Sliding-window attention: e.g. sliding_window=512 with swa_ratio=5 => 5 local : 1 global
    sliding_window: int | None = None
    swa_ratio: int = 0  # 0 disables SWA interleaving; e.g. 5 => layers where (i+1)%6 != 0 use SWA
    # Hybrid linear attention ratio (used when attention_type == "hybrid_linear")
    linear_attn_ratio: int = 3  # 3 linear (Gated DeltaNet-2) layers per 1 full attention layer
    linear_conv_kernel: int = 4
    # Multi-Head Latent Attention (MLA, DeepSeek-V3/V4 style)
    mla_kv_lora_rank: int = 64
    mla_q_lora_rank: int = 96
    mla_qk_rope_head_dim: int = 16
    mla_qk_nope_head_dim: int = 16
    mla_v_head_dim: int = 32

    # Positional encoding (RoPE / YaRN / NoPE hybrid)
    rope_theta: float = 500000.0
    rope_scaling_type: Literal["none", "linear", "yarn"] = "none"
    rope_scaling_factor: float = 1.0
    rope_orig_max_seq_len: int = 1024
    nope_layer_interval: int = 0  # e.g. 4 => every 4th layer uses NoPE (SmolLM3 / iRoPE style)

    # Normalization & residuals
    norm_type: Literal["rmsnorm", "single_scale_rmsnorm"] = "rmsnorm"
    norm_eps: float = 1e-6
    sandwich_norm: bool = False  # Gemma 3/4 Pre+Post RMSNorm around attn & FFN

    # Mixture of Experts (DeepSeek-V3 Auxiliary-Loss-Free style)
    use_moe: bool = False
    moe_layer_stride: int = 1  # 1 = every layer (after first_k_dense), 2 = alternating (Llama 4)
    moe_first_k_dense: int = 1
    n_routed_experts: int = 8
    moe_top_k: int = 2
    n_shared_experts: int = 1
    moe_expert_d_ff: int | None = None
    moe_aux_free_gamma: float = 1e-3  # Dynamic router bias step (DeepSeek-V3 aux-loss-free)

    # Embeddings, stability & Multi-Token Prediction (MTP)
    tie_word_embeddings: bool = True
    init_std: float = 0.02
    dropout: float = 0.0
    attn_logit_softcap: float | None = None
    final_logit_softcap: float | None = None
    z_loss_weight: float = 1e-4
    mtp_depth: int = 0  # 0 disables; 1..2 adds Multi-Token Prediction heads (DeepSeek-V3)
    mtp_loss_weight: float = 0.3
    # Outlier-Safe Pre-Training (OSP) optional learnable embedding projection
    use_osp_embed_proj: bool = False
    # Vocab padding multiple for Tensor Core efficiency
    pad_vocab_multiple: int = 64

    def __post_init__(self) -> None:
        if self.pad_vocab_multiple > 1:
            self.vocab_size = round_up_multiple(self.vocab_size, self.pad_vocab_multiple)
        if self.head_dim is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
                )
            self.head_dim = self.d_model // self.n_heads
        if self.attention_type != "mla" and self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")
        if self.d_ff is None:
            # Standard SwiGLU expansion: ~8/3 * d_model rounded to multiple of 64
            self.d_ff = round_up_multiple(int(self.d_model * 8 / 3), 64)
        if self.use_moe and self.moe_expert_d_ff is None:
            self.moe_expert_d_ff = round_up_multiple(max(64, self.d_ff // 2), 64)
        if self.use_moe and self.moe_top_k > self.n_routed_experts:
            raise ValueError(
                f"moe_top_k ({self.moe_top_k}) cannot exceed n_routed_experts ({self.n_routed_experts})"
            )

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.use_moe or layer_idx < self.moe_first_k_dense:
            return False
        return (layer_idx - self.moe_first_k_dense) % self.moe_layer_stride == 0

    def is_linear_attn_layer(self, layer_idx: int) -> bool:
        if self.attention_type != "hybrid_linear":
            return False
        period = self.linear_attn_ratio + 1
        return (layer_idx + 1) % period != 0

    def layer_sliding_window(self, layer_idx: int) -> int | None:
        if self.sliding_window is None or self.swa_ratio <= 0:
            return self.sliding_window
        period = self.swa_ratio + 1
        return self.sliding_window if (layer_idx + 1) % period != 0 else None

    def layer_uses_rope(self, layer_idx: int) -> bool:
        if self.nope_layer_interval <= 0:
            return True
        return (layer_idx + 1) % self.nope_layer_interval != 0


@dataclass
class OptimConfig:
    """Optimizer and learning rate schedule configuration."""

    optimizer: Literal["adamw", "muon", "muonclip"] = "muonclip"
    lr: float = 3e-4
    muon_lr: float = 2e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    # MuonClip QK-Clip threshold (Kimi K2): rescales per-head Q/K weights when max logit > tau
    qk_clip_tau: float = 100.0
    grad_clip: float = 1.0
    lr_schedule: Literal["wsd", "cosine"] = "wsd"
    warmup_steps: int = 50
    decay_steps: int = 100  # Used by WSD schedule at the tail of training


@dataclass
class TrainConfig:
    """Pre-training / mid-training loop configuration."""

    output_dir: str = "runs/pretrain"
    seed: int = 42
    max_steps: int = 200
    micro_batch_size: int = 4
    grad_accum_steps: int = 4
    seq_len: int = 256
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    compile_model: bool = False
    activation_checkpointing: bool = False
    distributed: Literal["none", "ddp", "fsdp2"] = "none"
    eval_interval: int = 50
    eval_steps: int = 10
    log_interval: int = 10
    save_interval: int = 100
    spike_loss_multiplier: float = 3.0  # Flag / guard loss spikes
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class PostTrainConfig:
    """Configuration for SFT, DPO/SimPO/IPO, and RLVR (GRPO/Dr.GRPO/DAPO/GSPO)."""

    stage: Literal["sft", "dpo", "simpo", "ipo", "grpo", "dr_grpo", "dapo", "gspo"] = "sft"
    output_dir: str = "runs/post_train"
    seed: int = 42
    max_steps: int = 50
    micro_batch_size: int = 2
    grad_accum_steps: int = 2
    max_seq_len: int = 512
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: Literal["fp32", "bf16"] = "fp32"
    # LoRA / DoRA
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    use_dora: bool = False
    # DPO / SimPO
    dpo_beta: float = 0.1
    simpo_gamma: float = 0.5
    dpo_nll_weight: float = 0.0
    # RLVR (GRPO / Dr.GRPO / DAPO / GSPO)
    rl_group_size: int = 4
    rl_max_new_tokens: int = 64
    rl_temperature: float = 0.8
    rl_clip_low: float = 0.2
    rl_clip_high: float = 0.28  # DAPO asymmetric Clip-Higher preserves exploration entropy
    rl_kl_beta: float = 0.0  # DAPO / Dr.GRPO default: KL=0 for verifiable reasoning
    rl_dynamic_sampling: bool = True  # Drop all-pass / all-fail zero-advantage prompts
    rl_overlong_penalty: float = 0.2
    rl_overlong_buffer: int = 16
    rl_ppo_epochs: int = 2


def _coerce_value(val: str) -> Any:
    """Coerce a CLI override string into Python bool/int/float/None/str."""
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively instantiate nested dataclasses from a dict with strict key validation."""
    valid_fields = {f.name: f for f in fields(cls)}
    unknown = set(data.keys()) - set(valid_fields.keys())
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    nested_map = {
        "model": ModelConfig,
        "optim": OptimConfig,
    }
    for k, v in data.items():
        if k in nested_map and isinstance(v, dict):
            kwargs[k] = _dataclass_from_dict(nested_map[k], v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-notation CLI overrides like ['model.d_model=512', 'optim.lr=1e-3']."""
    out = dict(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must have key=value format, got: {item!r}")
        key_path, val_str = item.split("=", 1)
        parts = key_path.split(".")
        cursor = out
        for p in parts[:-1]:
            if p not in cursor or not isinstance(cursor[p], dict):
                cursor[p] = {}
            cursor = cursor[p]
        cursor[parts[-1]] = _coerce_value(val_str)
    return out


def load_config(cls: type, path: str | Path | None = None, overrides: list[str] | None = None) -> Any:
    """Load a typed config from YAML (optional) and apply CLI dot-overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {path} must contain a YAML mapping")
        raw = loaded
    if overrides:
        raw = apply_overrides(raw, overrides)
    return _dataclass_from_dict(cls, raw)


def save_config(cfg: Any, path: str | Path) -> None:
    """Serialize a dataclass config cleanly to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
