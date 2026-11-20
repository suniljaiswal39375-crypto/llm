"""Import real Hugging Face open-weight LLMs (Llama-3, Qwen2.5, Qwen3, SmolLM2, Mistral) into NovaLM.

Converts Hugging Face `config.json` + SafeTensors weights directly into native `NovaLM`
architecture so you can run, LoRA/DoRA fine-tune, DPO/RLVR align, quantize, and serve
real pretrained billion-parameter models using the full Nova-LLM stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from nova_llm.config import ModelConfig
from nova_llm.models.transformer import NovaLM

SUPPORTED_HF_MODEL_TYPES = {"llama", "qwen2", "qwen3", "mistral"}


def hf_config_to_nova_config(hf_cfg: dict[str, Any]) -> ModelConfig:
    """Translate a Hugging Face `config.json` dictionary into a validated `ModelConfig`."""
    model_type = str(hf_cfg.get("model_type", "llama")).lower()
    if model_type not in SUPPORTED_HF_MODEL_TYPES:
        raise ValueError(
            f"Unsupported Hugging Face model_type={model_type!r}. "
            f"Supported: {sorted(SUPPORTED_HF_MODEL_TYPES)}"
        )

    vocab_size = int(hf_cfg["vocab_size"])
    d_model = int(hf_cfg["hidden_size"])
    n_layers = int(hf_cfg["num_hidden_layers"])
    n_heads = int(hf_cfg["num_attention_heads"])
    n_kv_heads = int(hf_cfg.get("num_key_value_heads", n_heads))
    head_dim = int(hf_cfg.get("head_dim", d_model // n_heads))
    ffn_hidden_dim = int(hf_cfg["intermediate_size"])
    max_seq_len = int(hf_cfg.get("max_position_embeddings", 4096))
    rope_theta = float(hf_cfg.get("rope_theta", 10000.0))
    rms_norm_eps = float(hf_cfg.get("rms_norm_eps", 1e-6))
    tie_word_embeddings = bool(hf_cfg.get("tie_word_embeddings", False))

    # Qwen2/2.5 uses QKV biases; Qwen3 uses per-head QK-Norm without QKV bias
    attention_bias = bool(hf_cfg.get("attention_bias", model_type == "qwen2"))
    qk_norm = model_type == "qwen3" or bool(hf_cfg.get("qk_norm", False))
    sliding_window = hf_cfg.get("sliding_window")
    if sliding_window is not None:
        sliding_window = int(sliding_window)

    # RoPE scaling (e.g. YaRN or linear)
    rope_scaling = hf_cfg.get("rope_scaling") or {}
    rope_scaling_type = "none"
    rope_scaling_factor = 1.0
    rope_orig_max_seq_len = max_seq_len
    if isinstance(rope_scaling, dict) and rope_scaling:
        rt = str(rope_scaling.get("type", rope_scaling.get("rope_type", "none"))).lower()
        if rt in ("yarn", "linear"):
            rope_scaling_type = rt
            rope_scaling_factor = float(rope_scaling.get("factor", 1.0))
            rope_orig_max_seq_len = int(
                rope_scaling.get("original_max_position_embeddings", max_seq_len)
            )

    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        d_ff=ffn_hidden_dim,
        max_seq_len=max_seq_len,
        attention_type="gqa",
        attention_bias=attention_bias,
        qk_norm=qk_norm,
        sliding_window=sliding_window,
        rope_theta=rope_theta,
        rope_scaling_type=rope_scaling_type,  # type: ignore[arg-type]
        rope_scaling_factor=rope_scaling_factor,
        rope_orig_max_seq_len=rope_orig_max_seq_len,
        norm_eps=rms_norm_eps,
        tie_word_embeddings=tie_word_embeddings,
        dropout=0.0,
        pad_vocab_multiple=1,
    )


def map_hf_state_dict_to_nova(
    hf_sd: dict[str, torch.Tensor],
    cfg: ModelConfig,
) -> dict[str, torch.Tensor]:
    """Map Hugging Face Llama/Qwen2/Qwen3/Mistral weight keys to exact `NovaLM` keys."""
    nova_sd: dict[str, torch.Tensor] = {}

    if "model.embed_tokens.weight" not in hf_sd:
        raise KeyError("Missing 'model.embed_tokens.weight' in Hugging Face state_dict")
    nova_sd["embed_tokens.weight"] = hf_sd["model.embed_tokens.weight"]
    nova_sd["final_norm.weight"] = hf_sd["model.norm.weight"]

    if "lm_head.weight" in hf_sd:
        nova_sd["lm_head.weight"] = hf_sd["lm_head.weight"]
    elif cfg.tie_word_embeddings:
        nova_sd["lm_head.weight"] = hf_sd["model.embed_tokens.weight"]
    else:
        raise KeyError("Missing 'lm_head.weight' for untied Hugging Face model")

    for i in range(cfg.n_layers):
        p = f"model.layers.{i}"
        q = f"layers.{i}"

        nova_sd[f"{q}.attn_norm.weight"] = hf_sd[f"{p}.input_layernorm.weight"]
        nova_sd[f"{q}.ffn_norm.weight"] = hf_sd[f"{p}.post_attention_layernorm.weight"]

        nova_sd[f"{q}.attn.wq.weight"] = hf_sd[f"{p}.self_attn.q_proj.weight"]
        nova_sd[f"{q}.attn.wk.weight"] = hf_sd[f"{p}.self_attn.k_proj.weight"]
        nova_sd[f"{q}.attn.wv.weight"] = hf_sd[f"{p}.self_attn.v_proj.weight"]
        nova_sd[f"{q}.attn.wo.weight"] = hf_sd[f"{p}.self_attn.o_proj.weight"]

        if cfg.attention_bias:
            for hf_name, nova_name in (("q_proj", "wq"), ("k_proj", "wk"), ("v_proj", "wv")):
                b_key = f"{p}.self_attn.{hf_name}.bias"
                if b_key in hf_sd:
                    nova_sd[f"{q}.attn.{nova_name}.bias"] = hf_sd[b_key]

        if cfg.qk_norm:
            if f"{p}.self_attn.q_norm.weight" in hf_sd:
                nova_sd[f"{q}.attn.q_norm.weight"] = hf_sd[f"{p}.self_attn.q_norm.weight"]
            if f"{p}.self_attn.k_norm.weight" in hf_sd:
                nova_sd[f"{q}.attn.k_norm.weight"] = hf_sd[f"{p}.self_attn.k_norm.weight"]

        nova_sd[f"{q}.ffn.w_gate.weight"] = hf_sd[f"{p}.mlp.gate_proj.weight"]
        nova_sd[f"{q}.ffn.w_up.weight"] = hf_sd[f"{p}.mlp.up_proj.weight"]
        nova_sd[f"{q}.ffn.w_down.weight"] = hf_sd[f"{p}.mlp.down_proj.weight"]

    return nova_sd


def _load_hf_safetensors_dir(model_dir: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    single = model_dir / "model.safetensors"
    if single.exists():
        return load_file(str(single), device=device)

    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        idx = json.loads(index_file.read_text(encoding="utf-8"))
        shard_files = sorted(set(idx["weight_map"].values()))
        merged: dict[str, torch.Tensor] = {}
        for sf in shard_files:
            merged.update(load_file(str(model_dir / sf), device=device))
        return merged

    shards = sorted(model_dir.glob("*.safetensors"))
    if shards:
        merged = {}
        for sf in shards:
            merged.update(load_file(str(sf), device=device))
        return merged

    raise FileNotFoundError(f"No .safetensors files found in {model_dir}")


def load_hf_pretrained_into_nova(
    hf_model_dir: str | Path,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> NovaLM:
    """Load a local Hugging Face Llama/Qwen2/Qwen3/SmolLM/Mistral checkpoint into `NovaLM`."""
    model_dir = Path(hf_model_dir)
    hf_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    nova_cfg = hf_config_to_nova_config(hf_cfg)

    hf_sd = _load_hf_safetensors_dir(model_dir, device=device)
    # Auto-detect QK-Norm or QKV bias directly from actual weight keys if config omitted them
    if "model.layers.0.self_attn.q_norm.weight" in hf_sd:
        nova_cfg.qk_norm = True
    if "model.layers.0.self_attn.q_proj.bias" in hf_sd:
        nova_cfg.attention_bias = True

    nova_sd = map_hf_state_dict_to_nova(hf_sd, nova_cfg)
    model = NovaLM(nova_cfg)
    model.load_state_dict(nova_sd, strict=True)
    if nova_cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device)
