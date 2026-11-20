"""Hugging Face SafeTensors + config.json export & reload with exact logit parity checks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from nova_llm.config import ModelConfig
from nova_llm.models.transformer import NovaLM
from nova_llm.tokenizer.bpe import ByteBPETokenizer


def export_safetensors(
    model: NovaLM,
    output_dir: str | Path,
    tokenizer: ByteBPETokenizer | None = None,
) -> Path:
    """Export NovaLM weights to `model.safetensors` + `config.json` (+ `tokenizer.json`)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # SafeTensors forbids shared tensor memory; handle tied lm_head cleanly
    raw_sd = model.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if model.cfg.tie_word_embeddings and k == "lm_head.weight":
            continue
        tensors[k] = v.detach().cpu().contiguous()

    save_file(tensors, str(out / "model.safetensors"))

    hf_cfg = asdict(model.cfg)
    hf_cfg["architectures"] = ["NovaForCausalLM"]
    hf_cfg["model_type"] = "nova_llm"
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_cfg, f, indent=2)

    if tokenizer is not None:
        tokenizer.save(out / "tokenizer.json")
    return out


def load_safetensors_model(model_dir: str | Path, device: str | torch.device = "cpu") -> NovaLM:
    """Reload NovaLM from a SafeTensors export directory."""
    d = Path(model_dir)
    with open(d / "config.json", "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    raw_cfg.pop("architectures", None)
    raw_cfg.pop("model_type", None)
    cfg = ModelConfig(**raw_cfg)
    model = NovaLM(cfg)
    tensors = load_file(str(d / "model.safetensors"), device=str(device))
    if cfg.tie_word_embeddings and "lm_head.weight" not in tensors:
        tensors["lm_head.weight"] = tensors["embed_tokens.weight"]
    model.load_state_dict(tensors, strict=True)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    return model.to(device)
