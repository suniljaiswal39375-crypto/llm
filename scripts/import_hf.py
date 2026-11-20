#!/usr/bin/env python3
"""Import a real Hugging Face Llama-3 / Qwen2.5 / Qwen3 / SmolLM2 / Mistral checkpoint into NovaLM."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from nova_llm.inference import export_safetensors
from nova_llm.models import load_hf_pretrained_into_nova


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Hugging Face checkpoint to native NovaLM SafeTensors")
    parser.add_argument("--hf-dir", required=True, help="Local directory containing HF config.json & *.safetensors")
    parser.add_argument("--out-dir", required=True, help="Destination directory for NovaLM export")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = load_hf_pretrained_into_nova(args.hf_dir, device=args.device)
    out_dir = export_safetensors(model, args.out_dir)

    # Preserve HF tokenizer files alongside converted weights if present
    hf_path = Path(args.hf_dir)
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = hf_path / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    print(f"Converted HF checkpoint -> NovaLM at {out_dir}")
    print("Parameters:", model.count_parameters())
    print(
        f"Config: d_model={model.cfg.d_model}, layers={model.cfg.n_layers}, "
        f"heads={model.cfg.n_heads}/{model.cfg.n_kv_heads}, qk_norm={model.cfg.qk_norm}, "
        f"attn_bias={model.cfg.attention_bias}, vocab={model.cfg.vocab_size}"
    )


if __name__ == "__main__":
    main()
