#!/usr/bin/env python3
"""Serve NovaLM OR Llama-3.2-3B (+ LoRA) with OpenAI API & Live Web-Search Chat UI."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from nova_llm.agent import HybridSearchBackend, LiveHTTPWebBackend, TransformersBackend
from nova_llm.inference import create_app, load_safetensors_model
from nova_llm.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve NovaLM or Hugging Face / Llama-3.2-3B with Web Search UI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-dir", help="Native NovaLM exported SafeTensors directory")
    group.add_argument("--hf-model", help="Hugging Face model name or local path (e.g. unsloth/Llama-3.2-3B-Instruct-bnb-4bit)")
    parser.add_argument("--lora-dir", default=None, help="Optional LoRA adapter directory (for --hf-model)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization for --hf-model")
    parser.add_argument("--live-web", action="store_true", help="Enable free live DuckDuckGo + Wikipedia web search")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    search_backend = HybridSearchBackend(live_backend=LiveHTTPWebBackend(enable_free_web=args.live_web))

    if args.hf_model:
        backend = TransformersBackend(
            model_name_or_path=args.hf_model,
            lora_dir=args.lora_dir,
            load_in_4bit=not args.no_4bit,
            model_name=args.hf_model,
        )
        app = create_app(backend=backend, model_name=args.hf_model, search_backend=search_backend)
    else:
        model_dir = Path(args.model_dir)
        model = load_safetensors_model(model_dir, device=args.device)
        tok = ByteBPETokenizer.load(model_dir / "tokenizer.json")
        app = create_app(model, tok, search_backend=search_backend)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
