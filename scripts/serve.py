"""CLI entrypoint to launch the OpenAI-compatible Nova-LLM FastAPI inference server."""

from __future__ import annotations

import argparse

import uvicorn

from nova_llm.inference import create_app, load_safetensors_model
from nova_llm.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Nova-LLM via OpenAI-compatible FastAPI")
    parser.add_argument("--model-dir", type=str, required=True, help="Exported SafeTensors directory")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = load_safetensors_model(args.model_dir, device=args.device)
    tokenizer = ByteBPETokenizer.load(f"{args.model_dir}/tokenizer.json")
    app = create_app(model, tokenizer)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
