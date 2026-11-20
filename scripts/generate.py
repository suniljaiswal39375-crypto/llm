"""CLI entrypoint for interactive or single-prompt generation (standard or speculative)."""

from __future__ import annotations

import argparse

import torch

from nova_llm.inference import SamplingParams, generate, load_safetensors_model, speculative_generate
from nova_llm.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from an exported Nova-LLM model")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--speculative", choices=["none", "ngram", "mtp"], default="none")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = load_safetensors_model(args.model_dir, device=args.device)
    tok = ByteBPETokenizer.load(f"{args.model_dir}/tokenizer.json")
    prompt_ids = torch.tensor([tok.encode(args.prompt, add_bos=True)], dtype=torch.long)
    params = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        stop_token_ids=[tok.eos_id],
    )
    if args.speculative == "none":
        res = generate(model, prompt_ids, params=params, tokenizer=tok)
    else:
        res = speculative_generate(model, prompt_ids, params=params, method=args.speculative, tokenizer=tok)
    print(res.text)
    print(f"\n[tok/s={res.tokens_per_sec:.1f}, target_passes={res.target_forward_passes}]")


if __name__ == "__main__":
    main()
