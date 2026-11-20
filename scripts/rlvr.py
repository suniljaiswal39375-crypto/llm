#!/usr/bin/env python3
"""Production RLVR CLI: GRPO / Dr. GRPO / DAPO / GSPO reasoning training on verifiable JSONL tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from nova_llm.agent import WebSearchRLVRVerifier
from nova_llm.config import PostTrainConfig
from nova_llm.data import load_rlvr_jsonl
from nova_llm.inference import export_safetensors, load_safetensors_model
from nova_llm.post_training import CompositeVerifier, RLVRTrainer
from nova_llm.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="RLVR Reasoning Trainer (GRPO / Dr.GRPO / DAPO / GSPO)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True, help="JSONL verifiable prompts (math, code, constraints, or search)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--algo", choices=["grpo", "dr_grpo", "dapo", "gspo"], default="dapo")
    parser.add_argument("--verifier", choices=["composite", "web_search"], default="composite")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = load_safetensors_model(args.model_dir, device="cpu")
    tokenizer = ByteBPETokenizer.load(Path(args.model_dir) / "tokenizer.json")
    prompts = load_rlvr_jsonl(args.data)
    if not prompts:
        raise SystemExit(f"No RLVR prompts loaded from {args.data}")

    cfg = PostTrainConfig(
        stage=args.algo,  # type: ignore[arg-type]
        output_dir=args.out_dir,
        max_steps=args.steps,
        lr=args.lr,
        rl_group_size=args.group_size,
        rl_max_new_tokens=args.max_new_tokens,
        rl_kl_beta=args.kl_beta,
    )
    verifier = WebSearchRLVRVerifier() if args.verifier == "web_search" else CompositeVerifier()
    trainer = RLVRTrainer(model, tokenizer, prompts, cfg, verifier=verifier, device=args.device)
    hist = trainer.fit()
    export_safetensors(trainer.model, args.out_dir, tokenizer=tokenizer)
    print(f"RLVR ({args.algo.upper()}) complete ({len(prompts)} prompts, {len(hist)} steps). Exported to {args.out_dir}")


if __name__ == "__main__":
    main()
