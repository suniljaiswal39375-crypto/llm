#!/usr/bin/env python3
"""Production Preference Alignment CLI: DPO / SimPO / IPO on any JSONL preference dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from nova_llm.config import PostTrainConfig
from nova_llm.data import load_dpo_jsonl
from nova_llm.inference import export_safetensors, load_safetensors_model
from nova_llm.post_training import DPOTrainer
from nova_llm.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Preference Alignment (DPO/SimPO/IPO) on JSONL datasets")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True, help="JSONL file with prompt, chosen, rejected")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=["dpo", "simpo", "ipo"], default="dpo")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nll-weight", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = load_safetensors_model(args.model_dir, device="cpu")
    tokenizer = ByteBPETokenizer.load(Path(args.model_dir) / "tokenizer.json")
    examples = load_dpo_jsonl(args.data)
    if not examples:
        raise SystemExit(f"No preference pairs loaded from {args.data}")

    cfg = PostTrainConfig(
        stage=args.method,  # type: ignore[arg-type]
        output_dir=args.out_dir,
        max_steps=args.steps,
        micro_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        dpo_beta=args.beta,
        dpo_nll_weight=args.nll_weight,
    )
    trainer = DPOTrainer(model, tokenizer, examples, cfg, device=args.device)
    hist = trainer.fit()
    export_safetensors(trainer.model, args.out_dir, tokenizer=tokenizer)
    print(f"{args.method.upper()} complete ({len(examples)} pairs): loss {hist[0]['pref_loss']:.4f} -> {hist[-1]['pref_loss']:.4f}, win_acc={hist[-1]['win_acc']:.2f}")
    print(f"Exported to {args.out_dir}")


if __name__ == "__main__":
    main()
