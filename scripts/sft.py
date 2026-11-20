#!/usr/bin/env python3
"""Production SFT CLI: full FT or LoRA/DoRA fine-tuning on any JSONL conversation dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from nova_llm.config import PostTrainConfig
from nova_llm.data import load_sft_jsonl
from nova_llm.inference import export_safetensors, load_safetensors_model
from nova_llm.models import merge_and_unload_lora
from nova_llm.post_training import SFTTrainer
from nova_llm.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) on JSONL chat datasets")
    parser.add_argument("--model-dir", required=True, help="Base SafeTensors model directory")
    parser.add_argument("--data", required=True, help="JSONL file (OpenAI messages, ShareGPT, or Alpaca format)")
    parser.add_argument("--out-dir", required=True, help="Output directory for exported SFT model")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA parameter-efficient fine-tuning")
    parser.add_argument("--dora", action="store_true", help="Enable DoRA (weight-decomposed LoRA)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = load_safetensors_model(args.model_dir, device="cpu")
    tokenizer = ByteBPETokenizer.load(Path(args.model_dir) / "tokenizer.json")
    convs = load_sft_jsonl(args.data)
    if not convs:
        raise SystemExit(f"No conversations loaded from {args.data}")

    cfg = PostTrainConfig(
        stage="sft",
        output_dir=args.out_dir,
        max_steps=args.steps,
        micro_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        lr=args.lr,
        precision=args.precision,  # type: ignore[arg-type]
        use_lora=args.lora or args.dora,
        use_dora=args.dora,
        lora_rank=args.lora_rank,
        lora_alpha=float(args.lora_rank * 2),
    )
    trainer = SFTTrainer(model, tokenizer, convs, cfg, device=args.device)
    hist = trainer.fit()
    if cfg.use_lora:
        merge_and_unload_lora(trainer.model)

    export_safetensors(trainer.model, args.out_dir, tokenizer=tokenizer)
    print(f"SFT complete ({len(convs)} convs, {len(hist)} steps): loss {hist[0]['sft_loss']:.4f} -> {hist[-1]['sft_loss']:.4f}")
    print(f"Exported to {args.out_dir}")


if __name__ == "__main__":
    main()
