#!/usr/bin/env python3
"""Prepare real pre-training data: stream .txt/.jsonl -> curate -> train SuperBPE -> binary shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova_llm.data import curate_corpus, stream_text_corpus, write_token_shard
from nova_llm.tokenizer import ByteBPETokenizer, evaluate_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate raw corpus, train SuperBPE, and pack binary shards")
    parser.add_argument("--input", nargs="+", required=True, help="Input .txt/.jsonl files or directories")
    parser.add_argument("--out-dir", required=True, help="Output directory for tokenizer and binary shards")
    parser.add_argument("--text-field", default="text", help="JSON field for JSONL inputs")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--superbpe-transition", type=int, default=3072)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--no-fuzzy-dedup", action="store_true")
    parser.add_argument("--tokenizer", default=None, help="Optional existing tokenizer.json to reuse")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_docs = list(stream_text_corpus(args.input, text_field=args.text_field))
    if not raw_docs:
        raise SystemExit("No non-empty documents found in input paths!")

    cleaned, curation_stats = curate_corpus(
        raw_docs,
        min_chars=args.min_chars,
        fuzzy_dedup=not args.no_fuzzy_dedup,
    )
    print("Curation stats:", json.dumps(curation_stats))

    if args.tokenizer:
        tokenizer = ByteBPETokenizer.load(args.tokenizer)
    else:
        trans = args.superbpe_transition if args.superbpe_transition > 0 else None
        tokenizer = ByteBPETokenizer.train(
            cleaned,
            vocab_size=args.vocab_size,
            superbpe_transition_vocab=trans,
            min_pair_freq=2,
        )
        tokenizer.save(out_dir / "tokenizer.json")

    tok_stats = evaluate_tokenizer(tokenizer, cleaned[: min(100, len(cleaned))])
    print("Tokenizer diagnostics:", json.dumps(tok_stats))

    n_val = max(1, int(len(cleaned) * args.val_ratio))
    train_docs = cleaned[n_val:] if len(cleaned) > n_val else cleaned
    val_docs = cleaned[:n_val]

    train_meta = write_token_shard(train_docs, tokenizer, out_dir / "train")
    val_meta = write_token_shard(val_docs, tokenizer, out_dir / "val")
    summary = {
        "curation": curation_stats,
        "tokenizer": tok_stats,
        "train_shard": train_meta,
        "val_shard": val_meta,
    }
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Prepared shards in {out_dir}: train={train_meta['num_tokens']} tok, val={val_meta['num_tokens']} tok")


if __name__ == "__main__":
    main()
