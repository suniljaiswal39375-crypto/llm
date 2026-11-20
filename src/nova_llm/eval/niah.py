"""Needle-in-a-Haystack (NIAH) long-context retrieval evaluation across depths & context lengths."""

from __future__ import annotations

from typing import Any

import torch

from nova_llm.inference.generator import generate
from nova_llm.inference.sampler import SamplingParams
from nova_llm.models.transformer import NovaLM
from nova_llm.tokenizer.bpe import ByteBPETokenizer


_FILLER = "The observatory recorded steady atmospheric measurements across every sector today. "


def evaluate_niah(
    model: NovaLM,
    tokenizer: ByteBPETokenizer,
    context_lengths: list[int] | None = None,
    depths: list[float] | None = None,
    passkey: str = "74921",
) -> dict[str, Any]:
    """Run Needle-in-a-Haystack passkey retrieval across context lengths and insertion depths."""
    max_cap = model.cfg.max_seq_len
    lengths = context_lengths or [min(128, max_cap - 32), min(256, max_cap - 32)]
    depths = depths or [0.1, 0.5, 0.9]

    needle = f"The secret verification passkey is {passkey}. "
    question = "\nQuestion: What is the secret verification passkey?\nAnswer: "

    results: list[dict[str, Any]] = []
    hits = 0

    for ctx_len in lengths:
        if ctx_len >= max_cap - 16:
            ctx_len = max(32, max_cap - 24)
        filler_tokens = tokenizer.encode(_FILLER, allow_special=False)
        needle_tokens = tokenizer.encode(needle, allow_special=False)
        q_tokens = tokenizer.encode(question, allow_special=False)
        budget = max(0, ctx_len - len(needle_tokens) - len(q_tokens) - 1)
        reps = (budget // max(1, len(filler_tokens))) + 1
        haystack = (filler_tokens * reps)[:budget]

        for depth in depths:
            ins_pos = int(len(haystack) * max(0.0, min(1.0, depth)))
            prompt_ids = [tokenizer.bos_id] + haystack[:ins_pos] + needle_tokens + haystack[ins_pos:] + q_tokens
            prompt_t = torch.tensor([prompt_ids], dtype=torch.long)
            res = generate(
                model,
                prompt_t,
                params=SamplingParams(max_new_tokens=16, temperature=0.0),
                tokenizer=tokenizer,
                use_cache=True,
            )
            matched = passkey in (res.text or "")
            if matched:
                hits += 1
            results.append(
                {
                    "context_tokens": len(prompt_ids),
                    "depth": depth,
                    "matched": matched,
                    "output_preview": (res.text or "")[:40],
                }
            )

    return {
        "niah_accuracy": hits / max(1, len(results)),
        "niah_hits": hits,
        "niah_total": len(results),
        "grid": results,
    }
