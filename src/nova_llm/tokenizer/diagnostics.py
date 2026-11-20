"""Tokenizer diagnostics: bytes-per-token, fertility, round-trip lossless UTF-8 check."""

from __future__ import annotations

from typing import Any

from nova_llm.tokenizer.bpe import ByteBPETokenizer


def evaluate_tokenizer(tokenizer: ByteBPETokenizer, texts: list[str]) -> dict[str, Any]:
    """Compute encoding efficiency and verify lossless round-trip on held-out text."""
    total_bytes = 0
    total_tokens = 0
    total_words = 0
    roundtrip_failures = 0

    for text in texts:
        ids = tokenizer.encode(text, allow_special=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != text:
            roundtrip_failures += 1
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
        total_words += max(1, len(text.split()))

    return {
        "vocab_size": tokenizer.vocab_size,
        "superbpe_enabled": tokenizer.superbpe_transition_idx is not None,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": total_bytes / max(1, total_tokens),
        "fertility_tokens_per_word": total_tokens / max(1, total_words),
        "roundtrip_failures": roundtrip_failures,
    }
