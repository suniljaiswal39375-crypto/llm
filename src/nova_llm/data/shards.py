"""Memory-mapped binary token shards with exact per-token document boundary tracking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from nova_llm.tokenizer.bpe import ByteBPETokenizer


@dataclass
class ShardMetadata:
    num_tokens: int
    num_docs: int
    vocab_size: int
    total_bytes: int
    dtype: str
    tokens_sha256: str


def write_token_shard(
    docs: list[str],
    tokenizer: ByteBPETokenizer,
    output_prefix: str | Path,
    add_bos: bool = True,
    add_eos: bool = True,
) -> ShardMetadata:
    """Tokenize documents into memory-mapped `<prefix>.tokens.bin`, `<prefix>.docs.bin`, and `<prefix>.meta.json`."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    token_list: list[int] = []
    doc_id_list: list[int] = []
    total_bytes = 0

    for doc_idx, text in enumerate(docs):
        if not text:
            continue
        ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos, allow_special=True)
        if not ids:
            continue
        token_list.extend(ids)
        doc_id_list.extend([doc_idx] * len(ids))
        total_bytes += len(text.encode("utf-8"))

    tok_arr = np.asarray(token_list, dtype=np.uint32)
    doc_arr = np.asarray(doc_id_list, dtype=np.int32)

    tok_path = Path(f"{prefix}.tokens.bin")
    doc_path = Path(f"{prefix}.docs.bin")
    meta_path = Path(f"{prefix}.meta.json")

    tok_arr.tofile(tok_path)
    doc_arr.tofile(doc_path)

    sha = hashlib.sha256(tok_arr.tobytes()).hexdigest()
    meta = ShardMetadata(
        num_tokens=int(tok_arr.size),
        num_docs=len(docs),
        vocab_size=tokenizer.vocab_size,
        total_bytes=total_bytes,
        dtype="uint32",
        tokens_sha256=sha,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)
    return meta


class TokenShardReader:
    """Zero-copy memory-mapped reader for binary token + document-ID shards."""

    def __init__(self, prefix: str | Path) -> None:
        self.prefix = Path(prefix)
        with open(f"{self.prefix}.meta.json", "r", encoding="utf-8") as f:
            self.meta = ShardMetadata(**json.load(f))
        self.tokens = np.memmap(f"{self.prefix}.tokens.bin", dtype=np.uint32, mode="r")
        self.doc_ids = np.memmap(f"{self.prefix}.docs.bin", dtype=np.int32, mode="r")
        if len(self.tokens) != self.meta.num_tokens or len(self.doc_ids) != self.meta.num_tokens:
            raise ValueError(f"Corrupt shard {self.prefix}: length mismatch vs metadata")

    def __len__(self) -> int:
        return self.meta.num_tokens

    @property
    def bytes_per_token(self) -> float:
        return self.meta.total_bytes / max(1, self.meta.num_tokens)
