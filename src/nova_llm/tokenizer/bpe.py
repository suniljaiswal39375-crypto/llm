"""Byte-Level BPE & SuperBPE (COLM 2025, arXiv:2503.13423) Tokenizer from scratch.

Implements:
- Lossless UTF-8 byte-level base vocabulary (256 byte tokens + special tokens)
- Standard BPE pretokenization (regex word/number/punctuation/whitespace splits)
- SuperBPE two-stage curriculum:
  * Stage 1 (vocab 256..t): learn subwords within whitespace-delimited chunks
  * Stage 2 (vocab t..T): lift whitespace pretokenization to learn cross-word superwords
- Exact special-token preservation during encode/decode
- Clean JSON serialization & chat-template tokenization with assistant-only loss masks
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nova_llm.tokenizer.special_tokens import SPECIAL_TOKENS, format_chat_messages

# Pre-tokenization regex: splits words (with optional leading space), digits individually or up to 3, punctuation, newlines
DEFAULT_PRETOK_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def _merge_sequence(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all non-overlapping occurrences of `pair` in `seq` with `new_id`."""
    a, b = pair
    out: list[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class ByteBPETokenizer:
    """Lossless Byte-Level BPE & SuperBPE tokenizer with versioned special tokens."""

    def __init__(
        self,
        merges: list[tuple[int, int]] | None = None,
        special_tokens: list[str] | None = None,
        superbpe_transition_idx: int | None = None,
    ) -> None:
        self.special_tokens: list[str] = list(
            special_tokens if special_tokens is not None else SPECIAL_TOKENS
        )
        self.num_special = len(self.special_tokens)
        # IDs [0 .. num_special - 1]: special tokens
        # IDs [num_special .. num_special + 255]: raw UTF-8 bytes 0..255
        # IDs [num_special + 256 ..]: learned BPE/SuperBPE merges
        self.byte_offset = self.num_special
        self.special_to_id: dict[str, int] = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.id_to_special: dict[int, str] = {i: tok for i, tok in enumerate(self.special_tokens)}

        self.merges: list[tuple[int, int]] = list(merges or [])
        self.merge_ranks: dict[tuple[int, int], int] = {
            pair: idx for idx, pair in enumerate(self.merges)
        }
        self.superbpe_transition_idx = superbpe_transition_idx

        # Build byte lookup table for fast decoding
        self._id_to_bytes: dict[int, bytes] = {
            self.byte_offset + b: bytes([b]) for b in range(256)
        }
        for rank, (a, b) in enumerate(self.merges):
            new_id = self.byte_offset + 256 + rank
            self._id_to_bytes[new_id] = self._id_to_bytes[a] + self._id_to_bytes[b]

        # Regex matching any registered special token
        if self.special_tokens:
            escaped = sorted((re.escape(t) for t in self.special_tokens), key=len, reverse=True)
            self._special_re: re.Pattern[str] | None = re.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_re = None

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256 + len(self.merges)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    def __len__(self) -> int:
        return self.vocab_size

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int = 1024,
        superbpe_transition_vocab: int | None = None,
        min_pair_freq: int = 2,
        special_tokens: list[str] | None = None,
    ) -> "ByteBPETokenizer":
        """Train a Byte-Level BPE or 2-stage SuperBPE tokenizer on `texts`.

        Args:
            texts: Training documents.
            vocab_size: Target total vocabulary size (must be >= 256 + len(special_tokens)).
            superbpe_transition_vocab: If set (between base vocab and vocab_size), executes
                Stage 1 with regex pretokenization up to this vocab size, then Stage 2
                without whitespace pretokenization to learn superwords (SuperBPE).
            min_pair_freq: Stop merging early if best pair frequency drops below this threshold.
        """
        tok = cls(merges=[], special_tokens=special_tokens)
        base_vocab = tok.byte_offset + 256
        if vocab_size < base_vocab:
            raise ValueError(f"vocab_size ({vocab_size}) must be >= base_vocab ({base_vocab})")
        total_merges = vocab_size - base_vocab

        if superbpe_transition_vocab is not None:
            stage1_merges = max(0, min(total_merges, superbpe_transition_vocab - base_vocab))
        else:
            stage1_merges = total_merges

        # Stage 1: Standard BPE with regex pretokenization (counts unique chunks for speed)
        chunk_counts: Counter[tuple[int, ...]] = Counter()
        for doc in texts:
            for chunk in DEFAULT_PRETOK_PATTERN.findall(doc):
                b_ids = tuple(tok.byte_offset + b for b in chunk.encode("utf-8"))
                if b_ids:
                    chunk_counts[b_ids] += 1

        merges: list[tuple[int, int]] = []
        next_id = base_vocab

        def _run_merges_on_weighted_seqs(
            seq_counts: dict[tuple[int, ...], int], max_merges: int
        ) -> None:
            nonlocal next_id
            for _ in range(max_merges):
                pair_freqs: Counter[tuple[int, int]] = Counter()
                for seq, freq in seq_counts.items():
                    if len(seq) < 2:
                        continue
                    for a, b in zip(seq, seq[1:]):
                        pair_freqs[(a, b)] += freq
                if not pair_freqs:
                    break
                best_pair, best_freq = pair_freqs.most_common(1)[0]
                if best_freq < min_pair_freq:
                    break
                merges.append(best_pair)
                new_counts: dict[tuple[int, ...], int] = {}
                for seq, freq in seq_counts.items():
                    merged = tuple(_merge_sequence(list(seq), best_pair, next_id))
                    new_counts[merged] = new_counts.get(merged, 0) + freq
                seq_counts.clear()
                seq_counts.update(new_counts)
                next_id += 1

        _run_merges_on_weighted_seqs(dict(chunk_counts), stage1_merges)

        transition_idx: int | None = None
        remaining = total_merges - len(merges)

        # Stage 2 (SuperBPE): Lift whitespace pretokenization to merge across words!
        if superbpe_transition_vocab is not None and remaining > 0:
            transition_idx = len(merges)
            # Re-encode full lines/segments with Stage 1 merges applied, then merge across whitespace
            stage1_tok = cls(merges=merges, special_tokens=tok.special_tokens)
            line_counts: dict[tuple[int, ...], int] = {}
            for doc in texts:
                for line in doc.splitlines(keepends=True):
                    if not line:
                        continue
                    ids = tuple(stage1_tok.encode(line, allow_special=False))
                    line_counts[ids] = line_counts.get(ids, 0) + 1
            _run_merges_on_weighted_seqs(line_counts, remaining)

        return cls(
            merges=merges,
            special_tokens=tok.special_tokens,
            superbpe_transition_idx=transition_idx,
        )

    def _apply_merges(self, ids: list[int], max_rank: int | None = None) -> list[int]:
        """Greedily apply learned merges in rank order."""
        if len(ids) < 2 or not self.merge_ranks:
            return ids
        while len(ids) >= 2:
            best_rank = None
            best_pair = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (max_rank is None or rank < max_rank):
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            if best_pair is None or best_rank is None:
                break
            ids = _merge_sequence(ids, best_pair, self.byte_offset + 256 + best_rank)
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        if not text:
            return []
        if self.superbpe_transition_idx is None:
            out: list[int] = []
            for chunk in DEFAULT_PRETOK_PATTERN.findall(text):
                ids = [self.byte_offset + b for b in chunk.encode("utf-8")]
                out.extend(self._apply_merges(ids))
            return out

        # SuperBPE 2-stage encode: Stage 1 inside regex chunks, Stage 2 across chunks
        stage1_ids: list[int] = []
        for chunk in DEFAULT_PRETOK_PATTERN.findall(text):
            ids = [self.byte_offset + b for b in chunk.encode("utf-8")]
            stage1_ids.extend(self._apply_merges(ids, max_rank=self.superbpe_transition_idx))
        return self._apply_merges(stage1_ids, max_rank=None)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = True,
    ) -> list[int]:
        """Encode string to token IDs with optional BOS/EOS and special-token preservation."""
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        if allow_special and self._special_re is not None:
            parts = self._special_re.split(text)
            for part in parts:
                if not part:
                    continue
                if part in self.special_to_id:
                    ids.append(self.special_to_id[part])
                else:
                    ids.extend(self._encode_plain(part))
        else:
            ids.extend(self._encode_plain(text))

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Losslessly decode token IDs back to UTF-8 string."""
        chunks: list[str] = []
        byte_buf = bytearray()

        def flush_bytes() -> None:
            if byte_buf:
                chunks.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for tid in ids:
            if tid in self.id_to_special:
                flush_bytes()
                if not skip_special_tokens:
                    chunks.append(self.id_to_special[tid])
            elif tid in self._id_to_bytes:
                byte_buf.extend(self._id_to_bytes[tid])
        flush_bytes()
        return "".join(chunks)

    def encode_chat_for_sft(
        self,
        messages: list[dict[str, Any]],
        max_seq_len: int | None = None,
    ) -> tuple[list[int], list[int]]:
        """Encode multi-turn chat and return (input_ids, labels) with assistant-only loss mask!

        Prompt/system/user tokens receive label=-100. Assistant response tokens AND the closing
        `<|im_end|>\\n` receive real token IDs so the model learns exact turn termination.
        """
        full_ids: list[int] = [self.bos_id]
        labels: list[int] = [-100]

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            thinking = msg.get("thinking")
            prefix_str = f"<|im_start|>{role}\n"
            if role == "assistant" and thinking:
                body_str = f"<|think|>\n{thinking}\n</|think|>\n{content}<|im_end|>\n"
            else:
                body_str = f"{content}<|im_end|>\n"

            prefix_ids = self.encode(prefix_str, allow_special=True)
            body_ids = self.encode(body_str, allow_special=True)

            full_ids.extend(prefix_ids)
            labels.extend([-100] * len(prefix_ids))

            full_ids.extend(body_ids)
            if role == "assistant":
                labels.extend(body_ids)
            else:
                labels.extend([-100] * len(body_ids))

        if max_seq_len is not None and len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            labels = labels[:max_seq_len]

        # Shift for next-token prediction: input = full[:-1], target = labels[1:]
        return full_ids[:-1], labels[1:]

    def format_chat(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        return format_chat_messages(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "special_tokens": self.special_tokens,
            "superbpe_transition_idx": self.superbpe_transition_idx,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(
            merges=merges,
            special_tokens=payload["special_tokens"],
            superbpe_transition_idx=payload.get("superbpe_transition_idx"),
        )
