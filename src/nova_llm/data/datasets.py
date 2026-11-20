"""Streaming real-world dataset loaders for Pre-Training, SFT, DPO, and RLVR (JSONL & plain text).

Supports standard open dataset formats out of the box:
  - Pre-training: `.txt` and `.jsonl` (`text` / `content` / `document` keys) — O(1) streaming.
  - SFT: OpenAI `messages`, ShareGPT `conversations`, and Alpaca `instruction`/`input`/`output`.
  - DPO: UltraFeedback / Anthropic `prompt`, `chosen`, `rejected` (string or message list).
  - RLVR: GSM8K / MATH / MBPP / Web-Search verifiable JSONL prompts + metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from nova_llm.post_training.dpo import DPOExample
from nova_llm.post_training.rlvr import RLVRPrompt


def stream_text_corpus(paths: list[str | Path], text_field: str = "text") -> Iterator[str]:
    """Lazily yield documents from `.txt` or `.jsonl` files with O(1) memory overhead."""
    for raw_path in paths:
        p = Path(raw_path)
        if p.is_dir():
            sub_files = sorted(list(p.glob("*.jsonl")) + list(p.glob("*.txt")))
            yield from stream_text_corpus(sub_files, text_field=text_field)
            continue

        if p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get(text_field) or obj.get("content") or obj.get("document") or ""
                    if isinstance(text, str) and text.strip():
                        yield text
        else:
            # Plain text: split by blank lines or yield file content
            content = p.read_text(encoding="utf-8")
            chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
            yield from chunks


def _extract_text_from_turn(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, list) and val:
        # Handle list of {"role": ..., "content": ...} -> extract final assistant content
        for item in reversed(val):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return str(item.get("content", ""))
        last = val[-1]
        return str(last.get("content", last)) if isinstance(last, dict) else str(last)
    if isinstance(val, dict):
        return str(val.get("content", ""))
    return str(val)


def load_sft_jsonl(path: str | Path, limit: int | None = None) -> list[list[dict[str, Any]]]:
    """Load multi-turn SFT conversations from JSONL (OpenAI, ShareGPT, or Alpaca schema)."""
    convs: list[list[dict[str, Any]]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj and isinstance(obj["messages"], list):
                convs.append(obj["messages"])
            elif "conversations" in obj and isinstance(obj["conversations"], list):
                role_map = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}
                mapped = [
                    {
                        "role": role_map.get(str(m.get("from", m.get("role", "user"))).lower(), "user"),
                        "content": str(m.get("value", m.get("content", ""))),
                    }
                    for m in obj["conversations"]
                ]
                convs.append(mapped)
            elif "instruction" in obj and "output" in obj:
                user_msg = str(obj["instruction"])
                if obj.get("input"):
                    user_msg = f"{user_msg}\n\n{obj['input']}"
                turn: dict[str, Any] = {"role": "assistant", "content": str(obj["output"])}
                if obj.get("thinking"):
                    turn["thinking"] = str(obj["thinking"])
                convs.append([{"role": "user", "content": user_msg}, turn])
            if limit is not None and len(convs) >= limit:
                break
    return convs


def load_dpo_jsonl(path: str | Path, limit: int | None = None) -> list[DPOExample]:
    """Load preference pairs from JSONL (`prompt`, `chosen`, `rejected`)."""
    examples: list[DPOExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt", ""))
            if not prompt and isinstance(obj.get("chosen"), list):
                for m in obj["chosen"]:
                    if isinstance(m, dict) and m.get("role") == "user":
                        prompt = str(m.get("content", ""))
                        break
            chosen = _extract_text_from_turn(obj["chosen"])
            rejected = _extract_text_from_turn(obj["rejected"])
            if prompt and chosen and rejected:
                examples.append(DPOExample(prompt=prompt, chosen=chosen, rejected=rejected))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def load_rlvr_jsonl(path: str | Path, limit: int | None = None) -> list[RLVRPrompt]:
    """Load verifiable RLVR prompts from JSONL (math, Python code, constraints, or web search)."""
    prompts: list[RLVRPrompt] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt") or obj.get("question") or "")
            if not prompt:
                continue
            meta = dict(obj.get("metadata", {}))
            for key in ("type", "task", "expected", "answer", "test_code", "keywords", "facts", "constraints"):
                if key in obj and key not in meta:
                    meta[key] = obj[key]
            if "answer" in meta and "expected" not in meta:
                meta["expected"] = meta["answer"]
            prompts.append(RLVRPrompt(prompt=prompt, metadata=meta))
            if limit is not None and len(prompts) >= limit:
                break
    return prompts
