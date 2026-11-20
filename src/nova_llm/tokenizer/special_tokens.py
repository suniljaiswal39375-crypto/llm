"""Versioned special-token registry and Qwen3-style chat / thinking / tool-call formatter."""

from __future__ import annotations

from typing import Any

SPECIAL_TOKENS: list[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|think|>",
    "</|think|>",
    "<|tool_call|>",
    "</|tool_call|>",
    "<|tool_response|>",
    "</|tool_response|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]


def format_chat_messages(
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Format multi-turn messages into deterministic ChatML + optional <think> prefix."""
    chunks: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        thinking = msg.get("thinking")
        if role == "assistant" and thinking:
            body = f"<|think|>\n{thinking}\n</|think|>\n{content}"
        else:
            body = content
        chunks.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")

    if add_generation_prompt:
        if enable_thinking:
            chunks.append("<|im_start|>assistant\n<|think|>\n")
        else:
            chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)
