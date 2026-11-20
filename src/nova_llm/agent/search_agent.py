"""Multi-turn Agentic Web Search & Reasoning Runtime for Nova-LLM.

Supports:
  - Optional chain-of-thought `<|think|>...</|think|>` extraction
  - Autonomous `<|tool_call|>` execution (`web_search` and `fetch_page`)
  - Multi-hop search rounds (up to `max_search_rounds`)
  - Streaming SSE events for live browser UIs
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx
import torch

from nova_llm.agent.search_engine import HybridSearchBackend, SearchResult, format_search_results
from nova_llm.agent.search_verifiers import parse_tool_calls
from nova_llm.inference.generator import generate, stream_generate
from nova_llm.inference.sampler import SamplingParams
from nova_llm.models.transformer import NovaLM
from nova_llm.tokenizer.bpe import ByteBPETokenizer

SEARCH_SYSTEM_PROMPT = (
    "You are Nova-Search, a web-searching AI. Emit a web_search tool call when factual lookup is needed, "
    "then synthesize a concise answer strictly grounded in the results with inline citations like [1]."
)

_THINK_RE = re.compile(r"<\|think\|>(.*?)</\|think\|>", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def split_thinking_and_answer(text: str) -> tuple[str | None, str]:
    """Separate `<|think|>...</|think|>` reasoning trace from the visible final answer."""
    matches = _THINK_RE.findall(text)
    if not matches:
        return None, text.strip()
    thinking = "\n".join(m.strip() for m in matches if m.strip())
    visible = _THINK_RE.sub("", text).strip()
    return thinking or None, visible


def fetch_web_page_text(url: str, max_chars: int = 1200, timeout: float = 6.0) -> str:
    """Fetch a public HTTP/HTTPS web page and return cleaned plain text excerpt."""
    if not url.startswith(("http://", "https://")):
        return "Invalid URL: must start with http:// or https://"
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        plain = _HTML_TAG_RE.sub(" ", resp.text)
        plain = re.sub(r"\s+", " ", plain).strip()
        return plain[:max_chars]
    except Exception as exc:
        return f"Could not fetch URL ({type(exc).__name__}): {exc}"


@dataclass
class AgentResponse:
    answer: str
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    sources: list[SearchResult] = field(default_factory=list)
    raw_turns: list[str] = field(default_factory=list)
    used_search: bool = False


class SearchAgent:
    """Runs NovaLM in native agentic web-search, page-fetch, and `<|think|>` reasoning mode."""

    def __init__(
        self,
        model: NovaLM,
        tokenizer: ByteBPETokenizer,
        search_backend: HybridSearchBackend | None = None,
        max_search_rounds: int = 2,
        top_k_results: int = 3,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.search_backend = search_backend or HybridSearchBackend()
        self.max_search_rounds = max_search_rounds
        self.top_k_results = top_k_results

    def _stop_ids(self) -> list[int]:
        return [self.tokenizer.eos_id, self.tokenizer.special_to_id["<|im_end|>"]]

    def _clean_text(self, text: str) -> str:
        return text.replace("<|im_end|>", "").replace("<|eos|>", "").strip()

    def _generate_turn(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        prompt_text = self.tokenizer.format_chat(messages, add_generation_prompt=True)
        ids = torch.tensor(
            [self.tokenizer.encode(prompt_text, add_bos=True, allow_special=True)], dtype=torch.long
        )
        max_ctx = self.model.cfg.max_seq_len - 16
        if ids.shape[1] > max_ctx:
            ids = ids[:, -max_ctx:]
        params = SamplingParams(
            max_new_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=1.0,
            stop_token_ids=self._stop_ids(),
            stop_strings=["<|im_end|>", "<|eos|>"],
        )
        res = generate(self.model, ids, params=params, tokenizer=self.tokenizer, use_cache=True)
        return self._clean_text(res.text or "")

    def _execute_tool(
        self,
        call: dict[str, Any],
        fallback_query: str,
    ) -> tuple[str, list[SearchResult]]:
        name = str(call.get("name", "web_search"))
        args = call.get("arguments", {})
        if name == "fetch_page":
            url = str(args.get("url", ""))
            return f"Page content ({url}): {fetch_web_page_text(url)}", []
        query = str(args.get("query", fallback_query))
        results = self.search_backend.search(query, top_k=self.top_k_results)
        return format_search_results(results), results

    def run(
        self,
        user_query: str,
        history: list[dict[str, str]] | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        force_search: bool = False,
    ) -> AgentResponse:
        messages: list[dict[str, str]] = [{"role": "system", "content": SEARCH_SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        tool_calls_made: list[dict[str, Any]] = []
        collected_sources: list[SearchResult] = []
        raw_turns: list[str] = []
        thoughts: list[str] = []

        for round_idx in range(self.max_search_rounds + 1):
            turn_text = self._generate_turn(messages, max_tokens=max_tokens, temperature=temperature)
            raw_turns.append(turn_text)
            think_part, visible_part = split_thinking_and_answer(turn_text)
            if think_part:
                thoughts.append(think_part)

            calls = parse_tool_calls(turn_text)
            if not calls and force_search and round_idx == 0:
                calls = [{"name": "web_search", "arguments": {"query": user_query}}]
                turn_text = (
                    f'<|tool_call|>{{"name": "web_search", "arguments": {{"query": "{user_query}"}}}}</|tool_call|>'
                )

            if calls and round_idx < self.max_search_rounds:
                call = calls[0]
                tool_calls_made.append(call)
                tool_text, new_sources = self._execute_tool(call, fallback_query=user_query)
                if new_sources:
                    collected_sources = new_sources
                messages.append({"role": "assistant", "content": turn_text})
                messages.append({"role": "tool", "content": f"<|tool_result|>{tool_text}</|tool_result|>"})
                continue

            return AgentResponse(
                answer=visible_part,
                thinking="\n".join(thoughts) if thoughts else None,
                tool_calls=tool_calls_made,
                sources=collected_sources,
                raw_turns=raw_turns,
                used_search=len(tool_calls_made) > 0,
            )

        _, final_vis = split_thinking_and_answer(raw_turns[-1] if raw_turns else "")
        return AgentResponse(
            answer=final_vis,
            thinking="\n".join(thoughts) if thoughts else None,
            tool_calls=tool_calls_made,
            sources=collected_sources,
            raw_turns=raw_turns,
            used_search=len(tool_calls_made) > 0,
        )

    def stream_run(
        self,
        user_query: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        force_search: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield structured SSE events: `tool_call`, `sources`, `delta`, `done`."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        turn1 = self._generate_turn(messages, max_tokens=96, temperature=temperature)
        calls = parse_tool_calls(turn1)

        if not calls and force_search:
            calls = [{"name": "web_search", "arguments": {"query": user_query}}]
            turn1 = f'<|tool_call|>{{"name": "web_search", "arguments": {{"query": "{user_query}"}}}}</|tool_call|>'

        if not calls:
            _, visible = split_thinking_and_answer(turn1)
            yield {"type": "delta", "content": visible}
            yield {"type": "done", "used_search": False}
            return

        call = calls[0]
        tool_name = str(call.get("name", "web_search"))
        q = str(call.get("arguments", {}).get("query", call.get("arguments", {}).get("url", user_query)))
        yield {"type": "tool_call", "name": tool_name, "query": q}

        tool_text, results = self._execute_tool(call, fallback_query=user_query)
        if results:
            yield {"type": "sources", "sources": [r.to_dict() for r in results]}

        messages.append({"role": "assistant", "content": turn1})
        messages.append({"role": "tool", "content": f"<|tool_result|>{tool_text}</|tool_result|>"})

        prompt_text = self.tokenizer.format_chat(messages, add_generation_prompt=True)
        ids = torch.tensor(
            [self.tokenizer.encode(prompt_text, add_bos=True, allow_special=True)], dtype=torch.long
        )
        max_ctx = self.model.cfg.max_seq_len - 16
        if ids.shape[1] > max_ctx:
            ids = ids[:, -max_ctx:]

        params = SamplingParams(
            max_new_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=1.0,
            stop_token_ids=self._stop_ids(),
            stop_strings=["<|im_end|>", "<|eos|>"],
        )
        for chunk in stream_generate(self.model, ids, self.tokenizer, params=params):
            clean = chunk.replace("<|im_end|>", "").replace("<|eos|>", "")
            if clean:
                yield {"type": "delta", "content": clean}
        yield {"type": "done", "used_search": True}
