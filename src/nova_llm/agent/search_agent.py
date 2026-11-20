"""Multi-turn Agentic Web Search & Reasoning Runtime for Nova-LLM & Llama-3.2-3B/HF Models.

Supports:
  - Pluggable backends: native `NovaLM` OR Hugging Face / Unsloth `TransformersBackend` (Llama 3.2 3B!)
  - Optional chain-of-thought `<|think|>...</|think|>` extraction
  - Autonomous `<|tool_call|>` execution (`web_search` and `fetch_page`)
  - Multi-hop search rounds (up to `max_search_rounds`)
  - Streaming SSE events (`thinking`, `tool_call`, `sources`, `delta`, `done`)
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

from nova_llm.agent.backends import ChatGeneratorBackend, NovaLMBackend
from nova_llm.agent.search_engine import HybridSearchBackend, SearchResult, format_search_results
from nova_llm.agent.search_verifiers import parse_tool_calls
from nova_llm.models.transformer import NovaLM
from nova_llm.tokenizer.bpe import ByteBPETokenizer

SEARCH_SYSTEM_PROMPT = (
    "You are Nova-Search, a web-searching AI. Emit a web_search tool call when factual lookup is needed: "
    '<|tool_call|>{"name": "web_search", "arguments": {"query": "your concise search query"}}</|tool_call|>. '
    "After receiving search results, synthesize a clear, accurate answer strictly grounded in the sources "
    "with inline citations like [1] and [2]. Never fabricate facts or citations."
)

_THINK_RE = re.compile(r"<\|think\|>(.*?)</\|think\|>", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def split_thinking_and_answer(text: str) -> tuple[str | None, str]:
    """Separate `<|think|>...</|think|>` reasoning trace from the visible final answer."""
    matches = _THINK_RE.findall(text)
    if not matches:
        return None, text.strip()
    thinking = "\n".join(m.strip() for m in matches if m.strip())
    visible = _THINK_RE.sub("", text).strip()
    return thinking or None, visible


def fetch_web_page_text(url: str, max_chars: int = 1600, timeout: float = 6.0) -> str:
    """Fetch a public HTTP/HTTPS web page and return cleaned plain text excerpt."""
    if not url.startswith(("http://", "https://")):
        return "Invalid URL: must start with http:// or https://"
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NovaSearchBot/1.0)"},
            timeout=timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        cleaned = _SCRIPT_STYLE_RE.sub(" ", resp.text)
        plain = html.unescape(_HTML_TAG_RE.sub(" ", cleaned))
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
    """Runs NovaLM OR Llama-3.2-3B/HF models in native agentic web-search & reasoning mode."""

    def __init__(
        self,
        model: NovaLM | None = None,
        tokenizer: ByteBPETokenizer | None = None,
        search_backend: HybridSearchBackend | None = None,
        max_search_rounds: int = 2,
        top_k_results: int = 3,
        backend: ChatGeneratorBackend | None = None,
        system_prompt: str = SEARCH_SYSTEM_PROMPT,
    ) -> None:
        if backend is not None:
            self.backend: ChatGeneratorBackend = backend
        elif model is not None and tokenizer is not None:
            self.backend = NovaLMBackend(model, tokenizer)
        else:
            raise ValueError("Provide either `backend` or both `(model, tokenizer)`")

        self.search_backend = search_backend or HybridSearchBackend()
        self.max_search_rounds = max_search_rounds
        self.top_k_results = top_k_results
        self.system_prompt = system_prompt

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
        max_tokens: int = 256,
        temperature: float = 0.0,
        force_search: bool = False,
    ) -> AgentResponse:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        tool_calls_made: list[dict[str, Any]] = []
        collected_sources: list[SearchResult] = []
        raw_turns: list[str] = []
        thoughts: list[str] = []

        for round_idx in range(self.max_search_rounds + 1):
            turn_text = self.backend.generate_chat(
                messages, max_tokens=max_tokens, temperature=temperature, repetition_penalty=1.0
            )
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
        max_tokens: int = 256,
        temperature: float = 0.0,
        force_search: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield structured SSE events: `thinking`, `tool_call`, `sources`, `delta`, `done`."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        turn1 = self.backend.generate_chat(
            messages, max_tokens=min(128, max_tokens), temperature=temperature, repetition_penalty=1.0
        )
        think1, visible1 = split_thinking_and_answer(turn1)
        if think1:
            yield {"type": "thinking", "content": think1}

        calls = parse_tool_calls(turn1)
        if not calls and force_search:
            calls = [{"name": "web_search", "arguments": {"query": user_query}}]
            turn1 = f'<|tool_call|>{{"name": "web_search", "arguments": {{"query": "{user_query}"}}}}</|tool_call|>'

        if not calls:
            yield {"type": "delta", "content": visible1}
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

        for chunk in self.backend.stream_chat(
            messages, max_tokens=max_tokens, temperature=temperature, repetition_penalty=1.0
        ):
            yield {"type": "delta", "content": chunk}
        yield {"type": "done", "used_search": True}
