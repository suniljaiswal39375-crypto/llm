"""Multi-turn Agentic Web Search Runtime for Nova-LLM.

Executes the genuine agent loop:
  User Question -> NovaLM Turn 1 -> (Optional <|tool_call|> web_search)
                -> SearchBackend executes query -> <|tool_result|> snippets
                -> NovaLM Turn 2 -> Evidence-grounded cited answer [1][2]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

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


@dataclass
class AgentResponse:
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    sources: list[SearchResult] = field(default_factory=list)
    raw_turns: list[str] = field(default_factory=list)
    used_search: bool = False


class SearchAgent:
    """Runs NovaLM in native agentic web-search-and-cite mode."""

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
        # Keep prompt within model's context budget
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

        for round_idx in range(self.max_search_rounds + 1):
            turn_text = self._generate_turn(messages, max_tokens=max_tokens, temperature=temperature)
            raw_turns.append(turn_text)
            calls = parse_tool_calls(turn_text)

            # Optional forced search fallback if user explicitly clicked "Force Web Search"
            if not calls and force_search and round_idx == 0:
                calls = [{"name": "web_search", "arguments": {"query": user_query}}]
                turn_text = (
                    f'<|tool_call|>{{"name": "web_search", "arguments": {{"query": "{user_query}"}}}}</|tool_call|>'
                )

            if calls and round_idx < self.max_search_rounds:
                call = calls[0]
                tool_calls_made.append(call)
                q = str(call.get("arguments", {}).get("query", user_query))
                results = self.search_backend.search(q, top_k=self.top_k_results)
                collected_sources = results
                formatted = format_search_results(results)
                messages.append({"role": "assistant", "content": turn_text})
                messages.append({"role": "tool", "content": f"<|tool_result|>{formatted}</|tool_result|>"})
                continue

            # Final answer turn!
            clean_ans = turn_text
            # If the model output a clean answer without tool calls, return it
            return AgentResponse(
                answer=clean_ans,
                tool_calls=tool_calls_made,
                sources=collected_sources,
                raw_turns=raw_turns,
                used_search=len(tool_calls_made) > 0,
            )

        return AgentResponse(
            answer=raw_turns[-1] if raw_turns else "",
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
        # Turn 1: let model decide whether to call web_search or answer directly
        turn1 = self._generate_turn(messages, max_tokens=96, temperature=temperature)
        calls = parse_tool_calls(turn1)

        if not calls and force_search:
            calls = [{"name": "web_search", "arguments": {"query": user_query}}]
            turn1 = f'<|tool_call|>{{"name": "web_search", "arguments": {{"query": "{user_query}"}}}}</|tool_call|>'

        if not calls:
            yield {"type": "delta", "content": turn1}
            yield {"type": "done", "used_search": False}
            return

        call = calls[0]
        q = str(call.get("arguments", {}).get("query", user_query))
        yield {"type": "tool_call", "name": call.get("name", "web_search"), "query": q}

        results = self.search_backend.search(q, top_k=self.top_k_results)
        yield {"type": "sources", "sources": [r.to_dict() for r in results]}

        formatted = format_search_results(results)
        messages.append({"role": "assistant", "content": turn1})
        messages.append({"role": "tool", "content": f"<|tool_result|>{formatted}</|tool_result|>"})

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
