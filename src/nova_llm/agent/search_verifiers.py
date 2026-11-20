"""Deterministic RLVR Verifiers for Agentic Web Search (Tool-Call Syntax, Query Quality & Grounded Citations)."""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(r"<\|tool_call\|>(.*?)</\|tool_call\|>", re.DOTALL)
_CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract and parse all valid JSON `<|tool_call|>...</|tool_call|>` blocks from model output."""
    parsed: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.findall(text):
        raw = match.strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
            parsed.append(obj)
    return parsed


def verify_search_tool_call(
    response: str,
    expected_keywords: list[str] | None = None,
    expected_tool: str = "web_search",
) -> float:
    """Deterministic reward in [0.0, 1.0] for valid search tool-calling & query precision.

    Scoring breakdown:
      +0.40 : Valid `<|tool_call|>` JSON block with matching tool name and non-empty query string.
      +0.40 : Fraction of expected high-signal query keywords present in `arguments["query"]`.
      +0.20 : Concise, well-formed search query (2 to 14 words, no hallucinated final answer mixed in).
    """
    calls = parse_tool_calls(response)
    if not calls:
        return 0.0
    call = calls[0]
    if call.get("name") != expected_tool:
        return 0.1
    query = call.get("arguments", {}).get("query")
    if not isinstance(query, str) or not query.strip():
        return 0.1

    reward = 0.4
    q_lower = query.lower()
    if expected_keywords:
        hits = sum(1 for kw in expected_keywords if kw.lower() in q_lower)
        reward += 0.4 * (hits / float(len(expected_keywords)))
    else:
        reward += 0.4

    words = query.strip().split()
    if 2 <= len(words) <= 14:
        reward += 0.2
    return min(1.0, reward)


def verify_grounded_synthesis(
    response: str,
    expected_facts: list[str],
    allowed_citation_ids: list[int] | None = None,
) -> float:
    """Deterministic reward in [0.0, 1.0] for evidence-grounded, inline-cited synthesis.

    Penalizes hallucinated citation IDs (e.g. citing `[5]` when only `[1]`..`[2]` exist)
    and rewards exact fact coverage + clean `[1]`/`[2]` citation grounding.
    """
    if not response.strip():
        return 0.0
    # Must not emit another raw tool call when synthesis is expected
    if "<|tool_call|>" in response:
        return 0.0

    cited_ids = [int(x) for x in _CITATION_RE.findall(response)]
    valid_ids = set(allowed_citation_ids if allowed_citation_ids is not None else [1, 2, 3])

    citation_score = 0.0
    if cited_ids:
        invalid_cites = [cid for cid in cited_ids if cid not in valid_ids]
        if invalid_cites:
            # Strict anti-hallucination penalty for fabricated citation indices
            return 0.0
        citation_score = 0.35

    resp_lower = response.lower()
    if expected_facts:
        fact_hits = sum(1 for f in expected_facts if f.lower() in resp_lower)
        fact_score = 0.65 * (fact_hits / float(len(expected_facts)))
    else:
        fact_score = 0.65

    return min(1.0, citation_score + fact_score)


def verify_search_abstention(response: str) -> float:
    """Reward honest abstention when search results do not contain the requested fact."""
    low = response.lower()
    cues = ("could not find", "no relevant", "not found in the search", "do not contain", "cannot verify", "insufficient")
    if any(c in low for c in cues):
        return 1.0
    return 0.0


class WebSearchRLVRVerifier:
    """Unified deterministic RLVR verifier for `RLVRTrainer` (GRPO / Dr. GRPO / DAPO / GSPO)."""

    def score(self, completion: str, metadata: dict[str, Any] | None = None) -> float:
        meta = metadata or {}
        task = meta.get("task", "tool_call")
        if task == "tool_call":
            return verify_search_tool_call(
                completion,
                expected_keywords=meta.get("keywords"),
                expected_tool=meta.get("tool", "web_search"),
            )
        if task == "synthesis":
            return verify_grounded_synthesis(
                completion,
                expected_facts=meta.get("facts", []),
                allowed_citation_ids=meta.get("allowed_citations", [1, 2, 3]),
            )
        if task == "abstention":
            return verify_search_abstention(completion)
        if task == "direct_no_search":
            # Reward answering directly without wasting a search call on greetings/trivial math
            if "<|tool_call|>" in completion:
                return 0.0
            expected = str(meta.get("expected", "")).lower()
            return 1.0 if expected and expected in completion.lower() else 0.3
        return 0.0
