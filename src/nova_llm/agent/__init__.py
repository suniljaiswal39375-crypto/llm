"""Agentic web search engine, RLVR verifiers, and multi-turn search-and-cite runtime."""

from nova_llm.agent.backends import ChatGeneratorBackend, NovaLMBackend, TransformersBackend
from nova_llm.agent.search_agent import (
    AgentResponse,
    SEARCH_SYSTEM_PROMPT,
    SearchAgent,
    fetch_web_page_text,
    split_thinking_and_answer,
)
from nova_llm.agent.search_engine import (
    BM25LocalWebIndex,
    HybridSearchBackend,
    LiveHTTPWebBackend,
    SearchResult,
    WebDocument,
    build_default_web_corpus,
    format_search_results,
)
from nova_llm.agent.search_verifiers import (
    WebSearchRLVRVerifier,
    parse_tool_calls,
    verify_grounded_synthesis,
    verify_search_abstention,
    verify_search_tool_call,
)

__all__ = [
    "AgentResponse",
    "BM25LocalWebIndex",
    "ChatGeneratorBackend",
    "HybridSearchBackend",
    "LiveHTTPWebBackend",
    "NovaLMBackend",
    "SEARCH_SYSTEM_PROMPT",
    "SearchAgent",
    "SearchResult",
    "TransformersBackend",
    "WebDocument",
    "WebSearchRLVRVerifier",
    "build_default_web_corpus",
    "fetch_web_page_text",
    "format_search_results",
    "parse_tool_calls",
    "split_thinking_and_answer",
    "verify_grounded_synthesis",
    "verify_search_abstention",
    "verify_search_tool_call",
]
