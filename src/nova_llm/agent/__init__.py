"""Agentic web search engine, RLVR verifiers, and multi-turn search-and-cite runtime."""

from nova_llm.agent.search_agent import AgentResponse, SearchAgent, SEARCH_SYSTEM_PROMPT
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
    "HybridSearchBackend",
    "LiveHTTPWebBackend",
    "SEARCH_SYSTEM_PROMPT",
    "SearchAgent",
    "SearchResult",
    "WebDocument",
    "WebSearchRLVRVerifier",
    "build_default_web_corpus",
    "format_search_results",
    "parse_tool_calls",
    "verify_grounded_synthesis",
    "verify_search_abstention",
    "verify_search_tool_call",
]
