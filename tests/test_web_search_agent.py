"""Verification tests for Nova-Search Agentic Web Search, BM25 Index, RLVR Verifiers & API."""

from __future__ import annotations

import torch
from fastapi.testclient import TestClient

from nova_llm.agent import (
    BM25LocalWebIndex,
    HybridSearchBackend,
    SearchAgent,
    WebSearchRLVRVerifier,
    format_search_results,
    parse_tool_calls,
    verify_grounded_synthesis,
    verify_search_abstention,
    verify_search_tool_call,
)
from nova_llm.config import ModelConfig
from nova_llm.inference import create_app
from nova_llm.models import NovaLM
from nova_llm.tokenizer import ByteBPETokenizer


def test_bm25_web_index_ranks_relevant_documents_first() -> None:
    index = BM25LocalWebIndex()
    r1 = index.search("Kimi K2 MuonClip QK-Clip optimizer", top_k=3)
    assert len(r1) >= 1 and "MuonClip" in r1[0].title

    r2 = index.search("Chandrayaan-3 ISRO lunar south pole", top_k=2)
    assert len(r2) >= 1 and "Chandrayaan-3" in r2[0].title

    formatted = format_search_results(r2)
    assert formatted.startswith("[1] ")


def test_tool_call_parser_and_rlvr_search_verifiers() -> None:
    good_call = '<|tool_call|>{"name": "web_search", "arguments": {"query": "DeepSeek-V3 MLA architecture"}}</|tool_call|>'
    parsed = parse_tool_calls(good_call)
    assert len(parsed) == 1 and parsed[0]["name"] == "web_search"
    assert verify_search_tool_call(good_call, expected_keywords=["deepseek", "mla"]) == 1.0

    # Malformed JSON or missing query must score low/0
    assert verify_search_tool_call("<|tool_call|>{bad json}</|tool_call|>") == 0.0
    assert verify_search_tool_call("I just guessed without searching.") == 0.0

    # Grounded synthesis with valid citation vs fabricated citation [9]
    good_ans = "DeepSeek-V3 introduces MLA and MTP with 671B total parameters [1]."
    assert verify_grounded_synthesis(good_ans, expected_facts=["mla", "mtp", "671b"], allowed_citation_ids=[1, 2]) == 1.0
    hallucinated_cite = "DeepSeek-V3 introduces MLA and MTP with 671B total parameters [9]."
    assert verify_grounded_synthesis(hallucinated_cite, expected_facts=["mla", "mtp"], allowed_citation_ids=[1, 2]) == 0.0

    # Abstention verifier
    assert verify_search_abstention("I could not find any relevant evidence in the search results.") == 1.0
    assert verify_search_abstention("The secret password is 1234.") == 0.0

    # Unified WebSearchRLVRVerifier dispatch
    v = WebSearchRLVRVerifier()
    assert v.score(good_call, {"task": "tool_call", "keywords": ["deepseek"]}) == 1.0
    assert v.score("4", {"task": "direct_no_search", "expected": "4"}) == 1.0
    assert v.score(good_call, {"task": "direct_no_search", "expected": "4"}) == 0.0


def test_search_agent_and_fastapi_search_endpoint() -> None:
    torch.manual_seed(99)
    tok = ByteBPETokenizer.train(
        ["Hello web search DeepSeek-V3 MLA MuonClip results [1]"], vocab_size=320
    )
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, d_model=48, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=256, pad_vocab_multiple=1
    )
    model = NovaLM(cfg)
    agent = SearchAgent(model, tok, search_backend=HybridSearchBackend())

    # Force-search mode guarantees tool execution -> retrieval -> Turn 2 synthesis
    resp = agent.run("What is DeepSeek-V3?", max_tokens=16, temperature=0.0, force_search=True)
    assert resp.used_search is True
    assert len(resp.sources) >= 1 and "DeepSeek-V3" in resp.sources[0].title

    # Verify FastAPI /v1/search/chat sync + SSE streaming
    app = create_app(model, tok, model_name="nova-search")
    client = TestClient(app)

    r_sync = client.post("/v1/search/chat", json={"query": "What is MuonClip?", "max_tokens": 12, "force_search": True})
    assert r_sync.status_code == 200
    body = r_sync.json()
    assert body["used_search"] is True and len(body["sources"]) >= 1

    r_stream = client.post(
        "/v1/search/chat", json={"query": "What is MuonClip?", "max_tokens": 12, "force_search": True, "stream": True}
    )
    assert r_stream.status_code == 200
    assert '"type": "tool_call"' in r_stream.text
    assert '"type": "sources"' in r_stream.text
    assert "data: [DONE]" in r_stream.text
