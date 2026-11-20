"""Tests for Llama-3.2-3B / TransformersBackend integration, free DuckDuckGo parser & Colab notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import torch
from fastapi.testclient import TestClient

from nova_llm.agent import (
    HybridSearchBackend,
    LiveHTTPWebBackend,
    SearchAgent,
    TransformersBackend,
)
from nova_llm.inference import create_app


class _FakeHFTokenizer:
    """Minimal mock of a Llama-3.2 Hugging Face tokenizer for offline unit testing."""

    vocab_size = 128256
    eos_token_id = 128001
    pad_token_id = 128001
    chat_template = "llama-3.1"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def convert_tokens_to_ids(self, tok: str) -> int:
        return 128009 if tok == "<|eot_id|>" else -1

    def __call__(self, texts: list[str], return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        out = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return out


class _FakeHFModel:
    device = torch.device("cpu")

    def parameters(self):
        yield torch.zeros(10)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Append 2 dummy generated token IDs after prompt
        extra = torch.tensor([[42, 43]], dtype=torch.long)
        return torch.cat([input_ids, extra], dim=1)


def test_transformers_backend_multi_turn_search_and_fastapi() -> None:
    """Verify Llama-3.2-3B style HF backend runs full <think> + tool_call -> Search -> Cited Answer."""
    fake_turns = [
        '<|think|>Need factual lookup on MuonClip.</|think|>\n<|tool_call|>{"name": "web_search", "arguments": {"query": "Kimi K2 MuonClip QK-Clip"}}</|tool_call|>',
        "MuonClip combines Muon orthogonalized momentum with per-head QK-Clip [1].",
    ]
    tok = _FakeHFTokenizer(fake_turns)
    backend = TransformersBackend(model=_FakeHFModel(), tokenizer=tok, model_name="Llama-3.2-3B-Test")
    agent = SearchAgent(backend=backend, search_backend=HybridSearchBackend())

    res = agent.run("What is MuonClip?", max_tokens=64, temperature=0.0)
    assert res.used_search is True
    assert res.thinking == "Need factual lookup on MuonClip."
    assert res.tool_calls[0]["arguments"]["query"] == "Kimi K2 MuonClip QK-Clip"
    assert len(res.sources) >= 1 and "MuonClip" in res.sources[0].title
    assert "[1]" in res.answer

    # Test FastAPI server powered directly by TransformersBackend (no NovaLM required!)
    tok2 = _FakeHFTokenizer(fake_turns)
    app = create_app(
        backend=TransformersBackend(model=_FakeHFModel(), tokenizer=tok2, model_name="Llama-3.2-3B-Test")
    )
    client = TestClient(app)
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["backend"] == "transformers"

    r = client.post("/v1/search/chat", json={"query": "What is MuonClip?", "max_tokens": 64})
    assert r.status_code == 200
    body = r.json()
    assert body["used_search"] is True and body["thinking"] == "Need factual lookup on MuonClip."
    assert "[1]" in body["answer"]


def test_duckduckgo_html_parser_extracts_clean_urls_and_snippets() -> None:
    sample_html = """
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpytorch.org%2Fdocs%2Ffsdp2">PyTorch <b>FSDP2</b> Docs</a>
      <a class="result__snippet">Per-parameter DTensor sharding via fully_shard().</a>
    </div>
    <div class="result">
      <a class="result__a" href="https://arxiv.org/abs/2507.20534">Kimi K2 &amp; MuonClip</a>
      <div class="result__snippet">Zero-spike 15.5T training with QK-Clip.</div>
    </div>
    """
    results = LiveHTTPWebBackend.parse_duckduckgo_html(sample_html, top_k=2)
    assert len(results) == 2
    assert results[0].url == "https://pytorch.org/docs/fsdp2"
    assert results[0].title == "PyTorch FSDP2 Docs"
    assert "DTensor" in results[0].snippet
    assert results[1].title == "Kimi K2 & MuonClip"


def test_colab_notebook_json_and_python_cells_valid() -> None:
    nb_path = Path("notebooks/Nova_Llama3.2_3B_WebSearch.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 8
    for cell in code_cells:
        src = "".join(cell["source"])
        # Replace IPython `!` shell magics with indented `pass` for AST syntax check
        py_lines = [
            (line[: len(line) - len(line.lstrip())] + "pass") if line.lstrip().startswith("!") else line
            for line in src.splitlines()
        ]
        ast.parse("\n".join(py_lines))
