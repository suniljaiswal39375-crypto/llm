"""Verification tests for Sampling, Speculative Decoding, Quantization, SafeTensors, Eval & FastAPI."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from nova_llm.config import ModelConfig
from nova_llm.eval import evaluate_bpb_and_ppl, evaluate_multiple_choice, evaluate_niah, generate_eval_report
from nova_llm.inference import (
    SamplingParams,
    apply_qat,
    create_app,
    export_safetensors,
    generate,
    load_safetensors_model,
    quantize_model_weight_only,
    sample_next_token,
    speculative_generate,
)
from nova_llm.models import NovaLM
from nova_llm.tokenizer import ByteBPETokenizer


def _model_and_tok(mtp_depth: int = 1) -> tuple[NovaLM, ByteBPETokenizer]:
    tok = ByteBPETokenizer.train(
        ["Hello world! Transformers predict tokens efficiently. Passkey is 74921."], vocab_size=300
    )
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        mtp_depth=mtp_depth,
        pad_vocab_multiple=1,
    )
    return NovaLM(cfg), tok


def test_sampler_edge_cases() -> None:
    """Verify greedy temp=0, tiny top_p (never empty!), min_p, and repetition penalty."""
    logits = torch.tensor([[1.0, 5.0, -2.0, 0.5]])
    tok_g, _ = sample_next_token(logits, SamplingParams(temperature=0.0))
    assert tok_g.item() == 1

    # Extreme top_p=0.001 must still keep the top-1 token without NaN
    tok_p, probs_p = sample_next_token(logits, SamplingParams(temperature=1.0, top_p=0.001))
    assert tok_p.item() == 1 and torch.isfinite(probs_p).all()

    # min_p=0.99 must still keep top-1 token
    tok_m, probs_m = sample_next_token(logits, SamplingParams(temperature=1.0, min_p=0.99))
    assert tok_m.item() == 1 and torch.isfinite(probs_m).all()

    # Repetition penalty reduces logit of seen token
    hist = torch.tensor([[1]])
    _, p_no_pen = sample_next_token(logits, SamplingParams(temperature=1.0, repetition_penalty=1.0), history_ids=hist)
    _, p_pen = sample_next_token(logits, SamplingParams(temperature=1.0, repetition_penalty=2.0), history_ids=hist)
    assert p_pen[0, 1].item() < p_no_pen[0, 1].item()


@pytest.mark.parametrize("method", ["ngram", "draft_model", "mtp"])
def test_speculative_decoding_exact_greedy_parity(method: str) -> None:
    """Greedy speculative decoding (ngram, draft_model, MTP) MUST match autoregressive decoding bit-for-bit!"""
    torch.manual_seed(30)
    target, tok = _model_and_tok(mtp_depth=2)
    draft = NovaLM(
        ModelConfig(
            vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_heads=2, n_kv_heads=2, max_seq_len=128, pad_vocab_multiple=1
        )
    )
    prompt = torch.tensor([tok.encode("Hello world! Hello world!", add_bos=True)], dtype=torch.long)
    params = SamplingParams(max_new_tokens=12, temperature=0.0)

    ar = generate(target, prompt, params=params, tokenizer=tok, use_cache=True)
    spec = speculative_generate(
        target,
        prompt,
        params=params,
        method=method,
        draft_model=draft if method == "draft_model" else None,
        num_speculative_tokens=2,
        tokenizer=tok,
    )
    assert torch.equal(ar.new_token_ids, spec.new_token_ids), (
        f"Speculative ({method}) diverged from greedy AR!\nAR:   {ar.new_token_ids}\nSpec: {spec.new_token_ids}"
    )


@pytest.mark.parametrize("bits", [8, 4])
def test_weight_only_quantization_and_qat(bits: int) -> None:
    """Verify INT8/INT4 group-wise weight quantization reduces memory and QAT flows gradients."""
    torch.manual_seed(31)
    model, _ = _model_and_tok(mtp_depth=0)
    x = torch.randint(0, 100, (1, 10))
    with torch.inference_mode():
        ref_logits = model(x).logits

    q_stats = quantize_model_weight_only(model, bits=bits, group_size=32)
    assert q_stats["layers_quantized"] > 0
    assert q_stats["compression_ratio"] > 1.2
    with torch.inference_mode():
        q_logits = model(x).logits
    assert torch.isfinite(q_logits).all()
    if bits == 8:
        assert (ref_logits - q_logits).abs().max().item() < 0.25

    # QAT STE gradient flow
    m_qat, _ = _model_and_tok(mtp_depth=0)
    wrapped = apply_qat(m_qat, bits=bits, group_size=32)
    assert wrapped > 0
    y = torch.randint(0, 100, (1, 10))
    out = m_qat(x, targets=y)
    out.loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in m_qat.parameters())
    assert has_grad


def test_safetensors_export_exact_parity(tmp_path: Path) -> None:
    """Exporting to SafeTensors + reloading must produce 0.0 logit difference."""
    torch.manual_seed(32)
    model, tok = _model_and_tok(mtp_depth=1)
    model.eval()
    export_dir = export_safetensors(model, tmp_path / "hf_out", tokenizer=tok)
    reloaded = load_safetensors_model(export_dir, device="cpu").eval()

    x = torch.randint(0, 100, (2, 12))
    with torch.inference_mode():
        diff = (model(x).logits - reloaded(x).logits).abs().max().item()
    assert diff == 0.0, f"SafeTensors export parity failed: diff={diff}"


def test_eval_metrics_and_fastapi_server(tmp_path: Path) -> None:
    """Verify BPB/PPL, MC, NIAH, JSON report, and OpenAI-compatible FastAPI endpoints."""
    torch.manual_seed(33)
    model, tok = _model_and_tok(mtp_depth=0)

    bpb = evaluate_bpb_and_ppl(model, tok, ["Transformers predict tokens efficiently."])
    assert bpb["bits_per_byte"] > 0 and bpb["perplexity"] > 1.0

    mc = evaluate_multiple_choice(
        model, tok, [{"prompt": "Hello ", "choices": ["world", "mars"], "answer_idx": 0}]
    )
    assert 0.0 <= mc["mc_accuracy"] <= 1.0

    niah = evaluate_niah(model, tok, context_lengths=[48], depths=[0.5])
    assert niah["niah_total"] == 1

    rep_path = tmp_path / "eval.json"
    rep = generate_eval_report(model, run_id="test_run", metrics={"bpb": bpb}, output_path=rep_path)
    assert rep_path.exists() and rep.model_params["total"] > 0

    # Test live OpenAI-compatible FastAPI endpoints (non-streaming + SSE streaming)
    app = create_app(model, tok, model_name="nova-tiny")
    client = TestClient(app)

    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/models").json()["data"][0]["id"] == "nova-tiny"

    r_sync = client.post(
        "/v1/chat/completions",
        json={"model": "nova-tiny", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8, "temperature": 0.0},
    )
    assert r_sync.status_code == 200
    body = r_sync.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] > 0

    r_stream = client.post(
        "/v1/chat/completions",
        json={"model": "nova-tiny", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 6, "temperature": 0.0, "stream": True},
    )
    assert r_stream.status_code == 200
    assert "data: [DONE]" in r_stream.text
