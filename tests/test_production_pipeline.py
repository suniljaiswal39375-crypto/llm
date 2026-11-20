"""Tests for production LLM upgrades: HF Llama/Qwen2.5/Qwen3 weight import, JSONL datasets, and <think>/fetch_page."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from nova_llm.agent.search_agent import split_thinking_and_answer
from nova_llm.data import load_dpo_jsonl, load_rlvr_jsonl, load_sft_jsonl, stream_text_corpus
from nova_llm.models import hf_config_to_nova_config, load_hf_pretrained_into_nova


def _write_synthetic_hf_checkpoint(
    hf_dir: Path,
    model_type: str,
    tie_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    hf_dir.mkdir(parents=True, exist_ok=True)
    vocab, hidden, inter, layers, n_heads, n_kv = 128, 64, 176, 2, 4, 2
    head_dim = hidden // n_heads
    has_bias = model_type == "qwen2"
    has_qk_norm = model_type == "qwen3"

    cfg = {
        "model_type": model_type,
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "max_position_embeddings": 256,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": tie_embeddings,
        "attention_bias": has_bias,
    }
    (hf_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    torch.manual_seed(123)
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden) * 0.02,
        "model.norm.weight": torch.ones(hidden),
    }
    if not tie_embeddings:
        tensors["lm_head.weight"] = torch.randn(vocab, hidden) * 0.02

    for i in range(layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = torch.ones(hidden)
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.ones(hidden)
        tensors[f"{p}.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, hidden) * 0.02
        tensors[f"{p}.self_attn.k_proj.weight"] = torch.randn(n_kv * head_dim, hidden) * 0.02
        tensors[f"{p}.self_attn.v_proj.weight"] = torch.randn(n_kv * head_dim, hidden) * 0.02
        tensors[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden, n_heads * head_dim) * 0.02
        if has_bias:
            tensors[f"{p}.self_attn.q_proj.bias"] = torch.randn(n_heads * head_dim) * 0.01
            tensors[f"{p}.self_attn.k_proj.bias"] = torch.randn(n_kv * head_dim) * 0.01
            tensors[f"{p}.self_attn.v_proj.bias"] = torch.randn(n_kv * head_dim) * 0.01
        if has_qk_norm:
            tensors[f"{p}.self_attn.q_norm.weight"] = torch.ones(head_dim)
            tensors[f"{p}.self_attn.k_norm.weight"] = torch.ones(head_dim)
        tensors[f"{p}.mlp.gate_proj.weight"] = torch.randn(inter, hidden) * 0.02
        tensors[f"{p}.mlp.up_proj.weight"] = torch.randn(inter, hidden) * 0.02
        tensors[f"{p}.mlp.down_proj.weight"] = torch.randn(hidden, inter) * 0.02

    save_file(tensors, str(hf_dir / "model.safetensors"))
    return tensors


@pytest.mark.parametrize(
    "model_type,tie",
    [("llama", False), ("qwen2", True), ("qwen3", False), ("mistral", False)],
)
def test_hf_pretrained_import_exact_weight_mapping(tmp_path: Path, model_type: str, tie: bool) -> None:
    """Verify Llama/Qwen2.5/Qwen3/Mistral HF checkpoints load cleanly into NovaLM with exact weights."""
    hf_dir = tmp_path / f"hf_{model_type}"
    orig_tensors = _write_synthetic_hf_checkpoint(hf_dir, model_type=model_type, tie_embeddings=tie)

    model = load_hf_pretrained_into_nova(hf_dir, device="cpu").eval()
    assert torch.equal(model.embed_tokens.weight, orig_tensors["model.embed_tokens.weight"])
    assert torch.equal(model.layers[0].attn.wq.weight, orig_tensors["model.layers.0.self_attn.q_proj.weight"])
    if model_type == "qwen2":
        assert model.cfg.attention_bias is True
        assert model.layers[0].attn.wq.bias is not None
        assert torch.equal(model.layers[0].attn.wq.bias, orig_tensors["model.layers.0.self_attn.q_proj.bias"])
    if model_type == "qwen3":
        assert model.cfg.qk_norm is True
        assert model.layers[0].attn.q_norm is not None

    x = torch.randint(0, 128, (2, 10))
    with torch.inference_mode():
        logits = model(x).logits
    assert logits.shape == (2, 10, 128) and torch.isfinite(logits).all()


def test_real_streaming_dataset_loaders(tmp_path: Path) -> None:
    """Verify streaming text/JSONL, OpenAI/ShareGPT/Alpaca SFT, DPO, and RLVR loaders."""
    # Pre-training JSONL + TXT
    pt_jsonl = tmp_path / "corpus.jsonl"
    pt_jsonl.write_text('{"text": "First real document."}\n{"content": "Second real document."}\n', encoding="utf-8")
    docs = list(stream_text_corpus([pt_jsonl]))
    assert docs == ["First real document.", "Second real document."]

    # SFT multi-schema (OpenAI, ShareGPT, Alpaca)
    sft_file = tmp_path / "sft.jsonl"
    sft_file.write_text(
        "\n".join(
            [
                json.dumps({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}),
                json.dumps({"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]}),
                json.dumps({"instruction": "Add 2+2", "output": "4", "thinking": "2+2=4"}),
            ]
        ),
        encoding="utf-8",
    )
    convs = load_sft_jsonl(sft_file)
    assert len(convs) == 3
    assert convs[1][0]["role"] == "user" and convs[1][1]["role"] == "assistant"
    assert convs[2][1]["thinking"] == "2+2=4"

    # DPO JSONL
    dpo_file = tmp_path / "dpo.jsonl"
    dpo_file.write_text(json.dumps({"prompt": "2+2?", "chosen": "4", "rejected": "5"}) + "\n", encoding="utf-8")
    pairs = load_dpo_jsonl(dpo_file)
    assert len(pairs) == 1 and pairs[0].chosen == "4"

    # RLVR JSONL
    rl_file = tmp_path / "rlvr.jsonl"
    rl_file.write_text(json.dumps({"question": "What is 3*7?", "answer": "21", "type": "math"}) + "\n", encoding="utf-8")
    prompts = load_rlvr_jsonl(rl_file)
    assert len(prompts) == 1 and prompts[0].metadata["expected"] == "21"


def test_thinking_parser_separates_reasoning_from_visible_answer() -> None:
    think, ans = split_thinking_and_answer("<|think|>Step 1: check facts.\nStep 2: cite [1].</|think|> Final answer [1].")
    assert think == "Step 1: check facts.\nStep 2: cite [1]."
    assert ans == "Final answer [1]."
