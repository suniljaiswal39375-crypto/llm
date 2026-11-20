"""Mathematical correctness proofs for NovaLM architectures (GQA, MLA, SWA, Gated DeltaNet-2, MoE, MTP)."""

from __future__ import annotations

import pytest
import torch

from nova_llm.config import ModelConfig
from nova_llm.inference.kv_cache import InferenceCache
from nova_llm.models import NovaLM, RotaryEmbedding, apply_rotary_emb
from nova_llm.models.moe import DeepSeekMoE


def _tiny_cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        attention_type="gqa",
        qk_norm=True,
        dropout=0.0,
        pad_vocab_multiple=1,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.mark.parametrize(
    "arch_kwargs",
    [
        {"attention_type": "gqa", "qk_norm": True},
        {"attention_type": "gqa", "sliding_window": 4, "swa_ratio": 1},
        {"attention_type": "mla", "mla_q_lora_rank": 32, "mla_kv_lora_rank": 24, "mla_qk_nope_head_dim": 8, "mla_qk_rope_head_dim": 8, "mla_v_head_dim": 16},
        {"attention_type": "hybrid_linear", "linear_attn_ratio": 1, "linear_conv_kernel": 3},
    ],
)
def test_causal_mask_no_future_leakage(arch_kwargs: dict) -> None:
    """Perturbing token t+1 must change logits at positions 0..t by EXACTLY 0.0."""
    torch.manual_seed(0)
    model = NovaLM(_tiny_cfg(**arch_kwargs)).eval()
    x1 = torch.randint(0, 128, (2, 12))
    x2 = x1.clone()
    x2[:, 7:] = (x2[:, 7:] + 13) % 128  # perturb positions 7..11

    with torch.inference_mode():
        y1 = model(x1).logits
        y2 = model(x2).logits

    diff_prefix = (y1[:, :7, :] - y2[:, :7, :]).abs().max().item()
    diff_suffix = (y1[:, 7:, :] - y2[:, 7:, :]).abs().max().item()
    assert diff_prefix < 1e-6, f"Future token leaked into prefix logits! max_diff={diff_prefix}"
    assert diff_suffix > 1e-3, "Perturbing suffix had no effect on suffix logits!"


@pytest.mark.parametrize(
    "arch_kwargs",
    [
        {"attention_type": "gqa"},
        {"attention_type": "mla", "mla_q_lora_rank": 32, "mla_kv_lora_rank": 24, "mla_qk_nope_head_dim": 8, "mla_qk_rope_head_dim": 8, "mla_v_head_dim": 16},
        {"attention_type": "hybrid_linear", "linear_attn_ratio": 1, "linear_conv_kernel": 3},
    ],
)
def test_packed_document_isolation(arch_kwargs: dict) -> None:
    """Packed doc 1 must never leak attention or conv state into packed doc 2 (and vice versa)."""
    torch.manual_seed(1)
    model = NovaLM(_tiny_cfg(**arch_kwargs)).eval()
    # Two packed documents: tokens 0..5 belong to doc 0; tokens 6..11 belong to doc 1
    doc_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    x1 = torch.randint(0, 128, (1, 12))
    x2 = x1.clone()
    # Perturb doc 0 completely; doc 1 logits MUST remain identical!
    x2[:, :6] = (x2[:, :6] + 37) % 128

    with torch.inference_mode():
        y1 = model(x1, doc_ids=doc_ids).logits
        y2 = model(x2, doc_ids=doc_ids).logits

    doc1_diff = (y1[:, 6:, :] - y2[:, 6:, :]).abs().max().item()
    assert doc1_diff < 1e-5, f"Cross-document leakage from doc 0 into doc 1! diff={doc1_diff}"


@pytest.mark.parametrize(
    "arch_kwargs",
    [
        {"attention_type": "gqa"},
        {"attention_type": "gqa", "sliding_window": 4, "swa_ratio": 1},
        {"attention_type": "mla", "mla_q_lora_rank": 32, "mla_kv_lora_rank": 24, "mla_qk_nope_head_dim": 8, "mla_qk_rope_head_dim": 8, "mla_v_head_dim": 16},
        {"attention_type": "hybrid_linear", "linear_attn_ratio": 1, "linear_conv_kernel": 3},
    ],
)
def test_kv_cache_exact_equivalence(arch_kwargs: dict) -> None:
    """Prefill(prefix) + step-by-step cached decode(suffix) must match full-sequence forward."""
    torch.manual_seed(2)
    cfg = _tiny_cfg(**arch_kwargs)
    model = NovaLM(cfg).eval()
    seq = torch.randint(0, 128, (2, 10))

    with torch.inference_mode():
        full_logits = model(seq).logits

        cache = InferenceCache.allocate(cfg, batch_size=2, max_seq_len=32, device="cpu", dtype=torch.float32)
        prefill_logits = model(seq[:, :5], cache=cache).logits
        step_logits = [prefill_logits]
        for t in range(5, 10):
            step_logits.append(model(seq[:, t : t + 1], cache=cache).logits)
        cached_logits = torch.cat(step_logits, dim=1)

    max_diff = (full_logits - cached_logits).abs().max().item()
    assert max_diff < 1e-4, f"KV/MLA/Linear cache mismatch vs full forward! max_diff={max_diff}"


def test_rope_offset_and_relative_invariance() -> None:
    """Verify RoPE start_pos offset matches full slice & inner product depends strictly on (m - n)."""
    torch.manual_seed(3)
    rope = RotaryEmbedding(dim=16, max_seq_len=64, theta=10000.0)
    cos_full, sin_full = rope(seq_len=20, start_pos=0)
    cos_off, sin_off = rope(seq_len=5, start_pos=11)
    assert torch.allclose(cos_full[:, :, 11:16, :], cos_off, atol=1e-7)
    assert torch.allclose(sin_full[:, :, 11:16, :], sin_off, atol=1e-7)

    # Relative position invariance: <RoPE(q, m), RoPE(k, n)> == <RoPE(q, m+d), RoPE(k, n+d)>
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)
    cm, sm = rope(seq_len=1, start_pos=4)
    cn, sn = rope(seq_len=1, start_pos=9)
    qm, _ = apply_rotary_emb(q, q, cm, sm)
    _, kn = apply_rotary_emb(k, k, cn, sn)
    dot_1 = (qm * kn).sum().item()

    cm2, sm2 = rope(seq_len=1, start_pos=14)
    cn2, sn2 = rope(seq_len=1, start_pos=19)
    qm2, _ = apply_rotary_emb(q, q, cm2, sm2)
    _, kn2 = apply_rotary_emb(k, k, cn2, sn2)
    dot_2 = (qm2 * kn2).sum().item()
    assert abs(dot_1 - dot_2) < 1e-5, f"RoPE relative invariance violated: {dot_1} vs {dot_2}"

    # Verify YaRN scaling runs cleanly with finite outputs and mscale > 1.0
    yarn_rope = RotaryEmbedding(
        dim=16, max_seq_len=128, theta=10000.0, scaling_type="yarn", scaling_factor=4.0, orig_max_seq_len=32
    )
    assert yarn_rope.attention_factor > 1.0
    cy, sy = yarn_rope(seq_len=64, start_pos=0)
    assert torch.isfinite(cy).all() and torch.isfinite(sy).all()


def test_moe_aux_free_load_balancing_and_param_counts() -> None:
    """Verify DeepSeekMoE sparse param count and auxiliary-loss-free bias update outside autograd."""
    torch.manual_seed(4)
    cfg = _tiny_cfg(
        use_moe=True,
        moe_first_k_dense=0,
        n_routed_experts=4,
        moe_top_k=2,
        n_shared_experts=1,
        moe_aux_free_gamma=0.05,
    )
    model = NovaLM(cfg)
    counts = model.count_parameters()
    assert counts["active"] < counts["total"], "Sparse MoE must have fewer active than total params"

    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    out = model(x, targets=y)
    assert torch.isfinite(out.loss)

    moe_layer = model.layers[0].ffn
    assert isinstance(moe_layer, DeepSeekMoE)
    bias_before = moe_layer.expert_bias.clone()
    model.update_moe_biases()
    bias_after = moe_layer.expert_bias.clone()
    assert not torch.equal(bias_before, bias_after), "Aux-loss-free router bias did not update"
    assert moe_layer.expert_bias.requires_grad is False


def test_mtp_heads_and_osp_options() -> None:
    """Verify Multi-Token Prediction loss & draft head + Outlier-Safe Pre-Training (OSP) mode."""
    torch.manual_seed(5)
    cfg = _tiny_cfg(
        mtp_depth=2,
        mtp_loss_weight=0.3,
        norm_type="single_scale_rmsnorm",
        use_osp_embed_proj=True,
        sandwich_norm=True,
    )
    model = NovaLM(cfg)
    x = torch.randint(0, 128, (2, 12))
    y = torch.randint(0, 128, (2, 12))
    out = model(x, targets=y)
    assert out.mtp_loss is not None and out.mtp_loss.item() > 0.0
    out.loss.backward()
    assert model.osp_embed_proj is not None and model.osp_embed_proj.weight.grad is not None

    drafts = model.predict_mtp_draft(out.hidden_states[:1, -1:, :].detach(), x[:1, -1:])
    assert len(drafts) == 2 and drafts[0].shape == (1, 1)
