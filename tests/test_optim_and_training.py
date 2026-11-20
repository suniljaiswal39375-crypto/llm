"""Verification proofs for Optimizers (MuonClip/QK-Clip, WD hygiene), Grad Accum, WSD & Resume."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nova_llm.config import ModelConfig, OptimConfig, TrainConfig
from nova_llm.data import PackedShardLoader, TokenShardReader, write_token_shard
from nova_llm.models import NovaLM
from nova_llm.models.norm import RMSNorm, SingleScaleRMSNorm
from nova_llm.optim import LRScheduler, apply_qk_clip_, build_optimizer
from nova_llm.tokenizer import ByteBPETokenizer
from nova_llm.training import PreTrainer


def _small_cfg(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        qk_norm=True,
        dropout=0.0,
        pad_vocab_multiple=1,
    )
    base.update(kwargs)
    return ModelConfig(**base)


@pytest.mark.parametrize("opt_name", ["adamw", "muon", "muonclip"])
def test_weight_decay_param_groups_strictly_zero_on_norms_biases_embeddings(opt_name: str) -> None:
    """NO norm, bias, 1D tensor, or embedding may EVER receive weight_decay > 0."""
    model = NovaLM(_small_cfg(sandwich_norm=True, tie_word_embeddings=False))
    opt = build_optimizer(model, OptimConfig(optimizer=opt_name, weight_decay=0.1))

    forbidden_ids: set[int] = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, (RMSNorm, SingleScaleRMSNorm, nn.LayerNorm, nn.Embedding)) or mod_name == "lm_head":
            for p in mod.parameters(recurse=False):
                forbidden_ids.add(id(p))
    for name, p in model.named_parameters():
        if p.ndim < 2 or name.endswith(".bias"):
            forbidden_ids.add(id(p))

    for group in opt.param_groups:
        wd = group.get("weight_decay", 0.0)
        if wd > 0.0:
            for p in group["params"]:
                assert id(p) not in forbidden_ids, f"Forbidden param received weight_decay={wd} in {opt_name}!"


def test_grad_accum_exact_equivalence_with_variable_masks() -> None:
    """1 step of batch 4 MUST match 2 accumulation steps of batch 2 EVEN with variable -100 masks!"""
    torch.manual_seed(10)
    cfg = _small_cfg(z_loss_weight=1e-4)
    m_full = NovaLM(cfg)
    m_acc = copy.deepcopy(m_full)

    x = torch.randint(0, 128, (4, 16))
    y = torch.randint(0, 128, (4, 16))
    # Intentionally make valid token counts unequal across the 2 micro-batches (14 vs 6 valid tokens)
    y[0, 10:] = -100
    y[1, 12:] = -100
    y[2, 3:] = -100
    y[3, 3:] = -100

    # Full batch forward/backward
    out_full = m_full(x, targets=y, loss_reduction="mean")
    out_full.loss.backward()

    # Exact token-weighted gradient accumulation over 2 micro-batches
    total_valid = int((y != -100).sum().item())
    for start in (0, 2):
        out_mb = m_acc(x[start : start + 2], targets=y[start : start + 2], loss_reduction="sum")
        (out_mb.loss / float(total_valid)).backward()

    for (n1, p1), (n2, p2) in zip(m_full.named_parameters(), m_acc.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None
        diff = (p1.grad - p2.grad).abs().max().item()
        assert diff < 1e-5, f"Grad accum mismatch on {n1}: max_diff={diff}"


@pytest.mark.parametrize(
    "arch_kwargs",
    [
        {"attention_type": "gqa", "qk_norm": False},
        {"attention_type": "mla", "qk_norm": False, "mla_q_lora_rank": 32, "mla_kv_lora_rank": 24, "mla_qk_nope_head_dim": 8, "mla_qk_rope_head_dim": 8, "mla_v_head_dim": 16},
    ],
)
def test_muonclip_qk_clip_rescales_exploding_heads(arch_kwargs: dict) -> None:
    """Verify Kimi K2 QK-Clip detects exploding attention logits and rescales Q/K weights cleanly."""
    torch.manual_seed(11)
    model = NovaLM(_small_cfg(**arch_kwargs))
    # Deliberately inflate Q/K weights to trigger exploding attention logits > tau
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "wq"):
                m.wq.weight.mul_(25.0)
            if hasattr(m, "wk"):
                m.wk.weight.mul_(25.0)
            if hasattr(m, "wq_b"):
                m.wq_b.weight.mul_(45.0)
            if hasattr(m, "wkv_b"):
                m.wkv_b.weight.mul_(45.0)

    x = torch.randint(0, 128, (2, 12))
    model(x, track_max_logits=True)
    tau = 20.0
    stats = apply_qk_clip_(model, tau=tau)
    assert stats["max_attn_logit_pre_clip"] > tau
    assert stats["qk_clipped_heads"] > 0

    # Re-run forward and confirm post-clip max logit dropped sharply
    model(x, track_max_logits=True)
    post = model.collect_telemetry()["max_attn_logit"]
    assert post < stats["max_attn_logit_pre_clip"] * 0.85


def test_wsd_and_cosine_scheduler() -> None:
    """Verify WSD warmup -> flat plateau -> smooth decay and Cosine schedule."""
    model = NovaLM(_small_cfg())
    opt = build_optimizer(model, OptimConfig(optimizer="muonclip", lr=1e-3, muon_lr=4e-3))
    sched = LRScheduler(opt, schedule="wsd", total_steps=20, warmup_steps=4, decay_steps=6, min_lr_ratio=0.1)

    mults = [sched.step(s) for s in range(20)]
    assert mults[0] == pytest.approx(0.25)
    assert mults[3] == pytest.approx(1.0)
    # Plateau steps 4..13 must stay flat at 1.0
    assert all(m == pytest.approx(1.0) for m in mults[4:14])
    # Tail steps 14..19 decay monotonically toward min_lr_ratio
    assert mults[-1] < 0.25
    assert mults[14] >= mults[16] >= mults[19]


def test_checkpoint_resume_determinism(tmp_path: Path) -> None:
    """Train 6 steps continuously vs train 3 -> save -> reload -> train 3: exact weight parity!"""
    docs = [f"Deterministic training document number {i} testing exact resume parity." for i in range(40)]
    tok = ByteBPETokenizer.train(docs, vocab_size=320, min_pair_freq=2)
    write_token_shard(docs, tok, tmp_path / "train")
    reader = TokenShardReader(tmp_path / "train")

    m_cfg = _small_cfg(vocab_size=tok.vocab_size, pad_vocab_multiple=1)
    t_cfg = TrainConfig(
        output_dir=str(tmp_path / "run_cont"),
        seed=77,
        max_steps=6,
        micro_batch_size=2,
        grad_accum_steps=2,
        seq_len=16,
        precision="fp32",
        log_interval=100,
        save_interval=0,
        model=m_cfg,
        optim=OptimConfig(optimizer="muonclip", lr=1e-3, muon_lr=3e-3, warmup_steps=2, decay_steps=2),
    )

    # Continuous 6 steps
    torch.manual_seed(77)
    m_cont = NovaLM(m_cfg)
    m_split_init = copy.deepcopy(m_cont)

    tr_cont = PreTrainer(
        m_cont, PackedShardLoader(reader, batch_size=2, seq_len=16, seed=77), t_cfg, device="cpu", quiet=True
    )
    hist_cont = tr_cont.fit(num_steps=6)

    # Split: 3 steps -> save -> fresh trainer -> resume -> 3 steps
    t_cfg_split = copy.deepcopy(t_cfg)
    t_cfg_split.output_dir = str(tmp_path / "run_split")
    tr_a = PreTrainer(
        m_split_init, PackedShardLoader(reader, batch_size=2, seq_len=16, seed=77), t_cfg_split, device="cpu", quiet=True
    )
    tr_a.fit(num_steps=3)
    ckpt = tr_a.save("mid.pt")

    tr_b = PreTrainer(
        NovaLM(m_cfg), PackedShardLoader(reader, batch_size=2, seq_len=16, seed=77), t_cfg_split, device="cpu", quiet=True
    )
    tr_b.resume(ckpt)
    hist_b = tr_b.fit(num_steps=3)

    assert abs(hist_cont[-1]["loss"] - hist_b[-1]["loss"]) < 1e-5
    for p1, p2 in zip(tr_cont.model.parameters(), tr_b.model.parameters()):
        assert (p1 - p2).abs().max().item() < 1e-5


def test_overfit_single_batch() -> None:
    """Sanity proof: model + MuonClip must overfit a single batch cleanly toward 0.0."""
    torch.manual_seed(12)
    model = NovaLM(_small_cfg(vocab_size=64, pad_vocab_multiple=1, z_loss_weight=0.0))
    opt = build_optimizer(model, OptimConfig(optimizer="muonclip", lr=5e-3, muon_lr=2e-2, weight_decay=0.0))

    x = torch.randint(0, 64, (2, 16))
    y = torch.roll(x, shifts=-1, dims=1)

    first_loss = None
    last_loss = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        out = model(x, targets=y)
        if first_loss is None:
            first_loss = out.ce_loss.item()
        out.loss.backward()
        opt.step()
        last_loss = out.ce_loss.item()

    assert first_loss is not None and last_loss is not None
    assert last_loss < 0.25 and last_loss < first_loss * 0.1, f"Failed to overfit: {first_loss} -> {last_loss}"
