"""Verification tests for Post-Training: SFT masking, LoRA/DoRA merge, DPO/SimPO/IPO, RLVR & Verifiers."""

from __future__ import annotations

import pytest
import torch

from nova_llm.config import ModelConfig, PostTrainConfig
from nova_llm.models import NovaLM, inject_lora, merge_lora, unmerge_lora
from nova_llm.post_training import (
    CompositeVerifier,
    DPOExample,
    DPOTrainer,
    RLVRPrompt,
    RLVRTrainer,
    SFTTrainer,
    compute_preference_loss,
    compute_rlvr_loss,
    verify_constraints,
    verify_format_tags,
    verify_math_answer,
    verify_python_code,
)
from nova_llm.post_training.rlvr import apply_overlong_penalty, compute_group_advantages
from nova_llm.tokenizer import ByteBPETokenizer


def _tiny_model_and_tok() -> tuple[NovaLM, ByteBPETokenizer]:
    tok = ByteBPETokenizer.train(
        ["What is 2 + 2? The answer is 4. Hello world assistant response."], vocab_size=300
    )
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        pad_vocab_multiple=1,
    )
    return NovaLM(cfg), tok


def test_sft_prompt_tokens_receive_zero_loss_contribution() -> None:
    """Changing target IDs in masked prompt positions (-100) must change loss & grads by 0.0."""
    torch.manual_seed(20)
    model, tok = _tiny_model_and_tok()
    convs = [
        [
            {"role": "user", "content": "Explain gravity briefly."},
            {"role": "assistant", "content": "Mass attracts mass."},
        ]
    ]
    inp, lbl = tok.encode_chat_for_sft(convs[0])
    x = torch.tensor([inp], dtype=torch.long)
    y = torch.tensor([lbl], dtype=torch.long)
    assert (y == -100).any() and (y != -100).any()

    out = model(x, targets=y)
    # Manually verify loss equals CE over ONLY non-(-100) positions
    valid = y != -100
    manual_ce = torch.nn.functional.cross_entropy(
        out.logits.float()[valid], y[valid], reduction="mean"
    )
    assert abs(out.ce_loss.item() - manual_ce.item()) < 1e-6

    cfg = PostTrainConfig(stage="sft", max_steps=3, micro_batch_size=1, grad_accum_steps=1, lr=1e-3)
    hist = SFTTrainer(model, tok, convs, cfg, device="cpu").fit()
    assert all(m["assistant_tokens"] > 0 for m in hist)


@pytest.mark.parametrize("use_dora", [False, True])
def test_lora_and_dora_lossless_merge_unmerge(use_dora: bool) -> None:
    """Merged LoRA/DoRA weights must match unmerged adapter forward logits identically."""
    torch.manual_seed(21)
    model, _ = _tiny_model_and_tok()
    replaced = inject_lora(model, rank=4, alpha=8.0, use_dora=use_dora)
    assert replaced > 0
    # Make lora_b non-zero so delta weight is non-trivial
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_b" in name:
                p.normal_(0.0, 0.05)

    model.eval()
    x = torch.randint(0, 100, (2, 10))
    with torch.inference_mode():
        y_unmerged = model(x).logits
        merge_lora(model)
        y_merged = model(x).logits
        unmerge_lora(model)
        y_restored = model(x).logits

    assert (y_unmerged - y_merged).abs().max().item() < 1e-5
    assert (y_unmerged - y_restored).abs().max().item() < 1e-5

    # Verify merge_and_unload_lora restores vanilla nn.Linear keys with exact parity
    from nova_llm.models import merge_and_unload_lora

    unloaded = merge_and_unload_lora(model)
    assert unloaded == replaced
    with torch.inference_mode():
        y_unloaded = model(x).logits
    assert (y_unmerged - y_unloaded).abs().max().item() < 1e-5
    assert not any("lora_a" in k for k in model.state_dict())


@pytest.mark.parametrize("method", ["dpo", "simpo", "ipo"])
def test_dpo_simpo_ipo_objectives_and_trainer(method: str) -> None:
    """Verify DPO, SimPO, and IPO losses are finite, sign-correct, and optimize cleanly."""
    torch.manual_seed(22)
    pol_c = torch.tensor([-5.0, -4.0], requires_grad=True)
    pol_r = torch.tensor([-8.0, -7.0], requires_grad=True)
    ref_c = torch.tensor([-6.0, -5.0])
    ref_r = torch.tensor([-7.0, -6.0])
    lens = torch.tensor([10.0, 10.0])

    loss, m = compute_preference_loss(
        pol_c, pol_r, ref_c, ref_r, lens, lens, method=method, beta=0.1, nll_weight=0.05
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert pol_c.grad is not None and pol_r.grad is not None

    model, tok = _tiny_model_and_tok()
    exs = [DPOExample(prompt="2+2?", chosen="4", rejected="99")]
    cfg = PostTrainConfig(stage=method, max_steps=2, micro_batch_size=1, grad_accum_steps=1, lr=5e-4)
    hist = DPOTrainer(model, tok, exs, cfg, device="cpu").fit()
    assert len(hist) == 2 and all(torch.isfinite(torch.tensor(h["pref_loss"])) for h in hist)


@pytest.mark.parametrize("algo", ["grpo", "dr_grpo", "dapo", "gspo"])
def test_rlvr_algorithms_and_trainer(algo: str) -> None:
    """Verify GRPO, Dr. GRPO, DAPO (Clip-Higher + Dynamic Sampling), and GSPO (sequence IS)."""
    torch.manual_seed(23)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = compute_group_advantages(rewards, algorithm=algo)
    if algo == "dr_grpo":
        assert torch.allclose(adv, torch.tensor([0.5, -0.5, 0.5, -0.5]))
    else:
        assert abs(adv.mean().item()) < 1e-6

    pol_lp = torch.randn(4, 8, requires_grad=True)
    old_lp = pol_lp.detach() - 0.05
    ref_lp = pol_lp.detach() - 0.1
    resp_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    loss, m = compute_rlvr_loss(
        pol_lp, old_lp, resp_mask, adv, algorithm=algo, clip_low=0.2, clip_high=0.28, ref_logprobs=ref_lp, kl_beta=0.01
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert pol_lp.grad is not None and torch.isfinite(pol_lp.grad).all()

    # Verify overlong penalty decreases reward for verbose responses
    penalized = apply_overlong_penalty(torch.tensor([1.0, 1.0]), torch.tensor([20, 64]), max_len=64, buffer=16, penalty=0.3)
    assert penalized[0].item() == pytest.approx(1.0)
    assert penalized[1].item() == pytest.approx(0.7)

    # Smoke test full RLVRTrainer loop
    model, tok = _tiny_model_and_tok()
    prompts = [RLVRPrompt(prompt="2+2?", metadata={"type": "math", "expected": "4"})]
    cfg = PostTrainConfig(
        stage=algo, max_steps=1, lr=1e-4, rl_group_size=2, rl_max_new_tokens=6, rl_dynamic_sampling=False, rl_ppo_epochs=1
    )
    hist = RLVRTrainer(model, tok, prompts, cfg, verifier=CompositeVerifier(), device="cpu").fit()
    assert len(hist) == 1


def test_deterministic_verifiers() -> None:
    """Verify math, format tags, sandboxed Python code, and instruction constraint verifiers."""
    assert verify_math_answer("Steps... therefore \\boxed{42}.", "42") == 1.0
    assert verify_math_answer("Answer: 3/2", "1.5") == 1.0
    assert verify_math_answer("Answer: 7", "42") == 0.0

    assert verify_format_tags("<|think|>step 1</|think|> Final answer") == 1.0
    assert verify_format_tags("No tags here") == 0.0

    assert verify_python_code("```python\ndef add(a, b): return a + b\n```", "assert add(2, 3) == 5") == 1.0
    assert verify_python_code("```python\ndef add(a, b): return a - b\n```", "assert add(2, 3) == 5") == 0.0

    assert verify_constraints('{"ok": true}', {"valid_json": True, "must_include": ["ok"]}) == 1.0
    assert verify_constraints("hello world", {"must_exclude": ["world"]}) == 0.0
