"""Tokenizer-independent Bits-Per-Byte (BPB), perplexity, and log-likelihood MC evaluation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from nova_llm.models.transformer import NovaLM
from nova_llm.tokenizer.bpe import ByteBPETokenizer


@torch.inference_mode()
def evaluate_bpb_and_ppl(
    model: NovaLM,
    tokenizer: ByteBPETokenizer,
    texts: list[str],
    max_seq_len: int | None = None,
) -> dict[str, float]:
    """Compute exact total negative log-likelihood, perplexity, and tokenizer-independent BPB."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    cap = max_seq_len or model.cfg.max_seq_len

    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0

    try:
        for text in texts:
            if not text:
                continue
            ids = tokenizer.encode(text, add_bos=True, add_eos=True, allow_special=False)
            if len(ids) < 2:
                continue
            # Evaluate in non-overlapping chunks up to cap
            for start in range(0, len(ids) - 1, cap - 1):
                chunk = ids[start : start + cap]
                if len(chunk) < 2:
                    break
                inp = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
                tgt = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
                out = model(input_ids=inp, targets=tgt, loss_reduction="sum")
                total_nll += float(out.ce_loss.item())
                total_tokens += int(out.num_valid_tokens)
            total_bytes += len(text.encode("utf-8"))
    finally:
        if was_training:
            model.train()

    avg_nll = total_nll / max(1, total_tokens)
    bpb = total_nll / max(1e-9, math.log(2.0) * max(1, total_bytes))
    return {
        "nll_per_token": avg_nll,
        "perplexity": math.exp(min(20.0, avg_nll)),
        "bits_per_byte": bpb,
        "evaluated_tokens": float(total_tokens),
        "evaluated_bytes": float(total_bytes),
    }


@torch.inference_mode()
def evaluate_multiple_choice(
    model: NovaLM,
    tokenizer: ByteBPETokenizer,
    items: list[dict[str, Any]],
) -> dict[str, float]:
    """Length-normalized log-likelihood evaluation for ARC / HellaSwag / MMLU-Pro style items.

    Each item: {"prompt": str, "choices": list[str], "answer_idx": int}
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    try:
        for item in items:
            p_ids = tokenizer.encode(item["prompt"], add_bos=True, allow_special=True)
            best_idx = 0
            best_score = float("-inf")

            for c_idx, choice in enumerate(item["choices"]):
                c_ids = tokenizer.encode(choice, allow_special=False)
                if not c_ids:
                    continue
                full = (p_ids + c_ids)[-model.cfg.max_seq_len :]
                c_len = min(len(c_ids), len(full) - 1)
                if c_len <= 0:
                    continue
                inp = torch.tensor([full[:-1]], dtype=torch.long, device=device)
                tgt = torch.tensor([full[1:]], dtype=torch.long, device=device)
                out = model(input_ids=inp)
                lp = F.log_softmax(out.logits.float(), dim=-1)
                tok_lp = torch.gather(lp, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)[0]
                choice_lp = tok_lp[-c_len:].mean().item()
                if choice_lp > best_score:
                    best_score = choice_lp
                    best_idx = c_idx

            if best_idx == int(item["answer_idx"]):
                correct += 1
    finally:
        if was_training:
            model.train()

    return {
        "mc_accuracy": correct / max(1, len(items)),
        "mc_correct": float(correct),
        "mc_total": float(len(items)),
    }
