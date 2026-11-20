"""Sampling engine: greedy, temperature, top-k, top-p (off-by-one safe), min-p, and penalties."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    stop_token_ids: list[int] = field(default_factory=list)
    stop_strings: list[str] = field(default_factory=list)


def _apply_penalties(
    logits: torch.Tensor,
    history_ids: torch.Tensor | None,
    repetition_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    if history_ids is None or (repetition_penalty == 1.0 and presence_penalty == 0.0):
        return logits
    out = logits.clone()
    bsz = out.shape[0]
    for b in range(bsz):
        unique_ids = torch.unique(history_ids[b])
        if repetition_penalty != 1.0:
            vals = out[b, unique_ids]
            # CTRL / HF convention: divide positive logits, multiply negative logits
            vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            out[b, unique_ids] = vals
        if presence_penalty != 0.0:
            out[b, unique_ids] = out[b, unique_ids] - presence_penalty
    return out


def sample_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
    history_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample next token IDs `(B, 1)` and return `(next_token, probs_fp32)`.

    Guarantees:
    - Exact greedy `argmax` when `temperature <= 0.0` (never divides by zero).
    - Top-p and min-p ALWAYS preserve at least the top-1 token (no empty-support NaNs).
    """
    logits_f = _apply_penalties(
        logits.float(), history_ids, params.repetition_penalty, params.presence_penalty
    )

    if params.temperature <= 0.0:
        next_tok = logits_f.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits_f, dim=-1)
        return next_tok, probs

    scaled = logits_f / float(params.temperature)

    if params.top_k > 0 and params.top_k < scaled.shape[-1]:
        kth_vals = torch.topk(scaled, k=params.top_k, dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < kth_vals, float("-inf"))

    if params.min_p > 0.0:
        probs_pre = F.softmax(scaled, dim=-1)
        max_p = probs_pre.max(dim=-1, keepdim=True).values
        threshold = max_p * float(params.min_p)
        mask_out = probs_pre < threshold
        # Always keep the argmax token
        top1_idx = probs_pre.argmax(dim=-1, keepdim=True)
        mask_out.scatter_(dim=-1, index=top1_idx, value=False)
        scaled = scaled.masked_fill(mask_out, float("-inf"))

    if 0.0 < params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Shift right by 1 so the first token crossing top_p is ALWAYS kept (prevents off-by-one bug)
        sorted_remove = cum_probs > float(params.top_p)
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove_mask = torch.zeros_like(sorted_remove)
        remove_mask.scatter_(dim=-1, index=sorted_idx, src=sorted_remove)
        scaled = scaled.masked_fill(remove_mask, float("-inf"))

    probs = F.softmax(scaled, dim=-1)
    # Guard numerical edge cases
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    next_tok = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_tok, probs
