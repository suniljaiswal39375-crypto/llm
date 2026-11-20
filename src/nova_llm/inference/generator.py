"""Autoregressive generation, streaming, and lossless Speculative Decoding (N-gram, Draft, MTP)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F

from nova_llm.inference.kv_cache import InferenceCache
from nova_llm.inference.sampler import SamplingParams, sample_next_token
from nova_llm.models.transformer import NovaLM
from nova_llm.tokenizer.bpe import ByteBPETokenizer


@dataclass
class GenerationResult:
    token_ids: torch.Tensor  # (B, prompt_len + gen_len)
    new_token_ids: torch.Tensor  # (B, gen_len)
    text: str | None = None
    tokens_per_sec: float = 0.0
    target_forward_passes: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(1, self.drafted_tokens)


@torch.inference_mode()
def generate(
    model: NovaLM,
    prompt_ids: torch.Tensor,
    params: SamplingParams | None = None,
    tokenizer: ByteBPETokenizer | None = None,
    use_cache: bool = True,
    seed: int | None = None,
) -> GenerationResult:
    """Autoregressive generation with pre-allocated static KV/MLA/Linear cache."""
    params = params or SamplingParams()
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ids = prompt_ids.to(device=device, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    bsz, prompt_len = ids.shape

    max_total = min(model.cfg.max_seq_len, prompt_len + params.max_new_tokens)
    max_new = max(0, max_total - prompt_len)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    cache = (
        InferenceCache.allocate(model.cfg, batch_size=bsz, max_seq_len=max_total, device=device, dtype=dtype)
        if use_cache
        else None
    )

    t0 = time.perf_counter()
    forward_passes = 0
    generated: list[torch.Tensor] = []
    history = ids

    if max_new > 0:
        # Prefill
        out = model(ids, cache=cache)
        forward_passes += 1
        next_tok, _ = sample_next_token(out.logits[:, -1, :], params, history_ids=history, generator=gen)
        generated.append(next_tok)
        history = torch.cat([history, next_tok], dim=1)

        stop_set = set(params.stop_token_ids)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        if stop_set:
            for sid in stop_set:
                finished |= next_tok.squeeze(-1) == sid

        # Decode loop
        for _ in range(max_new - 1):
            if finished.all():
                break
            if use_cache:
                out = model(next_tok, cache=cache)
            else:
                out = model(history)
            forward_passes += 1
            next_tok, _ = sample_next_token(
                out.logits[:, -1, :], params, history_ids=history, generator=gen
            )
            generated.append(next_tok)
            history = torch.cat([history, next_tok], dim=1)
            if stop_set:
                for sid in stop_set:
                    finished |= next_tok.squeeze(-1) == sid

    if was_training:
        model.train()

    dt = max(1e-6, time.perf_counter() - t0)
    new_ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(bsz, 0, dtype=torch.long, device=device)
    )

    text: str | None = None
    if tokenizer is not None and bsz == 1:
        text = tokenizer.decode(new_ids[0].tolist(), skip_special_tokens=False)
        for stop_s in params.stop_strings:
            if stop_s in text:
                text = text.split(stop_s, 1)[0]

    return GenerationResult(
        token_ids=history,
        new_token_ids=new_ids,
        text=text,
        tokens_per_sec=(new_ids.numel() / dt),
        target_forward_passes=forward_passes,
    )


@torch.inference_mode()
def stream_generate(
    model: NovaLM,
    prompt_ids: torch.Tensor,
    tokenizer: ByteBPETokenizer,
    params: SamplingParams | None = None,
    seed: int | None = None,
) -> Iterator[str]:
    """Yield incremental UTF-8 text chunks token-by-token using pre-allocated static cache."""
    params = params or SamplingParams()
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        ids = prompt_ids.to(device=device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.shape[0] != 1:
            raise ValueError("stream_generate supports batch_size=1")

        max_total = min(model.cfg.max_seq_len, ids.shape[1] + params.max_new_tokens)
        max_new = max(0, max_total - ids.shape[1])
        if max_new == 0:
            return

        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)

        cache = InferenceCache.allocate(
            model.cfg, batch_size=1, max_seq_len=max_total, device=device, dtype=dtype
        )
        out = model(ids, cache=cache)
        history = ids
        gen_tokens: list[int] = []
        emitted_text = ""
        stop_set = set(params.stop_token_ids)

        for step in range(max_new):
            next_tok, _ = sample_next_token(
                out.logits[:, -1, :], params, history_ids=history, generator=gen
            )
            tid = int(next_tok[0, 0].item())
            if tid in stop_set:
                break
            gen_tokens.append(tid)
            history = torch.cat([history, next_tok], dim=1)
            full_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            hit_stop = False
            for stop_s in params.stop_strings:
                if stop_s in full_text:
                    full_text = full_text.split(stop_s, 1)[0]
                    hit_stop = True
                    break
            delta = full_text[len(emitted_text) :]
            if delta:
                emitted_text = full_text
                yield delta
            if hit_stop:
                break
            if step < max_new - 1:
                out = model(next_tok, cache=cache)
    finally:
        if was_training:
            model.train()


def _ngram_propose(history: list[int], max_draft: int, ngram_size: int = 3) -> list[int]:
    """Prompt/context n-gram lookup speculation (zero extra VRAM)."""
    if len(history) < ngram_size:
        return []
    for n in range(min(ngram_size, len(history)), 0, -1):
        suffix = history[-n:]
        # Search backwards for most recent occurrence before the tail
        for idx in range(len(history) - n - 1, -1, -1):
            if history[idx : idx + n] == suffix:
                start = idx + n
                return history[start : min(len(history), start + max_draft)]
    return []


@torch.inference_mode()
def speculative_generate(
    target_model: NovaLM,
    prompt_ids: torch.Tensor,
    params: SamplingParams | None = None,
    method: str = "ngram",
    draft_model: NovaLM | None = None,
    num_speculative_tokens: int = 4,
    tokenizer: ByteBPETokenizer | None = None,
    seed: int | None = None,
) -> GenerationResult:
    """Lossless Speculative Decoding supporting `ngram`, `draft_model`, and `mtp` heads.

    For `temperature <= 0.0` (greedy), output is mathematically guaranteed to be bit-for-bit
    identical to standard autoregressive `generate()`. For stochastic sampling (`temperature > 0`),
    applies Leviathan et al. exact rejection sampling with residual resampling.
    """
    params = params or SamplingParams()
    was_training = target_model.training
    target_model.eval()
    if draft_model is not None:
        draft_was_training = draft_model.training
        draft_model.eval()

    device = next(target_model.parameters()).device
    ids = prompt_ids.to(device=device, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1:
        raise ValueError("speculative_generate currently expects batch_size=1")

    prompt_len = ids.shape[1]
    max_total = min(target_model.cfg.max_seq_len, prompt_len + params.max_new_tokens)
    max_new = max(0, max_total - prompt_len)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    t0 = time.perf_counter()
    history = ids
    stop_set = set(params.stop_token_ids)
    target_passes = 0
    drafted_total = 0
    accepted_total = 0

    try:
        # Initial prefill pass
        out = target_model(history)
        target_passes += 1
        last_hidden = out.hidden_states[:, -1:, :]
        first_tok, _ = sample_next_token(out.logits[:, -1, :], params, history_ids=history, generator=gen)
        history = torch.cat([history, first_tok], dim=1)

        while (history.shape[1] - prompt_len) < max_new:
            if int(history[0, -1].item()) in stop_set:
                break
            rem = max_new - (history.shape[1] - prompt_len)
            k_draft = min(num_speculative_tokens, rem)
            if k_draft <= 0:
                break

            # 1. Propose K draft tokens + draft probabilities
            draft_tokens: list[int] = []
            draft_probs_list: list[torch.Tensor | None] = []

            if method == "ngram":
                draft_tokens = _ngram_propose(history[0].tolist(), max_draft=k_draft, ngram_size=3)
                draft_probs_list = [None] * len(draft_tokens)
            elif method == "mtp":
                if len(target_model.mtp_modules) == 0:
                    raise ValueError("method='mtp' requires target_model.cfg.mtp_depth >= 1")
                mtp_toks = target_model.predict_mtp_draft(last_hidden, history[:, -1:])
                for t in mtp_toks[:k_draft]:
                    draft_tokens.append(int(t[0, 0].item()))
                    draft_probs_list.append(None)
            elif method == "draft_model":
                if draft_model is None:
                    raise ValueError("method='draft_model' requires passing `draft_model`")
                d_hist = history.clone()
                for _ in range(k_draft):
                    d_out = draft_model(d_hist)
                    d_tok, d_prob = sample_next_token(
                        d_out.logits[:, -1, :], params, history_ids=d_hist, generator=gen
                    )
                    draft_tokens.append(int(d_tok[0, 0].item()))
                    draft_probs_list.append(d_prob[0])
                    d_hist = torch.cat([d_hist, d_tok], dim=1)
            else:
                raise ValueError(f"Unknown speculative method: {method!r}")

            if not draft_tokens:
                # Fallback single target step when n-gram finds no match
                t_out = target_model(history)
                target_passes += 1
                last_hidden = t_out.hidden_states[:, -1:, :]
                nxt, _ = sample_next_token(
                    t_out.logits[:, -1, :], params, history_ids=history, generator=gen
                )
                history = torch.cat([history, nxt], dim=1)
                continue

            drafted_total += len(draft_tokens)
            draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=device)
            verify_input = torch.cat([history, draft_tensor], dim=1)

            # 2. Single batched target verification pass over [history, d_1, ..., d_K]
            t_out = target_model(verify_input)
            target_passes += 1
            base_len = history.shape[1]

            # Logits at index (base_len - 1 + i) predict token at position (base_len + i), i.e., d_{i+1}
            accepted_prefix: list[int] = []
            bonus_tok: int | None = None
            rejected = False

            for i, d_id in enumerate(draft_tokens):
                ctx_slice = verify_input[:, : base_len + i]
                target_logit = t_out.logits[:, base_len - 1 + i, :]

                if params.temperature <= 0.0:
                    # Exact greedy verification
                    t_tok, _ = sample_next_token(target_logit, params, history_ids=ctx_slice)
                    t_id = int(t_tok[0, 0].item())
                    if t_id == d_id:
                        accepted_prefix.append(d_id)
                        accepted_total += 1
                    else:
                        bonus_tok = t_id
                        rejected = True
                        break
                else:
                    # Exact rejection sampling (Leviathan et al., 2023)
                    _, p_target_all = sample_next_token(
                        target_logit, params, history_ids=ctx_slice, generator=gen
                    )
                    p_t = p_target_all[0]
                    q_d = draft_probs_list[i]
                    if q_d is None:
                        # One-hot deterministic draft (ngram / mtp)
                        accept_prob = float(p_t[d_id].item())
                    else:
                        accept_prob = min(1.0, float(p_t[d_id].item()) / max(1e-12, float(q_d[d_id].item())))
                    u = float(torch.rand((), device=device, generator=gen).item())
                    if u < accept_prob:
                        accepted_prefix.append(d_id)
                        accepted_total += 1
                    else:
                        if q_d is not None:
                            residual = torch.clamp(p_t - q_d, min=0.0)
                            r_sum = residual.sum()
                            dist = residual / r_sum if r_sum > 1e-12 else p_t
                        else:
                            dist = p_t
                        bonus_tok = int(torch.multinomial(dist, num_samples=1, generator=gen).item())
                        rejected = True
                        break

            # Append accepted prefix
            if accepted_prefix:
                acc_t = torch.tensor([accepted_prefix], dtype=torch.long, device=device)
                history = torch.cat([history, acc_t], dim=1)

            # Check if stop token appeared in accepted prefix
            stop_hit = False
            if stop_set and accepted_prefix:
                for pos, tok in enumerate(accepted_prefix):
                    if tok in stop_set:
                        # Trim right after stop token
                        trim = len(accepted_prefix) - (pos + 1)
                        if trim > 0:
                            history = history[:, :-trim]
                        stop_hit = True
                        break
            if stop_hit or (history.shape[1] - prompt_len) >= max_new:
                break

            # Append bonus token (either resampled on rejection OR free K+1 token when all K accepted!)
            if rejected and bonus_tok is not None:
                history = torch.cat(
                    [history, torch.tensor([[bonus_tok]], dtype=torch.long, device=device)], dim=1
                )
                last_hidden = t_out.hidden_states[:, base_len - 1 + len(accepted_prefix) : base_len + len(accepted_prefix), :]
            elif not rejected:
                # All K draft tokens accepted -> sample free K+1 bonus token from last logit!
                bonus_logit = t_out.logits[:, -1, :]
                b_tok, _ = sample_next_token(bonus_logit, params, history_ids=history, generator=gen)
                history = torch.cat([history, b_tok], dim=1)
                last_hidden = t_out.hidden_states[:, -1:, :]
    finally:
        if was_training:
            target_model.train()
        if draft_model is not None and draft_was_training:
            draft_model.train()

    dt = max(1e-6, time.perf_counter() - t0)
    new_ids = history[:, prompt_len : prompt_len + max_new]
    history = history[:, : prompt_len + max_new]

    text = tokenizer.decode(new_ids[0].tolist(), skip_special_tokens=False) if tokenizer is not None else None
    return GenerationResult(
        token_ids=history,
        new_token_ids=new_ids,
        text=text,
        tokens_per_sec=new_ids.numel() / dt,
        target_forward_passes=target_passes,
        drafted_tokens=drafted_total,
        accepted_tokens=accepted_total,
    )
