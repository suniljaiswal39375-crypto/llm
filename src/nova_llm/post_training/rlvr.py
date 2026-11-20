"""Critic-Free RLVR Engine: GRPO, Dr. GRPO, DAPO (Clip-Higher + Dynamic Sampling), and GSPO.

References:
- GRPO: DeepSeekMath / DeepSeek-R1 (Shao et al., 2024; DeepSeek-AI, 2025)
- Dr. GRPO: Understanding R1-Zero-Like Training (Liu et al., 2025) — removes length & std biases
- DAPO: ByteDance Seed (arXiv:2503.14476, 2025) — Clip-Higher, Dynamic Sampling, token-level loss, overlong penalty
- GSPO: Group Sequence Policy Optimization (Qwen, 2025) — sequence-level IS ratio & clipping (MoE-safe!)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from nova_llm.config import OptimConfig, PostTrainConfig
from nova_llm.inference.generator import generate
from nova_llm.inference.sampler import SamplingParams
from nova_llm.models.transformer import NovaLM
from nova_llm.optim.builder import build_optimizer
from nova_llm.post_training.verifiers import CompositeVerifier
from nova_llm.tokenizer.bpe import ByteBPETokenizer
from nova_llm.utils.device import resolve_device
from nova_llm.utils.seed import seed_everything


@dataclass
class RLVRPrompt:
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_group_advantages(
    rewards: torch.Tensor,
    algorithm: str = "dapo",
) -> torch.Tensor:
    """Compute group-relative advantages for a group of G sampled rollouts.

    - `dr_grpo`: unbiased mean-centered only `r_i - mean(r)` (removes difficulty std bias)
    - `grpo` / `dapo` / `gspo`: `(r_i - mean(r)) / (std(r) + 1e-8)`
    """
    mean_r = rewards.mean()
    centered = rewards - mean_r
    if algorithm == "dr_grpo":
        return centered
    std_r = rewards.std(unbiased=False)
    return centered / (std_r + 1e-8)


def apply_overlong_penalty(
    rewards: torch.Tensor,
    lengths: torch.Tensor,
    max_len: int,
    buffer: int = 16,
    penalty: float = 0.2,
) -> torch.Tensor:
    """DAPO soft overlong punishment discouraging runaway verbose rollouts near max_len."""
    if penalty <= 0.0 or buffer <= 0:
        return rewards
    start = max(1, max_len - buffer)
    excess = torch.clamp((lengths.float() - start) / float(buffer), min=0.0, max=1.0)
    return rewards - penalty * excess


def compute_rlvr_loss(
    policy_logprobs: torch.Tensor,  # (G, T_resp)
    old_logprobs: torch.Tensor,  # (G, T_resp) detached
    response_mask: torch.Tensor,  # (G, T_resp) bool/float (1 on completion tokens, 0 on pad)
    advantages: torch.Tensor,  # (G,)
    algorithm: str = "dapo",
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    ref_logprobs: torch.Tensor | None = None,
    kl_beta: float = 0.0,
    max_gen_len: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute clipped policy-gradient loss for GRPO, Dr. GRPO, DAPO, or GSPO.

    - `grpo`: per-sample length-normalized token PPO clip + optional KL
    - `dr_grpo`: unbiased fixed-denominator sum (no 1/|o_i| length bias), symmetric clip
    - `dapo`: token-level loss aggregation across all active tokens + asymmetric Clip-Higher
    - `gspo`: sequence-level length-normalized importance ratio & sequence-level clipping
    """
    mask = response_mask.float()
    adv = advantages.detach()
    old_lp = old_logprobs.detach()
    log_ratio = (policy_logprobs - old_lp).clamp(-20.0, 20.0)

    # Optional Schulman unbiased KL approximation: exp(ref - pol) - (ref - pol) - 1
    kl_term = torch.zeros_like(policy_logprobs)
    if kl_beta > 0.0 and ref_logprobs is not None:
        diff = (ref_logprobs.detach() - policy_logprobs).clamp(-20.0, 20.0)
        kl_term = kl_beta * (torch.exp(diff) - diff - 1.0)

    if algorithm == "gspo":
        # Qwen GSPO: sequence-level length-normalized importance ratio aligns with sequence reward!
        seq_lens = mask.sum(dim=-1).clamp(min=1.0)  # (G,)
        seq_log_ratio = (log_ratio * mask).sum(dim=-1) / seq_lens  # (G,)
        seq_ratio = torch.exp(seq_log_ratio)
        clipped = torch.clamp(seq_ratio, 1.0 - clip_low, 1.0 + clip_high)
        surr1 = seq_ratio * adv
        surr2 = clipped * adv
        pg_loss = -torch.minimum(surr1, surr2).mean()
        if kl_beta > 0.0 and ref_logprobs is not None:
            pg_loss = pg_loss + ((kl_term * mask).sum(dim=-1) / seq_lens).mean()
        with torch.no_grad():
            clip_frac = ((seq_ratio < 1.0 - clip_low) | (seq_ratio > 1.0 + clip_high)).float().mean().item()
    else:
        ratio = torch.exp(log_ratio)
        hi = clip_high if algorithm == "dapo" else clip_low
        clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + hi)
        adv_tok = adv.unsqueeze(-1)
        surr = -torch.minimum(ratio * adv_tok, clipped * adv_tok) + kl_term

        if algorithm == "dapo":
            # Token-level loss: every valid token contributes equally regardless of sequence length
            denom = mask.sum().clamp(min=1.0)
            pg_loss = (surr * mask).sum() / denom
        elif algorithm == "dr_grpo":
            # Unbiased constant denominator (G * max_gen_len) eliminates 1/|o_i| length bias
            const_len = float(max_gen_len or mask.shape[1])
            pg_loss = (surr * mask).sum() / (float(mask.shape[0]) * max(1.0, const_len))
        elif algorithm == "grpo":
            seq_lens = mask.sum(dim=-1).clamp(min=1.0)
            pg_loss = ((surr * mask).sum(dim=-1) / seq_lens).mean()
        else:
            raise ValueError(f"Unknown RLVR algorithm: {algorithm!r}")

        with torch.no_grad():
            clipped_mask = ((ratio < 1.0 - clip_low) | (ratio > 1.0 + hi)) & (mask > 0)
            clip_frac = float(clipped_mask.float().sum().item() / max(1.0, float(mask.sum().item())))

    with torch.no_grad():
        mean_kl = float((kl_term * mask).sum().item() / max(1.0, float(mask.sum().item())))

    return pg_loss, {
        "rl_loss": float(pg_loss.detach().item()),
        "clip_frac": clip_frac,
        "kl_div": mean_kl,
    }


def _completion_token_logprobs_and_entropy(
    model: NovaLM,
    full_ids: torch.Tensor,
    prompt_len: int,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token logprobs and policy entropy strictly over generated completion tokens."""
    inp = full_ids[:, :-1]
    tgt = full_ids[:, 1:]
    mask_in = attention_mask[:, :-1]

    out = model(input_ids=inp, attention_mask=mask_in)
    logits = out.logits.float()
    # Completion predictions start at logit index (prompt_len - 1)
    start = max(0, prompt_len - 1)
    comp_logits = logits[:, start:, :]
    comp_targets = tgt[:, start:]

    log_probs = F.log_softmax(comp_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # (G, T_comp)
    tok_lp = torch.gather(log_probs, dim=-1, index=comp_targets.unsqueeze(-1)).squeeze(-1)
    return tok_lp, entropy


class RLVRTrainer:
    """End-to-end critic-free RLVR trainer supporting GRPO, Dr. GRPO, DAPO, and GSPO."""

    def __init__(
        self,
        model: NovaLM,
        tokenizer: ByteBPETokenizer,
        prompts: list[RLVRPrompt],
        cfg: PostTrainConfig,
        verifier: CompositeVerifier | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if cfg.stage not in ("grpo", "dr_grpo", "dapo", "gspo"):
            raise ValueError(f"RLVRTrainer expects stage in grpo/dr_grpo/dapo/gspo, got {cfg.stage!r}")
        self.cfg = cfg
        self.device = resolve_device(str(device) if device is not None else None)
        seed_everything(cfg.seed)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.prompts = prompts
        self.verifier = verifier or CompositeVerifier(format_weight=0.1)

        if cfg.rl_kl_beta > 0.0:
            self.ref_model: NovaLM | None = copy.deepcopy(self.model).to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        else:
            self.ref_model = None

        opt_cfg = OptimConfig(
            optimizer="adamw", lr=cfg.lr, weight_decay=cfg.weight_decay, grad_clip=cfg.grad_clip
        )
        self.optimizer = build_optimizer(self.model, opt_cfg, device=self.device)
        self._cursor = 0

    def train_step(self) -> dict[str, float]:
        """Sample G rollouts for a prompt, verify rewards, filter zero-variance groups, and update."""
        item = self.prompts[self._cursor % len(self.prompts)]
        self._cursor += 1

        prompt_text = self.tokenizer.format_chat(
            [{"role": "user", "content": item.prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids_1d = self.tokenizer.encode(prompt_text, add_bos=True, allow_special=True)
        prompt_len = len(prompt_ids_1d)
        prompt_t = torch.tensor([prompt_ids_1d], dtype=torch.long, device=self.device)

        stop_ids = [self.tokenizer.eos_id, self.tokenizer.special_to_id["<|im_end|>"]]
        sparams = SamplingParams(
            max_new_tokens=self.cfg.rl_max_new_tokens,
            temperature=self.cfg.rl_temperature,
            top_p=0.95,
            stop_token_ids=stop_ids,
        )

        # 1. Sample G rollouts from current policy
        rollout_seqs: list[list[int]] = []
        comp_lens: list[int] = []
        raw_rewards: list[float] = []
        for _ in range(self.cfg.rl_group_size):
            res = generate(self.model, prompt_t, params=sparams, tokenizer=self.tokenizer, use_cache=True)
            seq = res.token_ids[0].tolist()
            comp_ids = res.new_token_ids[0].tolist()
            comp_text = self.tokenizer.decode(comp_ids, skip_special_tokens=False)
            reward = self.verifier.score(comp_text, item.metadata)
            rollout_seqs.append(seq)
            comp_lens.append(max(1, len(comp_ids)))
            raw_rewards.append(reward)

        rewards_t = torch.tensor(raw_rewards, dtype=torch.float32, device=self.device)
        lens_t = torch.tensor(comp_lens, dtype=torch.long, device=self.device)

        if self.cfg.stage == "dapo":
            rewards_t = apply_overlong_penalty(
                rewards_t,
                lens_t,
                max_len=self.cfg.rl_max_new_tokens,
                buffer=self.cfg.rl_overlong_buffer,
                penalty=self.cfg.rl_overlong_penalty,
            )

        # DAPO Dynamic Sampling: skip prompts where all rollouts yield identical reward (zero gradient)
        if self.cfg.rl_dynamic_sampling and float(rewards_t.std(unbiased=False).item()) < 1e-6:
            return {
                "rl_loss": 0.0,
                "mean_reward": float(rewards_t.mean().item()),
                "mean_len": float(lens_t.float().mean().item()),
                "skipped_zero_adv": 1.0,
                "entropy": 0.0,
            }

        advantages = compute_group_advantages(rewards_t, algorithm=self.cfg.stage)

        # Pad group rollouts into rectangular tensor
        max_seq = max(len(s) for s in rollout_seqs)
        bsz = len(rollout_seqs)
        full_ids = torch.full((bsz, max_seq), self.tokenizer.pad_id, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros((bsz, max_seq), dtype=torch.bool, device=self.device)
        resp_mask = torch.zeros((bsz, max_seq - prompt_len), dtype=torch.float32, device=self.device)

        for i, seq in enumerate(rollout_seqs):
            full_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
            attn_mask[i, : len(seq)] = True
            c_len = max(0, len(seq) - prompt_len)
            resp_mask[i, :c_len] = 1.0

        # Snapshot old & reference logprobs (strictly detached)
        self.model.eval()
        with torch.no_grad():
            old_lp, _ = _completion_token_logprobs_and_entropy(
                self.model, full_ids, prompt_len, attn_mask
            )
            if self.ref_model is not None:
                ref_lp, _ = _completion_token_logprobs_and_entropy(
                    self.ref_model, full_ids, prompt_len, attn_mask
                )
            else:
                ref_lp = None

        # 2. Policy optimization epochs
        self.model.train()
        last_metrics: dict[str, float] = {}
        last_entropy = 0.0
        for _ in range(self.cfg.rl_ppo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            pol_lp, ent = _completion_token_logprobs_and_entropy(
                self.model, full_ids, prompt_len, attn_mask
            )
            loss, m = compute_rlvr_loss(
                policy_logprobs=pol_lp,
                old_logprobs=old_lp,
                response_mask=resp_mask,
                advantages=advantages,
                algorithm=self.cfg.stage,
                clip_low=self.cfg.rl_clip_low,
                clip_high=self.cfg.rl_clip_high,
                ref_logprobs=ref_lp,
                kl_beta=self.cfg.rl_kl_beta,
                max_gen_len=self.cfg.rl_max_new_tokens,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            last_metrics = m
            last_entropy = float(
                (ent.detach() * resp_mask).sum().item() / max(1.0, float(resp_mask.sum().item()))
            )

        last_metrics.update(
            {
                "mean_reward": float(rewards_t.mean().item()),
                "mean_len": float(lens_t.float().mean().item()),
                "skipped_zero_adv": 0.0,
                "entropy": last_entropy,
            }
        )
        return last_metrics

    def fit(self, num_steps: int | None = None) -> list[dict[str, float]]:
        steps = num_steps or self.cfg.max_steps
        return [self.train_step() for _ in range(steps)]
