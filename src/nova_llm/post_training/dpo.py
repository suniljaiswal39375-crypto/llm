"""Direct Preference Optimization (DPO), SimPO, and IPO with strict pair-completeness guards."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nova_llm.config import OptimConfig, PostTrainConfig
from nova_llm.models.transformer import NovaLM
from nova_llm.optim.builder import build_optimizer
from nova_llm.tokenizer.bpe import ByteBPETokenizer
from nova_llm.utils.device import resolve_device
from nova_llm.utils.seed import seed_everything


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


def sequence_logprobs(
    model: NovaLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sequence summed log-probabilities & valid response token counts in FP32."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(out.logits.float(), dim=-1)
    valid = targets != -100
    safe_targets = targets.clamp(min=0)
    token_lp = torch.gather(log_probs, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * valid.float()
    seq_sum = token_lp.sum(dim=-1)
    seq_len = valid.float().sum(dim=-1).clamp(min=1.0)
    return seq_sum, seq_len


def compute_preference_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor | None,
    ref_rejected_lp: torch.Tensor | None,
    chosen_lens: torch.Tensor,
    rejected_lens: torch.Tensor,
    method: str = "dpo",
    beta: float = 0.1,
    simpo_gamma: float = 0.5,
    nll_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DPO, SimPO (reference-free length-normalized), or IPO preference loss."""
    if method == "simpo":
        # SimPO (Meng et al., 2024): length-normalized log-prob margin without reference model
        r_chosen = (policy_chosen_lp / chosen_lens) * beta
        r_rejected = (policy_rejected_lp / rejected_lens) * beta
        margins = r_chosen - r_rejected - simpo_gamma
        loss = -F.logsigmoid(margins).mean()
    elif method in ("dpo", "ipo"):
        if ref_chosen_lp is None or ref_rejected_lp is None:
            raise ValueError(f"{method.upper()} requires reference logprobs")
        chosen_logratios = policy_chosen_lp - ref_chosen_lp
        rejected_logratios = policy_rejected_lp - ref_rejected_lp
        logits = chosen_logratios - rejected_logratios
        r_chosen = beta * chosen_logratios
        r_rejected = beta * rejected_logratios
        if method == "dpo":
            loss = -F.logsigmoid(beta * logits).mean()
        else:
            # IPO: squared margin loss preventing overfitting to deterministic preferences
            loss = (logits - 1.0 / (2.0 * beta)).pow(2).mean()
    else:
        raise ValueError(f"Unknown preference method: {method!r}")

    if nll_weight > 0.0:
        nll = -(policy_chosen_lp / chosen_lens).mean()
        loss = loss + nll_weight * nll

    with torch.no_grad():
        margin = (r_chosen - r_rejected).mean().item()
        win_acc = (r_chosen > r_rejected).float().mean().item()
    return loss, {
        "pref_loss": float(loss.detach().item()),
        "reward_margin": float(margin),
        "win_acc": float(win_acc),
    }


class DPOTrainer:
    """Preference optimization trainer guaranteeing intact (chosen, rejected) pairs per micro-batch."""

    def __init__(
        self,
        model: NovaLM,
        tokenizer: ByteBPETokenizer,
        examples: list[DPOExample],
        cfg: PostTrainConfig,
        ref_model: NovaLM | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if not examples:
            raise ValueError("DPOTrainer requires at least one preference pair")
        self.cfg = cfg
        self.device = resolve_device(str(device) if device is not None else None)
        seed_everything(cfg.seed)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.examples = examples

        if cfg.stage == "simpo":
            self.ref_model = None
        else:
            self.ref_model = (
                ref_model.to(self.device)
                if ref_model is not None
                else copy.deepcopy(self.model).to(self.device)
            )
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        opt_cfg = OptimConfig(
            optimizer="adamw", lr=cfg.lr, weight_decay=cfg.weight_decay, grad_clip=cfg.grad_clip
        )
        self.optimizer = build_optimizer(self.model, opt_cfg, device=self.device)
        self._cursor = 0

    def _encode_pair(self, ex: DPOExample) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
        chosen_msgs = [
            {"role": "user", "content": ex.prompt},
            {"role": "assistant", "content": ex.chosen},
        ]
        rejected_msgs = [
            {"role": "user", "content": ex.prompt},
            {"role": "assistant", "content": ex.rejected},
        ]
        c_pair = self.tokenizer.encode_chat_for_sft(chosen_msgs, max_seq_len=self.cfg.max_seq_len)
        r_pair = self.tokenizer.encode_chat_for_sft(rejected_msgs, max_seq_len=self.cfg.max_seq_len)
        return c_pair, r_pair

    def _collate_pairs(
        self, batch_ex: list[DPOExample]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Pair-completeness invariant: every index i in chosen corresponds to index i in rejected
        chosen_items = []
        rejected_items = []
        for ex in batch_ex:
            c, r = self._encode_pair(ex)
            chosen_items.append(c)
            rejected_items.append(r)

        assert len(chosen_items) == len(rejected_items) == len(batch_ex)

        def _pad(items: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
            max_l = max(len(x) for x, _ in items)
            inp = torch.full((len(items), max_l), self.tokenizer.pad_id, dtype=torch.long, device=self.device)
            lbl = torch.full((len(items), max_l), -100, dtype=torch.long, device=self.device)
            mask = torch.zeros((len(items), max_l), dtype=torch.bool, device=self.device)
            for i, (x, y) in enumerate(items):
                inp[i, : len(x)] = torch.tensor(x, dtype=torch.long, device=self.device)
                lbl[i, : len(y)] = torch.tensor(y, dtype=torch.long, device=self.device)
                mask[i, : len(x)] = True
            return {"input_ids": inp, "targets": lbl, "attention_mask": mask}

        return _pad(chosen_items), _pad(rejected_items)

    def train_step(self) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        agg_metrics = {"pref_loss": 0.0, "reward_margin": 0.0, "win_acc": 0.0}

        for _ in range(self.cfg.grad_accum_steps):
            batch_ex = []
            for _ in range(self.cfg.micro_batch_size):
                batch_ex.append(self.examples[self._cursor % len(self.examples)])
                self._cursor += 1
            chosen_b, rejected_b = self._collate_pairs(batch_ex)

            pol_c_lp, c_lens = sequence_logprobs(
                self.model, chosen_b["input_ids"], chosen_b["targets"], chosen_b["attention_mask"]
            )
            pol_r_lp, r_lens = sequence_logprobs(
                self.model, rejected_b["input_ids"], rejected_b["targets"], rejected_b["attention_mask"]
            )

            if self.ref_model is not None:
                with torch.no_grad():
                    ref_c_lp, _ = sequence_logprobs(
                        self.ref_model, chosen_b["input_ids"], chosen_b["targets"], chosen_b["attention_mask"]
                    )
                    ref_r_lp, _ = sequence_logprobs(
                        self.ref_model, rejected_b["input_ids"], rejected_b["targets"], rejected_b["attention_mask"]
                    )
            else:
                ref_c_lp, ref_r_lp = None, None

            loss, m = compute_preference_loss(
                policy_chosen_lp=pol_c_lp,
                policy_rejected_lp=pol_r_lp,
                ref_chosen_lp=ref_c_lp,
                ref_rejected_lp=ref_r_lp,
                chosen_lens=c_lens,
                rejected_lens=r_lens,
                method=self.cfg.stage,
                beta=self.cfg.dpo_beta,
                simpo_gamma=self.cfg.simpo_gamma,
                nll_weight=self.cfg.dpo_nll_weight,
            )
            (loss / float(self.cfg.grad_accum_steps)).backward()
            for k in agg_metrics:
                agg_metrics[k] += m[k] / float(self.cfg.grad_accum_steps)

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip).item()
        )
        self.optimizer.step()
        agg_metrics["grad_norm"] = grad_norm
        return agg_metrics

    def fit(self, num_steps: int | None = None) -> list[dict[str, float]]:
        steps = num_steps or self.cfg.max_steps
        return [self.train_step() for _ in range(steps)]
