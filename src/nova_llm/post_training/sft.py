"""Assistant-masked Supervised Fine-Tuning (SFT) with exact token-weighted grad accumulation."""

from __future__ import annotations

from typing import Any

import torch

from nova_llm.config import OptimConfig, PostTrainConfig
from nova_llm.models.lora import inject_lora
from nova_llm.models.transformer import NovaLM
from nova_llm.optim.builder import build_optimizer
from nova_llm.optim.scheduler import LRScheduler
from nova_llm.tokenizer.bpe import ByteBPETokenizer
from nova_llm.utils.device import autocast_context, resolve_device
from nova_llm.utils.seed import seed_everything


def collate_sft_examples(
    examples: list[tuple[list[int], list[int]]],
    pad_id: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Pad variable-length (input_ids, labels) pairs; pad labels are strictly -100."""
    max_len = max(len(x) for x, _ in examples)
    bsz = len(examples)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long, device=device)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)

    for i, (inp, lbl) in enumerate(examples):
        n = len(inp)
        input_ids[i, :n] = torch.tensor(inp, dtype=torch.long, device=device)
        labels[i, :n] = torch.tensor(lbl, dtype=torch.long, device=device)
        attn_mask[i, :n] = True

    return {"input_ids": input_ids, "targets": labels, "attention_mask": attn_mask}


class SFTTrainer:
    """Assistant-only masked SFT trainer supporting full FT, LoRA/DoRA, and exact masked grad accum."""

    def __init__(
        self,
        model: NovaLM,
        tokenizer: ByteBPETokenizer,
        conversations: list[list[dict[str, Any]]],
        cfg: PostTrainConfig,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(str(device) if device is not None else None)
        seed_everything(cfg.seed)

        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        if cfg.use_lora:
            inject_lora(
                self.model,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                use_dora=cfg.use_dora,
            )

        self.encoded: list[tuple[list[int], list[int]]] = []
        for conv in conversations:
            inp, lbl = tokenizer.encode_chat_for_sft(conv, max_seq_len=cfg.max_seq_len)
            if any(t != -100 for t in lbl):
                self.encoded.append((inp, lbl))
        if not self.encoded:
            raise ValueError("No SFT examples contain unmasked assistant target tokens!")

        opt_cfg = OptimConfig(
            optimizer="adamw",
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip,
        )
        self.optimizer = build_optimizer(self.model, opt_cfg, device=self.device)
        self.scheduler = LRScheduler(
            self.optimizer,
            schedule="cosine",
            total_steps=max(1, cfg.max_steps),
            warmup_steps=min(10, max(1, cfg.max_steps // 10)),
            min_lr_ratio=0.05,
        )
        self.global_step = 0
        self._cursor = 0

    def _next_micro_batch(self) -> dict[str, torch.Tensor]:
        batch_items = []
        for _ in range(self.cfg.micro_batch_size):
            batch_items.append(self.encoded[self._cursor % len(self.encoded)])
            self._cursor += 1
        return collate_sft_examples(batch_items, pad_id=self.tokenizer.pad_id, device=self.device)

    def train_step(self) -> dict[str, float]:
        """Run one SFT step with mathematically exact token-weighted gradient accumulation."""
        self.model.train()
        mbs = [self._next_micro_batch() for _ in range(self.cfg.grad_accum_steps)]
        total_valid = sum(int((mb["targets"] != -100).sum().item()) for mb in mbs)
        if total_valid == 0:
            return {"sft_loss": 0.0, "assistant_tokens": 0.0}

        self.optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for mb in mbs:
            with autocast_context(self.cfg.precision, self.device):
                out = self.model(
                    input_ids=mb["input_ids"],
                    targets=mb["targets"],
                    attention_mask=mb["attention_mask"],
                    loss_reduction="sum",
                )
                scaled = out.loss / float(total_valid)
            scaled.backward()
            accum_loss += float(out.loss.detach().item())

        self.scheduler.step(self.global_step)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip).item()
        )
        self.optimizer.step()
        self.global_step += 1
        return {
            "sft_loss": accum_loss / float(total_valid),
            "grad_norm": grad_norm,
            "assistant_tokens": float(total_valid),
        }

    def fit(self, num_steps: int | None = None) -> list[dict[str, float]]:
        steps = num_steps or self.cfg.max_steps
        return [self.train_step() for _ in range(steps)]
