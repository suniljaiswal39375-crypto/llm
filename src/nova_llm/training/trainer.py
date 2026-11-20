"""Pre-training & Mid-Training engine with exact token-weighted grad accum, MuonClip, & spike guard."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from nova_llm.config import TrainConfig, save_config
from nova_llm.models.transformer import NovaLM
from nova_llm.optim.builder import build_optimizer
from nova_llm.optim.muon import apply_qk_clip_
from nova_llm.optim.scheduler import LRScheduler
from nova_llm.training.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from nova_llm.training.distributed import (
    DistributedContext,
    all_reduce_scalar,
    no_sync_context,
    wrap_model_distributed,
)
from nova_llm.utils.device import autocast_context, peak_memory_mb, resolve_device
from nova_llm.utils.logging import TelemetryLogger
from nova_llm.utils.seed import seed_everything


class PreTrainer:
    """Production-grade LLM pre-training & mid-training loop.

    Features:
    - Exact token-weighted gradient accumulation (zero bias with masked/packed sequences)
    - Muon / MuonClip (per-head QK-Clip) & AdamW with strict zero-WD on norms/biases/embeddings
    - WSD & Cosine LR schedules
    - Auxiliary-loss-free MoE dynamic router bias updates
    - NaN / loss-spike guard (prevents weight poisoning)
    - Deterministic full-state checkpointing & exact resume
    - Tokenizer-independent Bits-Per-Byte (BPB) validation telemetry
    """

    def __init__(
        self,
        model: NovaLM,
        train_loader: Any,
        cfg: TrainConfig,
        val_loader: Any | None = None,
        bytes_per_token: float | None = None,
        device: str | torch.device | None = None,
        dist_ctx: DistributedContext | None = None,
        quiet: bool = False,
    ) -> None:
        self.cfg = cfg
        self.dist_ctx = dist_ctx or DistributedContext()
        self.device = resolve_device(str(device) if device is not None else None)
        seed_everything(cfg.seed, rank=self.dist_ctx.rank)

        model.activation_checkpointing = cfg.activation_checkpointing
        model = model.to(self.device)
        if cfg.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)
        self.model = wrap_model_distributed(
            model, strategy=cfg.distributed, ctx=self.dist_ctx, precision=cfg.precision
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.bytes_per_token = bytes_per_token

        self.optimizer = build_optimizer(self.model, cfg.optim, device=self.device)
        self.scheduler = LRScheduler(
            optimizer=self.optimizer,
            schedule=cfg.optim.lr_schedule,
            total_steps=cfg.max_steps,
            warmup_steps=cfg.optim.warmup_steps,
            decay_steps=cfg.optim.decay_steps,
            min_lr_ratio=cfg.optim.min_lr_ratio,
        )

        self.use_grad_scaler = cfg.precision == "fp16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)

        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.dist_ctx.is_master:
            save_config(cfg, self.out_dir / "train_config.yaml")

        self.logger = TelemetryLogger(
            self.out_dir / "metrics.jsonl", rank=self.dist_ctx.rank, quiet=quiet
        )
        self.step = 0
        self.tokens_seen = 0
        self._ema_loss: float | None = None

    def resume(self, ckpt_path: str | Path) -> dict[str, Any]:
        info = load_checkpoint(
            ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loader=self.train_loader,
            restore_rng=True,
            map_location=self.device,
        )
        self.step = info["step"]
        self.tokens_seen = info["tokens_seen"]
        return info

    def save(self, name: str | None = None) -> Path:
        fname = name or f"ckpt_step_{self.step:06d}.pt"
        return save_checkpoint(
            self.out_dir / fname,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loader=self.train_loader,
            step=self.step,
            tokens_seen=self.tokens_seen,
        )

    @torch.inference_mode()
    def evaluate(self, max_batches: int | None = None) -> dict[str, float]:
        """Evaluate validation CE loss, perplexity, and Bits-Per-Byte (BPB) without mutating train state."""
        if self.val_loader is None:
            return {}
        n_batches = max_batches or self.cfg.eval_steps
        was_training = self.model.training
        self.model.eval()

        total_nll = 0.0
        total_tokens = 0
        for _ in range(n_batches):
            batch = self.val_loader.next_batch(device=self.device)
            with autocast_context(self.cfg.precision, self.device):
                out = self.model(
                    input_ids=batch["input_ids"],
                    targets=batch["targets"],
                    doc_ids=batch.get("doc_ids"),
                    attention_mask=batch.get("attention_mask"),
                    loss_reduction="sum",
                )
            total_nll += float(out.ce_loss.item())
            total_tokens += int(out.num_valid_tokens)

        if was_training:
            self.model.train()

        avg_loss = total_nll / max(1, total_tokens)
        avg_loss = all_reduce_scalar(avg_loss, self.dist_ctx, self.device)
        metrics: dict[str, float] = {
            "val_loss": avg_loss,
            "val_ppl": math.exp(min(20.0, avg_loss)),
        }
        if self.bytes_per_token and self.bytes_per_token > 0:
            # BPB = nats_per_token / (ln(2) * bytes_per_token)
            metrics["val_bpb"] = avg_loss / (math.log(2.0) * self.bytes_per_token)
        return metrics

    def train_step(self) -> dict[str, float]:
        """Execute ONE optimizer step over `grad_accum_steps` micro-batches with exact token weighting."""
        self.model.train()
        t0 = time.perf_counter()
        raw_model = unwrap_model(self.model)
        track_qk = self.cfg.optim.optimizer == "muonclip"

        # Pre-fetch micro-batches so we know the exact total valid token count across the window
        micro_batches = [
            self.train_loader.next_batch(device=self.device)
            for _ in range(self.cfg.grad_accum_steps)
        ]
        total_valid_tokens = sum(
            int((mb["targets"] != -100).sum().item()) for mb in micro_batches
        )
        if total_valid_tokens == 0:
            return {"loss": 0.0, "skipped_empty": 1.0}

        self.optimizer.zero_grad(set_to_none=True)
        lr_mult = self.scheduler.step(self.step)

        accum_total_loss = 0.0
        accum_ce_loss = 0.0
        accum_z_loss = 0.0

        for idx, mb in enumerate(micro_batches):
            is_last = idx == len(micro_batches) - 1
            with no_sync_context(self.model, enabled=not is_last):
                with autocast_context(self.cfg.precision, self.device):
                    out = self.model(
                        input_ids=mb["input_ids"],
                        targets=mb["targets"],
                        doc_ids=mb.get("doc_ids"),
                        attention_mask=mb.get("attention_mask"),
                        track_max_logits=track_qk,
                        loss_reduction="sum",
                    )
                    # Exact token-weighted normalization across full accumulation window
                    scaled_loss = out.loss / float(total_valid_tokens)

                if self.use_grad_scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            accum_total_loss += float(out.loss.detach().item())
            accum_ce_loss += float(out.ce_loss.detach().item())
            accum_z_loss += float(out.z_loss.detach().item())

        step_loss = accum_total_loss / float(total_valid_tokens)
        step_ce = accum_ce_loss / float(total_valid_tokens)
        step_z = accum_z_loss / float(total_valid_tokens)

        if self.use_grad_scaler:
            self.scaler.unscale_(self.optimizer)

        # Compute gradient norm & guard against NaN/Inf corrupting weights
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg.optim.grad_clip if self.cfg.optim.grad_clip > 0 else float("inf"),
            ).item()
        )

        skipped_nan = 0.0
        if not math.isfinite(step_loss) or not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            skipped_nan = 1.0
        else:
            if self.use_grad_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Kimi K2 MuonClip QK-Clip post-step weight rescaling
            if track_qk:
                qk_stats = apply_qk_clip_(raw_model, tau=self.cfg.optim.qk_clip_tau)
            else:
                qk_stats = {}

            # DeepSeek-V3 Auxiliary-Loss-Free MoE router bias update
            raw_model.update_moe_biases()

        self.step += 1
        self.tokens_seen += total_valid_tokens * self.dist_ctx.world_size
        dt = max(1e-6, time.perf_counter() - t0)

        spike_flag = 0.0
        if math.isfinite(step_loss):
            if self._ema_loss is not None and step_loss > self._ema_loss * self.cfg.spike_loss_multiplier:
                spike_flag = 1.0
            self._ema_loss = (
                step_loss if self._ema_loss is None else 0.95 * self._ema_loss + 0.05 * step_loss
            )

        metrics: dict[str, float] = {
            "loss": step_loss,
            "ce_loss": step_ce,
            "z_loss": step_z,
            "grad_norm": grad_norm if math.isfinite(grad_norm) else -1.0,
            "lr_mult": lr_mult,
            "tok_per_s": total_valid_tokens / dt,
            "tokens_seen": float(self.tokens_seen),
            "peak_mem_mb": peak_memory_mb(self.device),
            "spike": spike_flag,
            "skipped_nan": skipped_nan,
        }
        if not skipped_nan:
            metrics.update(qk_stats)
        metrics.update(raw_model.collect_telemetry())
        return metrics

    def fit(self, num_steps: int | None = None) -> list[dict[str, float]]:
        """Run pre-training loop up to `num_steps` (or `cfg.max_steps`)."""
        target_step = self.cfg.max_steps if num_steps is None else self.step + num_steps
        history: list[dict[str, float]] = []

        while self.step < target_step:
            metrics = self.train_step()
            history.append(metrics)

            if self.step % self.cfg.log_interval == 0 or self.step == target_step:
                self.logger.log(metrics, step=self.step)

            if self.val_loader is not None and (
                self.step % self.cfg.eval_interval == 0 or self.step == target_step
            ):
                val_metrics = self.evaluate()
                self.logger.log(val_metrics, step=self.step)

            if self.dist_ctx.is_master and self.cfg.save_interval > 0 and (
                self.step % self.cfg.save_interval == 0 or self.step == target_step
            ):
                self.save()

        return history
