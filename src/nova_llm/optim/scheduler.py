"""Warmup-Stable-Decay (WSD) and Cosine learning-rate schedules supporting hybrid optimizers."""

from __future__ import annotations

import math
from typing import Any


class LRScheduler:
    """Scales each param_group's `base_lr` by a deterministic schedule multiplier."""

    def __init__(
        self,
        optimizer: Any,
        schedule: str = "wsd",
        total_steps: int = 1000,
        warmup_steps: int = 50,
        decay_steps: int = 200,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.schedule = schedule
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.decay_steps = max(0, int(decay_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.last_step = 0

        for group in self.optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])

    def multiplier(self, step: int) -> float:
        """Compute LR multiplier in [min_lr_ratio, 1.0] at 0-indexed step."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)

        if self.schedule == "cosine":
            denom = max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, (step - self.warmup_steps) / denom))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        if self.schedule == "wsd":
            decay_start = max(self.warmup_steps, self.total_steps - self.decay_steps)
            if step < decay_start or self.decay_steps <= 0:
                return 1.0
            progress = min(1.0, max(0.0, (step - decay_start) / max(1, self.decay_steps)))
            # Smooth cosine cooldown in decay phase (Kimi K2 / OLMo 3 style)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        raise ValueError(f"Unknown LR schedule: {self.schedule!r}")

    def step(self, step_num: int | None = None) -> float:
        """Apply LR multiplier for current step and return the multiplier."""
        s = self.last_step if step_num is None else int(step_num)
        mult = self.multiplier(s)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * mult
        self.last_step = s + 1
        return mult

    def state_dict(self) -> dict[str, Any]:
        return {"last_step": self.last_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.last_step = int(state_dict["last_step"])
        # Re-apply exact LR for the restored step
        mult = self.multiplier(max(0, self.last_step - 1)) if self.last_step > 0 else self.multiplier(0)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * mult
