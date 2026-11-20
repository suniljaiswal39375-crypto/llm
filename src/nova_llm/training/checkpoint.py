"""Atomic, deterministic full-state checkpoint save & resume."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from nova_llm.utils.seed import capture_rng_state, restore_rng_state


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP / torch.compile wrappers to access underlying NovaLM module."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    loader: Any | None = None,
    step: int = 0,
    tokens_seen: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically save full training state (weights, optimizer, scheduler, RNG, loader cursor)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")

    raw_model = unwrap_model(model)
    payload: dict[str, Any] = {
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "model_state": raw_model.state_dict(),
        "model_config": asdict(raw_model.cfg) if hasattr(raw_model, "cfg") else None,
        "rng_state": capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if loader is not None and hasattr(loader, "state_dict"):
        payload["loader_state"] = loader.state_dict()
    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, tmp_dest)
    tmp_dest.replace(dest)
    return dest


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    loader: Any | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load full checkpoint and restore weights, optimizer, scheduler, RNG, and dataloader state."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    raw_model = unwrap_model(model)
    raw_model.load_state_dict(payload["model_state"])

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    if loader is not None and "loader_state" in payload and hasattr(loader, "load_state_dict"):
        loader.load_state_dict(payload["loader_state"])
    if restore_rng and "rng_state" in payload:
        restore_rng_state(payload["rng_state"])

    return {
        "step": int(payload.get("step", 0)),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "model_config": payload.get("model_config"),
        "extra": payload.get("extra", {}),
    }
