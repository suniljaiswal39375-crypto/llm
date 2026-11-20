"""Deterministic seeding across Python, NumPy, PyTorch CPU/CUDA, ranks, and workers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, rank: int = 0) -> int:
    """Seed Python, NumPy, and PyTorch deterministically with rank offset."""
    effective_seed = int(seed) + int(rank) * 1000
    random.seed(effective_seed)
    np.random.seed(effective_seed % (2**32 - 1))
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed


def worker_seed(base_seed: int, rank: int = 0, worker_id: int = 0) -> int:
    """Compute a unique deterministic seed per (rank, DataLoader worker)."""
    return int(base_seed) + int(rank) * 1000 + int(worker_id)


def capture_rng_state() -> dict[str, Any]:
    """Capture full RNG state for exact bit-for-bit checkpoint resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore full RNG state captured by capture_rng_state()."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        cpu_state = state["torch_cpu"]
        if not isinstance(cpu_state, torch.ByteTensor):
            cpu_state = torch.as_tensor(cpu_state, dtype=torch.uint8)
        torch.set_rng_state(cpu_state)
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = [
            s if isinstance(s, torch.ByteTensor) else torch.as_tensor(s, dtype=torch.uint8)
            for s in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
