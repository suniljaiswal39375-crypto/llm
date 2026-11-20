"""PyTorch-native DDP & FSDP2 (fully_shard + DTensor) distributed training utilities."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_distributed: bool = False
    backend: str = "none"

    @property
    def is_master(self) -> bool:
        return self.rank == 0


def setup_distributed(strategy: str = "none") -> DistributedContext:
    """Initialize torch.distributed when launched via torchrun; otherwise single-process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if strategy == "none" or world_size <= 1:
        return DistributedContext()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_distributed=True,
        backend=backend,
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_distributed(
    model: nn.Module,
    strategy: str,
    ctx: DistributedContext,
    precision: str = "bf16",
) -> nn.Module:
    """Wrap model with DDP or PyTorch-native FSDP2 (fully_shard + MixedPrecisionPolicy)."""
    if not ctx.is_distributed or strategy == "none":
        return model

    if strategy == "ddp":
        device_ids = [ctx.local_rank] if torch.cuda.is_available() else None
        return nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    if strategy == "fsdp2":
        # PyTorch 2.6+ native FSDP2 composable API (replaces deprecated FSDP1)
        try:
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        except ImportError:
            from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        param_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
        )
        # Shard each TransformerBlock individually (TorchTitan pattern)
        if hasattr(model, "layers"):
            for layer in model.layers:
                fully_shard(layer, mp_policy=mp_policy)
        # Keep root (embeddings / final norm / lm_head) with FP32 reduction
        fully_shard(model, mp_policy=mp_policy)
        return model

    raise ValueError(f"Unknown distributed strategy: {strategy!r}")


def no_sync_context(model: nn.Module, enabled: bool) -> ContextManager:
    """Disable DDP gradient all-reduce on intermediate micro-batches during grad accumulation."""
    if enabled and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


def all_reduce_scalar(value: float, ctx: DistributedContext, device: torch.device) -> float:
    """Average a scalar metric across all ranks."""
    if not ctx.is_distributed or not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / ctx.world_size)
