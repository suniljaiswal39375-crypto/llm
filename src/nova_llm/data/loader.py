"""Document-aware packed sequence dataloader with cross-doc loss masking & deterministic resume."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from nova_llm.data.shards import TokenShardReader


class PackedShardLoader:
    """Streams packed sequences with document IDs, cross-doc target masking, and exact cursor state.

    Guarantees:
    - Each distributed rank receives disjoint windows via stride `(world_size * batch_size)`.
    - Document transitions inside a packed window mask the transition target to `-100` so the model
      never predicts the first token of an unrelated next document!
    - `state_dict()` / `load_state_dict()` restores the exact next batch on checkpoint resume.
    """

    def __init__(
        self,
        shard: TokenShardReader,
        batch_size: int,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        shuffle_windows: bool = True,
    ) -> None:
        self.shard = shard
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.shuffle_windows = shuffle_windows

        # Window needs seq_len + 1 tokens to create (inputs=[:seq_len], targets=[1:seq_len+1])
        self.window_span = self.seq_len + 1
        self.num_windows = (len(shard) - 1) // self.seq_len
        if self.num_windows < 1:
            raise ValueError(
                f"Shard too small ({len(shard)} tokens) for seq_len={self.seq_len}"
            )
        self.epoch = 0
        self.cursor = 0
        self._order = self._build_order(self.epoch)

    def _build_order(self, epoch: int) -> np.ndarray:
        idx = np.arange(self.num_windows, dtype=np.int64)
        if self.shuffle_windows:
            rng = np.random.default_rng(self.seed + epoch * 9973)
            rng.shuffle(idx)
        return idx

    def next_batch(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Fetch next packed batch: {'input_ids', 'targets', 'doc_ids'}."""
        inputs_list = []
        targets_list = []
        docs_list = []

        for _ in range(self.batch_size):
            # Advance global window index strided by world_size for rank disjointness
            global_pos = self.cursor * self.world_size + self.rank
            if global_pos >= self.num_windows:
                self.epoch += 1
                self.cursor = 0
                self._order = self._build_order(self.epoch)
                global_pos = self.rank % self.num_windows

            win_id = int(self._order[global_pos])
            self.cursor += 1

            start = win_id * self.seq_len
            end = start + self.window_span
            tok_slice = np.asarray(self.shard.tokens[start:end], dtype=np.int64)
            doc_slice = np.asarray(self.shard.doc_ids[start:end], dtype=np.int64)

            x = tok_slice[:-1]
            y = tok_slice[1:].copy()
            d_in = doc_slice[:-1]
            d_tgt = doc_slice[1:]

            # Mask cross-document boundary target transitions to ignore_index=-100
            y[d_in != d_tgt] = -100

            inputs_list.append(x)
            targets_list.append(y)
            docs_list.append(d_in)

        return {
            "input_ids": torch.tensor(np.stack(inputs_list), dtype=torch.long, device=device),
            "targets": torch.tensor(np.stack(targets_list), dtype=torch.long, device=device),
            "doc_ids": torch.tensor(np.stack(docs_list), dtype=torch.long, device=device),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "cursor": self.cursor, "seed": self.seed}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self._order = self._build_order(self.epoch)


class MultiSourceMixer:
    """Weighted multi-source sampler for 3-stage curriculum (Pretrain -> Mid-train -> Long-Ctx)."""

    def __init__(
        self,
        loaders: dict[str, PackedShardLoader],
        weights: dict[str, float],
        seed: int = 42,
    ) -> None:
        self.names = list(loaders.keys())
        self.loaders = loaders
        raw_w = np.asarray([weights[n] for n in self.names], dtype=np.float64)
        if (raw_w < 0).any() or raw_w.sum() <= 0:
            raise ValueError("Mixer weights must be non-negative and sum > 0")
        self.probs = raw_w / raw_w.sum()
        self.rng = np.random.default_rng(seed)

    def next_batch(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        choice = int(self.rng.choice(len(self.names), p=self.probs))
        return self.loaders[self.names[choice]].next_batch(device=device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "loaders": {name: ldr.state_dict() for name, ldr in self.loaders.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng_state"]
        for name, st in state["loaders"].items():
            if name in self.loaders:
                self.loaders[name].load_state_dict(st)
