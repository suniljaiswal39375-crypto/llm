"""Structured JSONL + console telemetry logger for training and evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TelemetryLogger:
    """Writes structured metrics to JSONL and concise formatted lines to stdout."""

    def __init__(self, log_path: str | Path, rank: int = 0, quiet: bool = False) -> None:
        self.log_path = Path(log_path)
        self.rank = rank
        self.quiet = quiet
        self.start_time = time.perf_counter()
        if self.rank == 0:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.rank != 0:
            return
        record: dict[str, Any] = {"elapsed_s": round(time.perf_counter() - self.start_time, 3)}
        if step is not None:
            record["step"] = int(step)
        for k, v in metrics.items():
            if isinstance(v, float):
                record[k] = round(v, 6)
            else:
                record[k] = v
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if not self.quiet:
            parts = []
            if step is not None:
                parts.append(f"step={step:5d}")
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print(" | ".join(parts), flush=True)
