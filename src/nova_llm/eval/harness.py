"""Structured evaluation report generator & lm-evaluation-harness integration bridge."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nova_llm.models.transformer import NovaLM


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


@dataclass
class EvalReport:
    run_id: str
    timestamp_utc: str
    git_commit: str
    model_params: dict[str, int]
    model_config: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)


def generate_eval_report(
    model: NovaLM,
    run_id: str,
    metrics: dict[str, Any],
    output_path: str | Path | None = None,
) -> EvalReport:
    """Create and optionally persist a structured JSON evaluation report."""
    report = EvalReport(
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit_hash(),
        model_params=model.count_parameters(),
        model_config=asdict(model.cfg),
        metrics=metrics,
    )
    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
    return report


def run_lm_eval_command(
    hf_model_dir: str | Path,
    tasks: list[str] | None = None,
    output_dir: str | Path = "eval_results/lm_eval",
) -> list[str]:
    """Return the exact reproducible `lm_eval` CLI command for exported HF SafeTensors models.

    Default contamination-resistant 2026 task bundle:
    - Early base signal: hellaswag, arc_easy, piqa
    - Frontier reasoning / instruction: mmlu_pro, gpqa_diamond, ifeval
    """
    task_list = tasks or ["hellaswag", "arc_easy", "mmlu_pro", "ifeval"]
    return [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={hf_model_dir},trust_remote_code=True",
        "--tasks",
        ",".join(task_list),
        "--output_path",
        str(output_dir),
    ]
