"""Deterministic rule-based verifiers for Reinforcement Learning with Verifiable Rewards (RLVR)."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from typing import Any


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_RE = re.compile(r"(?:answer|result)\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _parse_numeric(val: str) -> float | None:
    s = val.strip().replace(",", "")
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                num, den = float(parts[0]), float(parts[1])
                return num / den if den != 0 else None
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def verify_math_answer(completion: str, expected: str, tol: float = 1e-6) -> float:
    """Return 1.0 if extracted math answer matches expected value, else 0.0."""
    extracted: str | None = None
    boxed = _BOXED_RE.findall(completion)
    if boxed:
        extracted = boxed[-1].strip()
    else:
        ans = _ANSWER_RE.findall(completion)
        if ans:
            extracted = ans[-1].strip()
        else:
            nums = _NUM_RE.findall(completion)
            if nums:
                extracted = nums[-1].strip()

    if extracted is None:
        return 0.0
    if extracted.strip().lower() == expected.strip().lower():
        return 1.0
    num_got = _parse_numeric(extracted)
    num_exp = _parse_numeric(expected)
    if num_got is not None and num_exp is not None:
        return 1.0 if math.isclose(num_got, num_exp, rel_tol=tol, abs_tol=tol) else 0.0
    return 0.0


def verify_format_tags(completion: str) -> float:
    """Return 1.0 if completion properly wraps reasoning in `<|think|>...</|think|>` before final answer."""
    open_tag = "<|think|>"
    close_tag = "</|think|>"
    if open_tag in completion and close_tag in completion:
        i0 = completion.find(open_tag)
        i1 = completion.find(close_tag)
        if i0 < i1 and len(completion[i1 + len(close_tag) :].strip()) > 0:
            return 1.0
    return 0.0


def verify_python_code(completion: str, test_code: str, timeout_s: float = 2.0) -> float:
    """Execute generated Python code + unit assertions in an isolated subprocess with timeout."""
    blocks = _CODE_BLOCK_RE.findall(completion)
    code = blocks[-1] if blocks else completion
    script = f"{code}\n\n{test_code}\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return 1.0 if proc.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0


def verify_constraints(completion: str, constraints: dict[str, Any]) -> float:
    """IFEval-style verifiable instruction constraint checker (returns fraction of satisfied checks)."""
    checks: list[bool] = []
    if "must_include" in constraints:
        for kw in constraints["must_include"]:
            checks.append(kw.lower() in completion.lower())
    if "must_exclude" in constraints:
        for kw in constraints["must_exclude"]:
            checks.append(kw.lower() not in completion.lower())
    if "max_words" in constraints:
        checks.append(len(completion.split()) <= int(constraints["max_words"]))
    if "min_words" in constraints:
        checks.append(len(completion.split()) >= int(constraints["min_words"]))
    if constraints.get("valid_json"):
        try:
            json.loads(completion.strip())
            checks.append(True)
        except Exception:
            checks.append(False)
    if not checks:
        return 1.0
    return sum(1.0 for c in checks if c) / len(checks)


class CompositeVerifier:
    """Combines task accuracy verifier with optional format adherence reward."""

    def __init__(self, format_weight: float = 0.1) -> None:
        self.format_weight = format_weight

    def score(self, completion: str, metadata: dict[str, Any]) -> float:
        task_type = metadata.get("type", "math")
        if task_type == "math":
            acc = verify_math_answer(completion, str(metadata["expected"]))
        elif task_type == "code":
            acc = verify_python_code(completion, str(metadata["test_code"]))
        elif task_type == "constraints":
            acc = verify_constraints(completion, metadata["constraints"])
        else:
            acc = 1.0 if str(metadata.get("expected", "")) in completion else 0.0

        if self.format_weight > 0.0:
            fmt = verify_format_tags(completion)
            return (1.0 - self.format_weight) * acc + self.format_weight * fmt
        return acc
