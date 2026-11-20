"""Contamination-aware evaluation: BPB, Perplexity, Multiple-Choice, NIAH, and lm-eval harness."""

from nova_llm.eval.harness import EvalReport, generate_eval_report, run_lm_eval_command
from nova_llm.eval.metrics import evaluate_bpb_and_ppl, evaluate_multiple_choice
from nova_llm.eval.niah import evaluate_niah

__all__ = [
    "EvalReport",
    "generate_eval_report",
    "run_lm_eval_command",
    "evaluate_bpb_and_ppl",
    "evaluate_multiple_choice",
    "evaluate_niah",
]
