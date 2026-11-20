"""Post-training pipelines: Assistant-masked SFT, DPO/SimPO/IPO, and RLVR (GRPO/Dr.GRPO/DAPO/GSPO)."""

from nova_llm.post_training.dpo import DPOExample, DPOTrainer, compute_preference_loss
from nova_llm.post_training.rlvr import RLVRPrompt, RLVRTrainer, compute_rlvr_loss
from nova_llm.post_training.sft import SFTTrainer, collate_sft_examples
from nova_llm.post_training.verifiers import (
    CompositeVerifier,
    verify_constraints,
    verify_format_tags,
    verify_math_answer,
    verify_python_code,
)

__all__ = [
    "DPOExample",
    "DPOTrainer",
    "compute_preference_loss",
    "RLVRPrompt",
    "RLVRTrainer",
    "compute_rlvr_loss",
    "SFTTrainer",
    "collate_sft_examples",
    "CompositeVerifier",
    "verify_constraints",
    "verify_format_tags",
    "verify_math_answer",
    "verify_python_code",
]
