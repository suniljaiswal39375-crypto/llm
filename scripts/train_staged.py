#!/usr/bin/env python3
"""Staged, checkpoint-resumable LLM training runner for Nova-LLM.

Trains the model in 4 clean, independently resumable parts:
  --part 1 : SuperBPE tokenizer + binary shards + MuonClip/WSD Base Pre-Training
  --part 2 : Assistant-masked ChatML Supervised Fine-Tuning (SFT)
  --part 3 : Direct Preference Optimization (DPO) alignment
  --part 4 : RLVR (DAPO) reasoning refinement + SafeTensors export & live Q&A eval
  --part all : Run all 4 parts sequentially
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path

import torch

from nova_llm.config import ModelConfig, OptimConfig, PostTrainConfig, TrainConfig
from nova_llm.data import PackedShardLoader, TokenShardReader, write_token_shard
from nova_llm.eval import evaluate_bpb_and_ppl
from nova_llm.inference import SamplingParams, export_safetensors, generate, load_safetensors_model
from nova_llm.models import NovaLM
from nova_llm.post_training import (
    CompositeVerifier,
    DPOExample,
    DPOTrainer,
    RLVRPrompt,
    RLVRTrainer,
    SFTTrainer,
)
from nova_llm.tokenizer import ByteBPETokenizer, evaluate_tokenizer
from nova_llm.training import PreTrainer
from nova_llm.utils import seed_everything

# Expanded multi-topic conversational, technical, coding, math, science & world knowledge corpus
STAGED_QA_PAIRS: list[tuple[list[str], str]] = [
    # Greetings & Identity
    (
        ["Hello", "Hi", "Hey", "Hello!", "Hi there", "Good morning"],
        "Hello! I am Nova-LLM, your AI assistant built from scratch in PyTorch. How can I help you today?",
    ),
    (
        ["Who are you?", "What is your name?", "Tell me about yourself.", "Introduce yourself."],
        "I am Nova-LLM, a compact Large Language Model trained in stages from scratch using SuperBPE tokenization, GQA with QK-Norm, MuonClip pre-training, SFT, DPO, and RLVR.",
    ),
    (
        ["How are you?", "How are you doing?"],
        "I am doing great and ready to help! You can ask me about AI, LLMs, Python, math, science, geography, or history.",
    ),
    (
        ["Thank you", "Thanks!", "Thanks", "Thank you so much"],
        "You are welcome! Feel free to ask if you have any more questions.",
    ),
    (
        ["Goodbye", "Bye", "See you later"],
        "Goodbye! Have a wonderful day ahead.",
    ),
    # AI, Deep Learning & Frontier LLM Architecture
    (
        ["What is a transformer?", "Explain transformers.", "How do transformers work?"],
        "A transformer is a neural network architecture that uses self-attention to model relationships between all tokens in a sequence in parallel.",
    ),
    (
        ["What is an LLM?", "What is a large language model?"],
        "A Large Language Model (LLM) is a deep neural network trained to predict the next token in text, enabling it to reason, write code, summarize, and answer questions.",
    ),
    (
        ["What is attention in deep learning?", "What is self-attention?"],
        "Self-attention lets each token weigh and gather relevant information from previous tokens by computing scaled dot products between query, key, and value projections.",
    ),
    (
        ["What is MuonClip?", "Explain MuonClip."],
        "MuonClip combines the Muon orthogonalized momentum optimizer for 2D hidden weights with per-head QK-Clip to prevent attention logit explosions at scale.",
    ),
    (
        ["What is RoPE?", "What is rotary positional embedding?"],
        "RoPE (Rotary Position Embedding) rotates query and key vectors by position-dependent angles so attention scores naturally reflect relative token distances.",
    ),
    (
        ["What is YaRN?"],
        "YaRN (Yet another RoPE extensioN) rescales RoPE frequencies across wavelength bands and adjusts attention temperature to extend an LLM's context window.",
    ),
    (
        ["What is GQA?", "What is Grouped Query Attention?"],
        "Grouped-Query Attention (GQA) shares key and value heads across groups of query heads, drastically cutting KV-cache memory while keeping strong quality.",
    ),
    (
        ["What is MLA?", "What is Multi-Head Latent Attention?"],
        "Multi-Head Latent Attention (MLA) compresses keys and values into a shared low-rank latent vector, dramatically shrinking KV-cache memory during long-context inference.",
    ),
    (
        ["What is MoE?", "What is Mixture of Experts?"],
        "Mixture of Experts (MoE) routes each token to only top-k specialized feed-forward experts, increasing total model capacity while keeping active compute small.",
    ),
    (
        ["What is SuperBPE?", "Explain SuperBPE."],
        "SuperBPE is a two-stage tokenizer curriculum that first learns subwords and then merges frequent multi-word expressions into superwords, cutting sequence length by ~30%.",
    ),
    (
        ["What is RLVR?", "What is GRPO?", "What is DAPO?"],
        "RLVR (Reinforcement Learning with Verifiable Rewards) uses critic-free methods like GRPO, Dr. GRPO, and DAPO to sharpen reasoning using deterministic math or code verifiers.",
    ),
    (
        ["What is DPO?", "What is Direct Preference Optimization?"],
        "Direct Preference Optimization (DPO) aligns an LLM directly from chosen vs. rejected response pairs without needing a separate reward model.",
    ),
    (
        ["What is LoRA?", "Explain LoRA and DoRA."],
        "LoRA fine-tunes models efficiently by learning low-rank adapter matrices A and B; DoRA further decouples weight magnitude and direction for higher accuracy.",
    ),
    (
        ["What is speculative decoding?"],
        "Speculative decoding drafts multiple tokens cheaply and verifies them in parallel with the target model, speeding up inference with mathematically identical output.",
    ),
    (
        ["What is quantization in LLMs?"],
        "Quantization stores weights in lower precision such as INT8 or INT4 instead of FP16/FP32, reducing model memory usage and speeding up memory-bound decoding.",
    ),
    (
        ["What is FSDP2?"],
        "FSDP2 (PyTorch Fully Sharded Data Parallel v2) shards model parameters, gradients, and optimizer states across GPUs per parameter using DTensor for scalable training.",
    ),
    (
        ["What is KV cache?", "Why do LLMs use a KV cache?"],
        "A KV cache stores previously computed key and value tensors during autoregressive decoding so each new token does not recompute the entire prompt.",
    ),
    # Math & Logic
    (["What is 2 + 2?", "What is 2+2?"], "2 + 2 equals 4."),
    (["What is 12 * 12?", "What is 12 times 12?"], "12 * 12 equals 144."),
    (["What is 15 + 27?"], "15 + 27 equals 42."),
    (["What is 100 - 37?"], "100 - 37 equals 63."),
    (["What is 9 * 8?"], "9 * 8 equals 72."),
    (["What is 144 / 12?"], "144 divided by 12 equals 12."),
    (["What is the square root of 81?"], "The square root of 81 is 9."),
    (["What is the square root of 64?"], "The square root of 64 is 8."),
    (
        ["What is pi?", "What is the value of pi?"],
        "Pi is a mathematical constant approximately equal to 3.14159, representing the ratio of a circle's circumference to its diameter.",
    ),
    (
        ["What is a prime number?"],
        "A prime number is an integer greater than 1 that has no positive divisors other than 1 and itself, such as 2, 3, 5, 7, and 11.",
    ),
    # Python & Software Engineering
    (
        ["Write a Python function to reverse a string.", "How do I reverse a string in Python?"],
        "Here is a concise Python function:\n```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```",
    ),
    (
        ["Write a Python function to add two numbers.", "How do I add two numbers in Python?"],
        "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```",
    ),
    (
        ["Write a Python function to check if a number is even."],
        "```python\ndef is_even(n: int) -> bool:\n    return n % 2 == 0\n```",
    ),
    (
        ["Write a Python function to compute factorial."],
        "```python\ndef factorial(n: int) -> int:\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n```",
    ),
    (
        ["Write a Python function to check a palindrome."],
        "```python\ndef is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n```",
    ),
    (
        ["What is Python?"],
        "Python is a high-level, readable programming language widely used for machine learning, data science, web backends, and automation.",
    ),
    (
        ["What is PyTorch?"],
        "PyTorch is an open-source deep learning framework offering dynamic computation graphs, GPU tensor acceleration, and distributed training.",
    ),
    (
        ["What is Git?"],
        "Git is a distributed version control system that tracks source code history, supports branching, and enables safe team collaboration.",
    ),
    (
        ["What is an API?"],
        "An API (Application Programming Interface) is a defined set of endpoints and rules that lets software applications communicate with each other.",
    ),
    # Science, Physics, Biology & Space
    (
        ["What is photosynthesis?"],
        "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and release oxygen.",
    ),
    (
        ["What is gravity?"],
        "Gravity is the fundamental attractive force between objects with mass; it keeps planets orbiting stars and anchors us to Earth.",
    ),
    (
        ["What is DNA?"],
        "DNA (deoxyribonucleic acid) is the double-helix molecule carrying genetic instructions for the growth, development, and function of living organisms.",
    ),
    (
        ["What is the speed of light?"],
        "The speed of light in a vacuum is approximately 299,792 kilometers per second (about 300,000 km/s).",
    ),
    (
        ["What is the largest planet in our solar system?", "Which planet is the largest?"],
        "Jupiter is the largest planet in our solar system, a gas giant famous for its Great Red Spot.",
    ),
    (
        ["Which planet is known as the Red Planet?"],
        "Mars is known as the Red Planet because iron oxide on its surface gives it a reddish appearance.",
    ),
    (
        ["What is the chemical formula for water?", "What is water made of?"],
        "The chemical formula for water is H2O, meaning each molecule consists of two hydrogen atoms bonded to one oxygen atom.",
    ),
    (
        ["What gas do humans breathe in to survive?"],
        "Humans breathe in oxygen (O2), which cells use during respiration to produce energy.",
    ),
    (
        ["What is an atom?"],
        "An atom is the basic unit of matter, made up of a nucleus of protons and neutrons surrounded by orbiting electrons.",
    ),
    # Geography & World Knowledge
    (["What is the capital of India?", "Capital of India?"], "The capital of India is New Delhi."),
    (["What is the capital of France?", "Capital of France?"], "The capital of France is Paris."),
    (["What is the capital of Japan?", "Capital of Japan?"], "The capital of Japan is Tokyo."),
    (
        ["What is the capital of the United States?", "Capital of USA?"],
        "The capital of the United States is Washington, D.C.",
    ),
    (
        ["What is the capital of the United Kingdom?", "Capital of UK?"],
        "The capital of the United Kingdom is London.",
    ),
    (
        ["How many continents are there?"],
        "There are seven continents on Earth: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.",
    ),
    (
        ["What is the largest ocean on Earth?"],
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
    ),
    (
        ["What is the tallest mountain in the world?"],
        "Mount Everest in the Himalayas is the tallest mountain above sea level at 8,848.86 meters.",
    ),
    (
        ["Who wrote Romeo and Juliet?"],
        "William Shakespeare wrote the tragedy Romeo and Juliet.",
    ),
    (
        ["Who invented the World Wide Web?"],
        "Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN.",
    ),
]


def _build_datasets() -> tuple[list[str], list[list[dict[str, str]]], list[DPOExample], list[RLVRPrompt]]:
    pretrain_docs: list[str] = []
    sft_convs: list[list[dict[str, str]]] = []
    dpo_pairs: list[DPOExample] = []

    for questions, answer in STAGED_QA_PAIRS:
        for q in questions:
            pretrain_docs.append(f"Question: {q}\nAnswer: {answer}\n")
            sft_convs.append([{"role": "user", "content": q}, {"role": "assistant", "content": answer}])
        dpo_pairs.append(
            DPOExample(
                prompt=questions[0],
                chosen=answer,
                rejected="I cannot answer that question right now.",
            )
        )

    rlvr_prompts = [
        RLVRPrompt(prompt="What is 2 + 2?", metadata={"type": "math", "expected": "4"}),
        RLVRPrompt(prompt="What is 12 * 12?", metadata={"type": "math", "expected": "144"}),
        RLVRPrompt(prompt="What is 15 + 27?", metadata={"type": "math", "expected": "42"}),
        RLVRPrompt(prompt="What is the square root of 81?", metadata={"type": "math", "expected": "9"}),
    ]
    return pretrain_docs, sft_convs, dpo_pairs, rlvr_prompts


def _model_cfg(vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=160,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256,
        attention_type="gqa",
        qk_norm=True,
        z_loss_weight=1e-4,
        mtp_depth=0,
        dropout=0.0,
        pad_vocab_multiple=1,
    )


def _save_stage(path: Path, model: NovaLM, stage: str, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "model_config": asdict(model.cfg),
            "model": model.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def _load_stage(path: Path) -> tuple[NovaLM, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required prior stage checkpoint not found: {path}. Run earlier parts first.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload["model_config"])
    model = NovaLM(cfg)
    model.load_state_dict(payload["model"])
    return model, payload.get("metrics", {})


def run_part1_pretrain(base_dir: Path, steps: int = 100) -> dict:
    t0 = time.perf_counter()
    seed_everything(42)
    pretrain_docs, _, _, _ = _build_datasets()

    # Train SuperBPE tokenizer
    tok_path = base_dir / "tokenizer.json"
    tokenizer = ByteBPETokenizer.train(
        pretrain_docs * 3, vocab_size=1408, superbpe_transition_vocab=1024, min_pair_freq=2
    )
    tokenizer.save(tok_path)
    tok_stats = evaluate_tokenizer(tokenizer, pretrain_docs[:30])

    # Build train & val binary shards
    shard_dir = base_dir / "shards"
    write_token_shard(pretrain_docs * 8, tokenizer, shard_dir / "train")
    write_token_shard(pretrain_docs[:25], tokenizer, shard_dir / "val")

    cfg = _model_cfg(tokenizer.vocab_size)
    model = NovaLM(cfg)
    train_cfg = TrainConfig(
        output_dir=str(base_dir / "part1_logs"),
        seed=42,
        max_steps=steps,
        micro_batch_size=8,
        grad_accum_steps=1,
        seq_len=64,
        precision="fp32",
        log_interval=25,
        eval_interval=50,
        save_interval=0,
        model=cfg,
        optim=OptimConfig(
            optimizer="muonclip",
            lr=3e-3,
            muon_lr=1.5e-2,
            weight_decay=0.01,
            lr_schedule="wsd",
            warmup_steps=12,
            decay_steps=25,
        ),
    )
    train_loader = PackedShardLoader(TokenShardReader(shard_dir / "train"), batch_size=8, seq_len=64, seed=42)
    val_loader = PackedShardLoader(TokenShardReader(shard_dir / "val"), batch_size=4, seq_len=64, seed=99)
    trainer = PreTrainer(model, train_loader, train_cfg, val_loader=val_loader, device="cpu", quiet=False)
    hist = trainer.fit()

    bpb = evaluate_bpb_and_ppl(model, tokenizer, pretrain_docs[:20])
    metrics = {
        "part": 1,
        "stage": "pretrain",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "params": model.count_parameters(),
        "tokenizer": tok_stats,
        "loss_start": hist[0]["loss"],
        "loss_end": hist[-1]["loss"],
        "eval_bpb": bpb["bits_per_byte"],
        "eval_ppl": bpb["perplexity"],
    }
    _save_stage(base_dir / "checkpoints" / "part1_pretrain.pt", model, "part1_pretrain", metrics)
    print(f"[PART 1 DONE] Pre-training: loss {metrics['loss_start']:.4f} -> {metrics['loss_end']:.4f} | BPB={bpb['bits_per_byte']:.3f} ({metrics['elapsed_s']}s)")
    return metrics


def run_part2_sft(base_dir: Path, steps: int = 420) -> dict:
    t0 = time.perf_counter()
    seed_everything(43)
    tokenizer = ByteBPETokenizer.load(base_dir / "tokenizer.json")
    model, _ = _load_stage(base_dir / "checkpoints" / "part1_pretrain.pt")
    _, sft_convs, _, _ = _build_datasets()

    sft_cfg = PostTrainConfig(
        stage="sft",
        output_dir=str(base_dir / "part2_logs"),
        seed=43,
        max_steps=steps,
        micro_batch_size=8,
        grad_accum_steps=1,
        max_seq_len=160,
        lr=3e-3,
        weight_decay=0.0,
        use_lora=False,
    )
    hist = SFTTrainer(model, tokenizer, sft_convs, sft_cfg, device="cpu").fit()
    metrics = {
        "part": 2,
        "stage": "sft",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "conversations": len(sft_convs),
        "sft_loss_start": hist[0]["sft_loss"],
        "sft_loss_end": hist[-1]["sft_loss"],
    }
    _save_stage(base_dir / "checkpoints" / "part2_sft.pt", model, "part2_sft", metrics)
    print(f"[PART 2 DONE] SFT ({len(sft_convs)} convs): loss {metrics['sft_loss_start']:.4f} -> {metrics['sft_loss_end']:.4f} ({metrics['elapsed_s']}s)")
    return metrics


def run_part3_dpo(base_dir: Path, steps: int = 16) -> dict:
    t0 = time.perf_counter()
    seed_everything(44)
    tokenizer = ByteBPETokenizer.load(base_dir / "tokenizer.json")
    model, _ = _load_stage(base_dir / "checkpoints" / "part2_sft.pt")
    _, _, dpo_pairs, _ = _build_datasets()

    dpo_cfg = PostTrainConfig(
        stage="dpo",
        output_dir=str(base_dir / "part3_logs"),
        seed=44,
        max_steps=steps,
        micro_batch_size=4,
        grad_accum_steps=1,
        max_seq_len=160,
        lr=8e-5,
        dpo_beta=0.1,
        dpo_nll_weight=0.5,
    )
    hist = DPOTrainer(model, tokenizer, dpo_pairs, dpo_cfg, device="cpu").fit()
    metrics = {
        "part": 3,
        "stage": "dpo",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "pairs": len(dpo_pairs),
        "pref_loss_start": hist[0]["pref_loss"],
        "pref_loss_end": hist[-1]["pref_loss"],
        "win_acc": hist[-1]["win_acc"],
    }
    _save_stage(base_dir / "checkpoints" / "part3_dpo.pt", model, "part3_dpo", metrics)
    print(f"[PART 3 DONE] DPO: loss {metrics['pref_loss_start']:.4f} -> {metrics['pref_loss_end']:.4f} | win_acc={metrics['win_acc']:.2f} ({metrics['elapsed_s']}s)")
    return metrics


def run_part4_rlvr_and_export(base_dir: Path, steps: int = 4) -> dict:
    t0 = time.perf_counter()
    seed_everything(45)
    tokenizer = ByteBPETokenizer.load(base_dir / "tokenizer.json")
    model, _ = _load_stage(base_dir / "checkpoints" / "part3_dpo.pt")
    _, _, _, rlvr_prompts = _build_datasets()

    rl_cfg = PostTrainConfig(
        stage="dapo",
        output_dir=str(base_dir / "part4_logs"),
        seed=45,
        max_steps=steps,
        lr=2e-5,
        rl_group_size=4,
        rl_max_new_tokens=20,
        rl_kl_beta=0.02,
        rl_dynamic_sampling=True,
        rl_ppo_epochs=1,
    )
    hist = RLVRTrainer(model, tokenizer, rlvr_prompts, rl_cfg, verifier=CompositeVerifier(), device="cpu").fit()

    export_dir = base_dir / "model"
    export_safetensors(model, export_dir, tokenizer=tokenizer)
    reloaded = load_safetensors_model(export_dir, device="cpu").eval()

    # Live Q&A check
    stop_ids = [tokenizer.eos_id, tokenizer.special_to_id["<|im_end|>"]]
    test_qs = [
        "Who are you?",
        "What is MoE?",
        "What is FSDP2?",
        "What is the capital of India?",
        "What is 12 * 12?",
        "Write a Python function to check a palindrome.",
        "What is the tallest mountain in the world?",
        "What is photosynthesis?",
    ]
    samples = []
    print("\n=== STAGED NOVA-LLM (PARTS 1-4) LIVE Q&A ===")
    for q in test_qs:
        prompt = tokenizer.format_chat([{"role": "user", "content": q}], add_generation_prompt=True)
        ids = torch.tensor([tokenizer.encode(prompt, add_bos=True, allow_special=True)], dtype=torch.long)
        res = generate(
            reloaded,
            ids,
            params=SamplingParams(max_new_tokens=96, temperature=0.0, stop_token_ids=stop_ids, stop_strings=["<|im_end|>"]),
            tokenizer=tokenizer,
            use_cache=True,
        )
        ans = (res.text or "").replace("<|im_end|>", "").replace("<|eos|>", "").strip()
        print(f"Q: {q}\nA: {ans}\n")
        samples.append({"question": q, "answer": ans})

    metrics = {
        "part": 4,
        "stage": "rlvr_dapo_export",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "rl_steps": len(hist),
        "mean_reward_last": hist[-1]["mean_reward"] if hist else 0.0,
        "export_dir": str(export_dir),
        "samples": samples,
    }
    _save_stage(base_dir / "checkpoints" / "part4_rlvr.pt", model, "part4_rlvr", metrics)
    print(f"[PART 4 DONE] RLVR (DAPO) + SafeTensors exported to {export_dir} ({metrics['elapsed_s']}s)")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged resumable Nova-LLM trainer (Parts 1-4)")
    parser.add_argument("--part", choices=["1", "2", "3", "4", "all"], required=True)
    parser.add_argument("--out-dir", default="runs/staged_llm")
    args = parser.parse_args()

    base_dir = Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    report_path = base_dir / "staged_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    parts = ["1", "2", "3", "4"] if args.part == "all" else [args.part]
    for p in parts:
        if p == "1":
            report["part1"] = run_part1_pretrain(base_dir)
        elif p == "2":
            report["part2"] = run_part2_sft(base_dir)
        elif p == "3":
            report["part3"] = run_part3_dpo(base_dir)
        elif p == "4":
            report["part4"] = run_part4_rlvr_and_export(base_dir)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
