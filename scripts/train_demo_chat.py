#!/usr/bin/env python3
"""Train a real mini conversational Nova-LLM from scratch on CPU and export for live chat.

Pipeline:
  1. Train a SuperBPE tokenizer on a curated multi-topic knowledge + Q&A corpus.
  2. Stage 1 — Pre-train NovaLM (GQA + QK-Norm + RoPE + SwiGLU) with MuonClip + WSD.
  3. Stage 2 — Assistant-masked ChatML SFT across diverse Q&A pairs + phrasing variants.
  4. Stage 3 — DPO preference alignment (helpful accurate answers vs. poor answers).
  5. Export SafeTensors checkpoint + tokenizer to runs/nova_chat_demo/ and run live Q&A eval.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from nova_llm.config import ModelConfig, OptimConfig, PostTrainConfig, TrainConfig
from nova_llm.data import PackedShardLoader, TokenShardReader, write_token_shard
from nova_llm.inference import SamplingParams, export_safetensors, generate, load_safetensors_model
from nova_llm.models import NovaLM
from nova_llm.post_training import DPOExample, DPOTrainer, SFTTrainer
from nova_llm.tokenizer import ByteBPETokenizer, evaluate_tokenizer
from nova_llm.training import PreTrainer
from nova_llm.utils import seed_everything

# Curated multi-topic Q&A dataset with natural question phrasing variations
QA_PAIRS: list[tuple[list[str], str]] = [
    # Greetings & Identity
    (
        ["Hello", "Hi", "Hey", "Hello!", "Hi there"],
        "Hello! I am Nova-LLM, your AI assistant built from scratch in PyTorch. How can I help you today?",
    ),
    (
        ["Who are you?", "What is your name?", "Tell me about yourself."],
        "I am Nova-LLM, a compact Large Language Model trained from scratch using SuperBPE tokenization, GQA with QK-Norm, MuonClip pre-training, SFT, and DPO.",
    ),
    (
        ["How are you?", "How are you doing?"],
        "I am doing great and ready to help! Ask me about AI, Python, math, science, or general knowledge.",
    ),
    (
        ["Thank you", "Thanks!", "Thanks"],
        "You are welcome! Let me know if you have any other questions.",
    ),
    # AI & Nova-LLM Architecture
    (
        ["What is a transformer?", "Explain transformers."],
        "A transformer is a neural network architecture that uses self-attention to process relationships between all tokens in a sequence in parallel.",
    ),
    (
        ["What is an LLM?", "What is a large language model?"],
        "A Large Language Model (LLM) is a neural network trained to predict the next token in text, enabling it to understand instructions, reason, code, and answer questions.",
    ),
    (
        ["What is MuonClip?", "Explain MuonClip."],
        "MuonClip combines the Muon orthogonalized momentum optimizer for 2D weight matrices with per-head QK-Clip to prevent attention logit explosions during training.",
    ),
    (
        ["What is RoPE?", "What is rotary positional embedding?"],
        "RoPE (Rotary Position Embedding) encodes token positions by rotating query and key vectors in 2D subspaces so attention naturally depends on relative token distance.",
    ),
    (
        ["What is GQA?", "What is Grouped Query Attention?"],
        "Grouped-Query Attention (GQA) shares key and value heads across groups of query heads, drastically shrinking KV-cache memory while preserving attention quality.",
    ),
    (
        ["What is MLA?", "What is Multi-Head Latent Attention?"],
        "Multi-Head Latent Attention (MLA) compresses keys and values into a low-rank latent vector, greatly reducing KV-cache size for long-context inference.",
    ),
    (
        ["What is SuperBPE?", "Explain SuperBPE."],
        "SuperBPE is a two-stage tokenizer curriculum that first learns subwords and then merges frequent multi-word expressions into superwords, cutting token count by ~30%.",
    ),
    (
        ["What is RLVR?", "What is GRPO?"],
        "RLVR (Reinforcement Learning with Verifiable Rewards) uses critic-free algorithms like GRPO, Dr. GRPO, and DAPO to improve reasoning using deterministic math or code checks.",
    ),
    (
        ["What is LoRA?", "Explain LoRA and DoRA."],
        "LoRA fine-tunes large models efficiently by learning low-rank adapter matrices A and B; DoRA further separates weight magnitude and direction for higher accuracy.",
    ),
    (
        ["What is speculative decoding?"],
        "Speculative decoding drafts multiple future tokens cheaply and verifies them in parallel with the target model, speeding up generation with mathematically identical output.",
    ),
    # Math
    (
        ["What is 2 + 2?", "What is 2+2?"],
        "2 + 2 equals 4.",
    ),
    (
        ["What is 12 * 12?", "What is 12 times 12?"],
        "12 * 12 equals 144.",
    ),
    (
        ["What is 15 + 27?"],
        "15 + 27 equals 42.",
    ),
    (
        ["What is the square root of 81?"],
        "The square root of 81 is 9.",
    ),
    (
        ["What is pi?", "What is the value of pi?"],
        "Pi is a mathematical constant approximately equal to 3.14159, representing the ratio of a circle's circumference to its diameter.",
    ),
    # Python & Coding
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
        ["What is Python?"],
        "Python is a high-level, readable programming language widely used for machine learning, data science, web development, and automation.",
    ),
    (
        ["What is Git?"],
        "Git is a distributed version control system that tracks code changes, supports branching, and lets developers collaborate safely.",
    ),
    # Science & Space
    (
        ["What is photosynthesis?"],
        "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce glucose and release oxygen.",
    ),
    (
        ["What is gravity?"],
        "Gravity is the fundamental attractive force between objects with mass; it keeps planets in orbit around stars and anchors us to Earth.",
    ),
    (
        ["What is DNA?"],
        "DNA (deoxyribonucleic acid) is the double-helix molecule that carries genetic instructions for the growth, development, and function of living organisms.",
    ),
    (
        ["What is the speed of light?"],
        "The speed of light in a vacuum is approximately 299,792 kilometers per second (about 300,000 km/s).",
    ),
    (
        ["What is the largest planet in our solar system?", "Which planet is the largest?"],
        "Jupiter is the largest planet in our solar system, a gas giant known for its Great Red Spot.",
    ),
    (
        ["What is the chemical formula for water?", "What is water made of?"],
        "The chemical formula for water is H2O, meaning each molecule contains two hydrogen atoms bonded to one oxygen atom.",
    ),
    # Geography & World Knowledge
    (
        ["What is the capital of India?", "Capital of India?"],
        "The capital of India is New Delhi.",
    ),
    (
        ["What is the capital of France?", "Capital of France?"],
        "The capital of France is Paris.",
    ),
    (
        ["What is the capital of Japan?", "Capital of Japan?"],
        "The capital of Japan is Tokyo.",
    ),
    (
        ["What is the capital of the United States?", "Capital of USA?"],
        "The capital of the United States is Washington, D.C.",
    ),
    (
        ["How many continents are there?"],
        "There are seven continents on Earth: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.",
    ),
    (
        ["What is the largest ocean on Earth?"],
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
    ),
]


def build_datasets() -> tuple[list[str], list[list[dict[str, str]]], list[DPOExample]]:
    pretrain_docs: list[str] = []
    sft_convs: list[list[dict[str, str]]] = []
    dpo_pairs: list[DPOExample] = []

    for questions, answer in QA_PAIRS:
        for q in questions:
            pretrain_docs.append(f"Question: {q}\nAnswer: {answer}\n")
            sft_convs.append(
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": answer},
                ]
            )
        # Build DPO pairs contrasting accurate concise answers vs unhelpful filler
        dpo_pairs.append(
            DPOExample(
                prompt=questions[0],
                chosen=answer,
                rejected="I do not know and cannot help with anything.",
            )
        )

    return pretrain_docs, sft_convs, dpo_pairs


def main() -> None:
    t_start = time.perf_counter()
    seed_everything(42)

    out_dir = Path("runs/nova_chat_demo")
    shard_dir = out_dir / "shards"
    export_dir = out_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrain_docs, sft_convs, dpo_pairs = build_datasets()
    print(f"Dataset ready: {len(sft_convs)} SFT conversations across {len(QA_PAIRS)} knowledge topics.")

    # 1. Train SuperBPE tokenizer
    tok_corpus = pretrain_docs * 3
    tokenizer = ByteBPETokenizer.train(
        tok_corpus,
        vocab_size=960,
        superbpe_transition_vocab=720,
        min_pair_freq=2,
    )
    tok_stats = evaluate_tokenizer(tokenizer, pretrain_docs[:20])
    print(
        f"[1/4] SuperBPE tokenizer trained: vocab={tok_stats['vocab_size']}, "
        f"bytes/tok={tok_stats['bytes_per_token']:.2f}, roundtrip_failures={tok_stats['roundtrip_failures']}"
    )

    # 2. Build binary pre-training shards & run Stage 1 Pre-training (MuonClip + WSD)
    write_token_shard(pretrain_docs * 8, tokenizer, shard_dir / "train")
    reader = TokenShardReader(shard_dir / "train")

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
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
    model = NovaLM(model_cfg)
    print(f"Model initialized: {model.count_parameters()}")

    train_cfg = TrainConfig(
        output_dir=str(out_dir / "pretrain"),
        seed=42,
        max_steps=80,
        micro_batch_size=8,
        grad_accum_steps=1,
        seq_len=64,
        precision="fp32",
        log_interval=20,
        eval_interval=0,
        save_interval=0,
        model=model_cfg,
        optim=OptimConfig(
            optimizer="muonclip",
            lr=3e-3,
            muon_lr=1.5e-2,
            weight_decay=0.01,
            lr_schedule="wsd",
            warmup_steps=10,
            decay_steps=20,
        ),
    )
    loader = PackedShardLoader(reader, batch_size=8, seq_len=64, seed=42)
    pretrainer = PreTrainer(model, loader, train_cfg, device="cpu", quiet=False)
    pt_hist = pretrainer.fit()
    print(f"[2/4] Pre-training complete: loss {pt_hist[0]['loss']:.4f} -> {pt_hist[-1]['loss']:.4f}")

    # 3. Stage 2 — Assistant-masked ChatML SFT
    sft_cfg = PostTrainConfig(
        stage="sft",
        output_dir=str(out_dir / "sft"),
        seed=42,
        max_steps=380,
        micro_batch_size=8,
        grad_accum_steps=1,
        max_seq_len=160,
        lr=3e-3,
        weight_decay=0.0,
        use_lora=False,
    )
    sft_trainer = SFTTrainer(model, tokenizer, sft_convs, sft_cfg, device="cpu")
    sft_hist = sft_trainer.fit()
    print(
        f"[3/4] Assistant-masked SFT complete: loss {sft_hist[0]['sft_loss']:.4f} -> {sft_hist[-1]['sft_loss']:.4f}"
    )

    # 4. Stage 3 — Light DPO preference alignment
    dpo_cfg = PostTrainConfig(
        stage="dpo",
        output_dir=str(out_dir / "dpo"),
        seed=42,
        max_steps=12,
        micro_batch_size=4,
        grad_accum_steps=1,
        max_seq_len=160,
        lr=8e-5,
        dpo_beta=0.1,
        dpo_nll_weight=0.5,
    )
    dpo_trainer = DPOTrainer(model, tokenizer, dpo_pairs, dpo_cfg, device="cpu")
    dpo_hist = dpo_trainer.fit()
    print(
        f"[4/4] DPO alignment complete: loss {dpo_hist[0]['pref_loss']:.4f} -> {dpo_hist[-1]['pref_loss']:.4f} "
        f"| win_acc={dpo_hist[-1]['win_acc']:.2f}"
    )

    # Export SafeTensors + tokenizer
    export_safetensors(model, export_dir, tokenizer=tokenizer)
    reloaded = load_safetensors_model(export_dir, device="cpu").eval()

    elapsed = time.perf_counter() - t_start
    print(f"\nExported trained model to {export_dir} in {elapsed:.1f}s!")

    # Live Q&A verification
    test_prompts = [
        "Who are you?",
        "What is a transformer?",
        "What is MuonClip?",
        "What is the capital of India?",
        "What is 12 * 12?",
        "Write a Python function to reverse a string.",
        "What is photosynthesis?",
        "What is RoPE?",
    ]
    print("\n=== LIVE TRAINED NOVA-LLM Q&A SAMPLE ===")
    stop_ids = [tokenizer.eos_id, tokenizer.special_to_id["<|im_end|>"]]
    qa_samples = []
    for q in test_prompts:
        chat = tokenizer.format_chat([{"role": "user", "content": q}], add_generation_prompt=True)
        ids = torch.tensor([tokenizer.encode(chat, add_bos=True, allow_special=True)], dtype=torch.long)
        res = generate(
            reloaded,
            ids,
            params=SamplingParams(max_new_tokens=96, temperature=0.0, stop_token_ids=stop_ids, stop_strings=["<|im_end|>"]),
            tokenizer=tokenizer,
            use_cache=True,
        )
        ans = (res.text or "").replace("<|im_end|>", "").replace("<|eos|>", "").strip()
        print(f"Q: {q}\nA: {ans}\n")
        qa_samples.append({"question": q, "answer": ans})

    report = {
        "elapsed_s": elapsed,
        "params": model.count_parameters(),
        "vocab_size": tokenizer.vocab_size,
        "pretrain_loss_start": pt_hist[0]["loss"],
        "pretrain_loss_end": pt_hist[-1]["loss"],
        "sft_loss_start": sft_hist[0]["sft_loss"],
        "sft_loss_end": sft_hist[-1]["sft_loss"],
        "dpo_win_acc": dpo_hist[-1]["win_acc"],
        "samples": qa_samples,
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
