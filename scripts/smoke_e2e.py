"""Complete end-to-end smoke pipeline (<2 min on CPU):

1. Curate, deduplicate, and decontaminate a synthetic corpus
2. Train a 2-stage SuperBPE byte-level tokenizer & verify lossless round-trip
3. Write memory-mapped binary token shards with document boundaries
4. Pre-train a NovaLM (GQA + QK-Norm + Sandwich Norm + MTP) with Hybrid MuonClip + WSD
5. Fine-tune via Assistant-Masked SFT (with LoRA merge)
6. Align via DPO
7. Run Reasoning RLVR via DAPO (Clip-Higher + Dynamic Sampling + Verifier)
8. Run Autoregressive + Speculative Decoding (MTP & N-gram)
9. Quantize to INT8 & Export to SafeTensors + verify exact reload parity
10. Evaluate BPB, Perplexity, Multiple-Choice, and NIAH & save JSON report
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from nova_llm.config import ModelConfig, OptimConfig, PostTrainConfig, TrainConfig
from nova_llm.data import PackedShardLoader, TokenShardReader, curate_corpus, write_token_shard
from nova_llm.eval import evaluate_bpb_and_ppl, evaluate_multiple_choice, evaluate_niah, generate_eval_report
from nova_llm.inference import (
    SamplingParams,
    export_safetensors,
    generate,
    load_safetensors_model,
    quantize_model_weight_only,
    speculative_generate,
)
from nova_llm.models import NovaLM, merge_and_unload_lora
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


SAMPLE_DOCS = [
    "Transformers process sequences in parallel using scaled dot-product attention and rotary position embeddings.",
    "Grouped-query attention reduces key-value cache memory while retaining strong modeling quality at long context.",
    "Multi-head latent attention compresses keys and values into a shared low-rank latent vector for efficient inference.",
    "The MuonClip optimizer applies Newton-Schulz orthogonalization and per-head QK-Clip to eliminate training spikes.",
    "Warmup-stable-decay schedules allow continuing pre-training from any stable plateau checkpoint before annealing.",
    "Reinforcement learning with verifiable rewards trains reasoning models using deterministic math and code verifiers.",
    "SuperBPE introduces a two-stage pretokenization curriculum to learn both subwords and multi-word superwords.",
    "DeepSeekMoE combines fine-grained routed experts with shared experts and auxiliary-loss-free load balancing.",
    "Contact admin@example.com for duplicate spam test.",
    "Transformers process sequences in parallel using scaled dot-product attention and rotary position embeddings.",
]


def run_smoke_e2e(work_dir: Path | None = None) -> dict:
    ctx_dir = tempfile.TemporaryDirectory() if work_dir is None else None
    root = Path(ctx_dir.name) if ctx_dir is not None else work_dir
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Data curation & decontamination
        expanded = SAMPLE_DOCS * 4
        clean_docs, curate_stats = curate_corpus(
            expanded,
            benchmark_texts=["What is the secret benchmark test sentence that must never appear?"],
            min_chars=15,
            fuzzy_dedup=True,
        )
        print("[1/10] Curated docs:", curate_stats)

        # 2. Train 2-stage SuperBPE tokenizer
        tok = ByteBPETokenizer.train(
            clean_docs * 5,
            vocab_size=384,
            superbpe_transition_vocab=330,
            min_pair_freq=2,
        )
        tok_stats = evaluate_tokenizer(tok, clean_docs)
        assert tok_stats["roundtrip_failures"] == 0, "Tokenizer failed lossless UTF-8 round-trip!"
        tok.save(root / "tokenizer.json")
        print("[2/10] SuperBPE tokenizer:", tok_stats)

        # 3. Build memory-mapped binary token shards
        train_meta = write_token_shard(clean_docs * 20, tok, root / "shards" / "train")
        val_meta = write_token_shard(clean_docs * 4, tok, root / "shards" / "val")
        train_reader = TokenShardReader(root / "shards" / "train")
        val_reader = TokenShardReader(root / "shards" / "val")
        print(f"[3/10] Shards ready: train={train_meta.num_tokens} tok, val={val_meta.num_tokens} tok")

        # 4. Pre-train tiny NovaLM with MuonClip + WSD + MTP
        model_cfg = ModelConfig(
            vocab_size=tok.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=128,
            attention_type="gqa",
            qk_norm=True,
            sandwich_norm=True,
            z_loss_weight=1e-4,
            mtp_depth=1,
            mtp_loss_weight=0.3,
        )
        model = NovaLM(model_cfg)
        train_cfg = TrainConfig(
            output_dir=str(root / "pretrain_run"),
            seed=42,
            max_steps=12,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=32,
            precision="fp32",
            eval_interval=6,
            eval_steps=2,
            log_interval=6,
            save_interval=12,
            model=model_cfg,
            optim=OptimConfig(
                optimizer="muonclip",
                lr=3e-3,
                muon_lr=1e-2,
                qk_clip_tau=100.0,
                lr_schedule="wsd",
                warmup_steps=3,
                decay_steps=4,
            ),
        )
        train_loader = PackedShardLoader(train_reader, batch_size=2, seq_len=32, seed=42)
        val_loader = PackedShardLoader(val_reader, batch_size=2, seq_len=32, seed=123)
        trainer = PreTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=train_cfg,
            bytes_per_token=val_reader.bytes_per_token,
            device="cpu",
            quiet=False,
        )
        pt_hist = trainer.fit()
        print(f"[4/10] Pre-training loss: {pt_hist[0]['loss']:.4f} -> {pt_hist[-1]['loss']:.4f}")

        # 5. Assistant-masked SFT with LoRA + lossless merge
        sft_convs = [
            [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "thinking": "Add 2 and 2.", "content": "The answer is \\boxed{4}."},
            ],
            [
                {"role": "user", "content": "What does GQA save?"},
                {"role": "assistant", "content": "GQA saves key-value cache memory."},
            ],
        ]
        sft_cfg = PostTrainConfig(
            stage="sft", max_steps=4, micro_batch_size=2, grad_accum_steps=1, max_seq_len=96, lr=1e-3, use_lora=True, lora_rank=4
        )
        sft_trainer = SFTTrainer(model, tok, sft_convs, sft_cfg, device="cpu")
        sft_hist = sft_trainer.fit()
        merge_and_unload_lora(model)
        # Re-enable all params (including norms/embeddings) for subsequent full-FT DPO/RLVR smoke test
        for p in model.parameters():
            p.requires_grad = True
        print(f"[5/10] SFT (LoRA merged) loss: {sft_hist[0]['sft_loss']:.4f} -> {sft_hist[-1]['sft_loss']:.4f}")

        # 6. DPO alignment
        dpo_ex = [
            DPOExample(
                prompt="What is 2 + 2?",
                chosen="The answer is 4.",
                rejected="The answer is 999.",
            )
        ]
        dpo_cfg = PostTrainConfig(
            stage="dpo", max_steps=3, micro_batch_size=1, grad_accum_steps=1, max_seq_len=96, lr=5e-4, dpo_beta=0.1
        )
        dpo_trainer = DPOTrainer(model, tok, dpo_ex, dpo_cfg, device="cpu")
        dpo_hist = dpo_trainer.fit()
        print(f"[6/10] DPO loss: {dpo_hist[-1]['pref_loss']:.4f} | win_acc={dpo_hist[-1]['win_acc']:.2f}")

        # 7. RLVR (DAPO)
        rl_prompts = [RLVRPrompt(prompt="Compute 2 + 2.", metadata={"type": "math", "expected": "4"})]
        rl_cfg = PostTrainConfig(
            stage="dapo",
            max_steps=2,
            lr=1e-4,
            rl_group_size=2,
            rl_max_new_tokens=12,
            rl_temperature=0.8,
            rl_dynamic_sampling=False,
            rl_ppo_epochs=1,
        )
        rl_trainer = RLVRTrainer(model, tok, rl_prompts, rl_cfg, verifier=CompositeVerifier(), device="cpu")
        rl_hist = rl_trainer.fit()
        print(f"[7/10] RLVR (DAPO) step metrics: {rl_hist[-1]}")

        # 8. Autoregressive + Speculative Decoding (MTP & N-gram)
        prompt_ids = torch.tensor([tok.encode("Transformers process", add_bos=True)], dtype=torch.long)
        greedy_params = SamplingParams(max_new_tokens=10, temperature=0.0)
        ar_res = generate(model, prompt_ids, params=greedy_params, tokenizer=tok, use_cache=True)
        mtp_res = speculative_generate(
            model, prompt_ids, params=greedy_params, method="mtp", num_speculative_tokens=1, tokenizer=tok
        )
        assert torch.equal(ar_res.new_token_ids, mtp_res.new_token_ids), "Greedy MTP speculative mismatch!"
        print(
            f"[8/10] Speculative (MTP) exact match verified! accept_rate={mtp_res.acceptance_rate:.2f}, "
            f"out={ar_res.text!r}"
        )

        # 9. Export SafeTensors + reload parity + INT8 quantization check
        export_dir = export_safetensors(model, root / "hf_export", tokenizer=tok)
        reloaded = load_safetensors_model(export_dir, device="cpu")
        with torch.inference_mode():
            diff = (model(prompt_ids).logits - reloaded(prompt_ids).logits).abs().max().item()
        assert diff == 0.0, f"SafeTensors reload logit mismatch: {diff}"
        q_stats = quantize_model_weight_only(reloaded, bits=8, group_size=32)
        print(f"[9/10] SafeTensors export parity={diff:.1e} | INT8 compression={q_stats['compression_ratio']:.2f}x")

        # 10. Evaluation report (BPB, PPL, MC, NIAH)
        bpb_metrics = evaluate_bpb_and_ppl(model, tok, clean_docs[:3])
        mc_metrics = evaluate_multiple_choice(
            model,
            tok,
            [{"prompt": "Transformers use ", "choices": ["attention", "bananas"], "answer_idx": 0}],
        )
        niah_metrics = evaluate_niah(model, tok, context_lengths=[48], depths=[0.5])
        report = generate_eval_report(
            model,
            run_id="smoke_e2e",
            metrics={"lm": bpb_metrics, "mc": mc_metrics, "niah": niah_metrics},
            output_path=root / "eval_report.json",
        )
        print(
            f"[10/10] Eval complete: BPB={bpb_metrics['bits_per_byte']:.3f}, "
            f"PPL={bpb_metrics['perplexity']:.2f}, params={report.model_params}"
        )
        return {
            "pretrain_loss_start": pt_hist[0]["loss"],
            "pretrain_loss_end": pt_hist[-1]["loss"],
            "bpb": bpb_metrics["bits_per_byte"],
            "safetensors_max_diff": diff,
            "quant_compression": q_stats["compression_ratio"],
        }
    finally:
        if ctx_dir is not None:
            ctx_dir.cleanup()


if __name__ == "__main__":
    summary = run_smoke_e2e()
    print("\nSMOKE E2E SUMMARY:", json.dumps(summary, indent=2))
