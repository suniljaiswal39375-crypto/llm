# Nova-LLM

A frontier 2026 modular Large Language Model (LLM) engineering suite built in PyTorch 2.x—covering the complete lifecycle from **SuperBPE tokenization & data curation** through **GQA/MLA/Gated-DeltaNet-2/MoE pre-training with MuonClip & FSDP2**, **post-training (SFT, LoRA/DoRA, DPO/SimPO/IPO, RLVR: GRPO/Dr.GRPO/DAPO/GSPO)**, **speculative decoding & quantization**, **SafeTensors export**, **OpenAI-compatible serving**, and **contamination-aware evaluation**.

---

## Architecture & Research Foundations (2025–2026 SOTA)

| Layer | Implemented Innovations | Reference Blueprint |
|---|---|---|
| **Tokenization** | Byte-Level BPE + **SuperBPE** 2-stage subword→superword curriculum; ChatML + `<\|think\|>` / tool-call / FIM registry | SuperBPE (COLM 2025), Qwen3 |
| **Data Pipeline** | PII scrubbing, heuristics, exact SHA-256 + **MinHash/LSH** fuzzy dedup, **n-gram benchmark decontamination**, memory-mapped binary shards with **document-boundary isolation & cross-doc loss masking**, 3-stage `MultiSourceMixer` | FineWeb-Edu, DCLM, Nemotron-CC, OLMo 3 |
| **Attention** | **GQA + QK-Norm**, **Multi-Head Latent Attention (MLA)**, **Sliding-Window + Global** interleaving (`5:1` / `3:1`), **Gated DeltaNet-2** 3:1 hybrid linear attention, FP32 **RoPE + YaRN** + optional **NoPE** layers | DeepSeek-V3/V4, Qwen3, Gemma 3/4, Kimi Linear, Gated DeltaNet-2 |
| **Normalization & MoE** | FP32 **RMSNorm**, **Gemma Sandwich Norm** (Pre+Post), **Single-Scale RMSNorm + Embed Proj (OSP)**, Fine-grained **Sparse MoE + Shared Experts** with **Auxiliary-Loss-Free** dynamic router bias & FP32 router | DeepSeek-V3, Gemma 3/4, OSP (ACL 2025) |
| **Objectives** | FP32 Cross-Entropy, **Z-Loss** (`1e-4`), optional Logit Soft-Capping, **Multi-Token Prediction (MTP)** auxiliary heads (boosts pre-training AND serves as built-in speculative draft heads!) | DeepSeek-V3, OLMo 2/3, Gemma 3 |
| **Optimization & Systems** | Hybrid **Muon / MuonClip** (5-step Newton-Schulz + Adam RMS matching + per-head **QK-Clip**) + AdamW (strictly zero weight decay on norms/biases/embeddings), **WSD** & Cosine schedules, exact token-weighted grad accumulation, loss-spike/NaN guard, PyTorch-native **DDP & FSDP2 (`fully_shard`)**, deterministic full-state checkpoint resume | Moonlight, Kimi K2, TorchTitan |
| **Post-Training** | Assistant-only masked **SFT**, **LoRA & DoRA** (lossless merge/unmerge), **DPO / SimPO / IPO** (with pair-completeness guards), Critic-free **RLVR**: **GRPO**, **Dr. GRPO**, **DAPO** (Clip-Higher, Dynamic Sampling, token-level loss, overlong penalty), **GSPO** (sequence-level IS ratio), plus deterministic math/code/format/constraint verifiers | DeepSeek-R1, DAPO, Dr. GRPO, Qwen GSPO |
| **Inference & Serving** | Pre-allocated **Static KV-Cache / MLA Latent Cache / O(1) Linear State**, full sampling (greedy, temp, top-k, top-p, min-p, rep/presence penalty), lossless **Speculative Decoding** (N-gram, Draft Model, MTP), group-wise **INT8/INT4** weight quantization + **QAT**, **SafeTensors** export, OpenAI-compatible **FastAPI** SSE server | vLLM / SGLang interop, EAGLE/MTP, Kimi K2.6 QAT |
| **Evaluation** | Tokenizer-independent **Bits-Per-Byte (BPB)**, perplexity, multiple-choice log-likelihood, **Needle-in-a-Haystack (NIAH)**, structured JSON reports + `lm-evaluation-harness` bridge | lm-eval, RULER/NIAH |

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. Run full mathematical & silent-bug verification suite (46 tests)
pytest -v

# 2. Run the complete 10-stage end-to-end smoke pipeline (<15 seconds on CPU)
python scripts/smoke_e2e.py
```

### Pre-Training CLI
```bash
# Single GPU / CPU
python scripts/pretrain.py \
  --config configs/train/pretrain_muonclip.yaml \
  --train-shard data_shards/train \
  --val-shard data_shards/val

# Multi-GPU with PyTorch FSDP2
torchrun --standalone --nproc_per_node=8 scripts/pretrain.py \
  --config configs/train/pretrain_muonclip.yaml \
  --train-shard data_shards/train \
  --override distributed=fsdp2 \
  --override precision=bf16
```

### Inference & OpenAI-Compatible Serving
```bash
python scripts/serve.py --model-dir runs/exported_hf --host 0.0.0.0 --port 8000
```

---

## Project Layout

```text
├── configs/
│   ├── models/          # tiny, small_125m, dense_1b, moe_deepseek, hybrid_linear
│   └── train/           # pretrain_muonclip.yaml
├── src/nova_llm/
│   ├── config.py        # Typed dataclasses + YAML loader + CLI dot-overrides
│   ├── tokenizer/       # Byte-Level BPE & 2-stage SuperBPE + ChatML/Think registry
│   ├── data/            # Curation, MinHash dedup, decontamination, binary shards, packed loader
│   ├── models/          # NovaLM, GQA, MLA, Gated DeltaNet-2, DeepSeekMoE, MTP, RoPE/YaRN, LoRA/DoRA
│   ├── optim/           # Muon, MuonClip (QK-Clip), HybridOptimizer, WSD/Cosine schedulers
│   ├── training/        # PreTrainer, DDP/FSDP2, atomic deterministic checkpoints
│   ├── post_training/   # SFT, DPO/SimPO/IPO, RLVR (GRPO/Dr.GRPO/DAPO/GSPO), verifiers
│   ├── inference/       # Static KV/MLA/Linear caches, sampler, speculative decoding, INT8/INT4/QAT, SafeTensors, FastAPI
│   ├── eval/            # BPB/PPL, Multiple-Choice, NIAH, JSON eval reports & lm-eval bridge
│   └── utils/           # Deterministic rank/worker seeding, device/autocast, JSONL telemetry
├── scripts/             # smoke_e2e.py, pretrain.py, generate.py, serve.py
├── tests/               # 46 mathematical, silent-bug, and integration proofs
└── docs/ARCHITECTURE.md # Deep technical design & silent-bug prevention ledger
```
