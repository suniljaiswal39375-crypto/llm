# Production Training & Real-LLM Deployment Guide

A small ~1M-parameter model trained on 2 CPUs in minutes cannot generalize like GPT, Gemini, or Claude—general open-world reasoning requires **hundreds of millions to billions of parameters** trained on **tens of billions to trillions of tokens on GPUs**.

This repository gives you **two real paths** to run a genuinely intelligent, thinking, web-searching LLM using `nova_llm`:

---

## Path 1 (Fastest — Single GPU / Laptop): Import a Real Pretrained Model into `NovaLM`

`NovaLM` natively imports real Hugging Face **Llama-3**, **Qwen2.5**, **Qwen3**, **SmolLM2**, and **Mistral** SafeTensors checkpoints with exact mathematical parity (`src/nova_llm/models/hf_import.py`).

### Step 1: Download & Import Any Open-Weight Frontier Model
```bash
# Example: download Qwen3-1.7B or Llama-3.2-3B-Instruct locally via huggingface-cli
huggingface-cli download Qwen/Qwen3-1.7B --local-dir weights/qwen3_1.7b

# Convert directly into native NovaLM SafeTensors format
python scripts/import_hf.py --hf-dir weights/qwen3_1.7b --out-dir runs/nova_qwen3_1.7b
```

### Step 2: Specialize It with LoRA/DoRA SFT, DPO, and Web-Search RLVR on Real Datasets
```bash
# 1. Assistant-masked DoRA SFT on real JSONL chat/search traces (OpenAI, ShareGPT, or Alpaca schema)
python scripts/sft.py \
  --model-dir runs/nova_qwen3_1.7b \
  --data datasets/smoltalk_and_search.jsonl \
  --out-dir runs/nova_qwen3_sft \
  --dora --lora-rank 32 --steps 2000 --lr 1e-4

# 2. Preference alignment (DPO / SimPO / IPO) on UltraFeedback / groundedness pairs
python scripts/dpo.py \
  --model-dir runs/nova_qwen3_sft \
  --data datasets/ultrafeedback.jsonl \
  --out-dir runs/nova_qwen3_dpo \
  --method dpo --steps 500 --lr 5e-6

# 3. Reasoning & Web-Search RLVR (DAPO) with deterministic verifiers
python scripts/rlvr.py \
  --model-dir runs/nova_qwen3_dpo \
  --data datasets/math_and_search_rlvr.jsonl \
  --out-dir runs/nova_qwen3_final \
  --algo dapo --verifier web_search --steps 500
```

### Step 3: Serve With Live Internet Web Search
```bash
export TAVILY_API_KEY="tvly-..."   # or SERPER_API_KEY / BRAVE_API_KEY
python scripts/serve.py --model-dir runs/nova_qwen3_final --host 0.0.0.0 --port 8000
```

---

## Path 2 (From Scratch — Multi-GPU Cluster): Pre-Train Your Own 125M → 1B+ `NovaLM`

### Step 1: Curate Real Pre-Training Data (FineWeb-Edu / DCLM / Code) & Pack Binary Shards
```bash
python scripts/prepare_data.py \
  --input data_raw/fineweb_edu/ \
  --out-dir data_shards/fineweb_edu \
  --vocab-size 65536 --superbpe-transition 49152
```

### Step 2: Distributed Pre-Training with PyTorch FSDP2 + MuonClip + WSD + MTP
```bash
torchrun --standalone --nproc_per_node=8 scripts/pretrain.py \
  --config configs/train/gpu_1b_pretrain.yaml \
  --train-shard data_shards/fineweb_edu/train \
  --val-shard data_shards/fineweb_edu/val
```

### Step 3: Post-Train (SFT → DPO → DAPO RLVR) & Serve
Run `scripts/sft.py`, `scripts/dpo.py`, `scripts/rlvr.py`, and `scripts/serve.py` exactly as in Path 1!
