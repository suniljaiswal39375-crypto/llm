# HYPERCHARGED MASTER PROMPT — LLM AUDIT, BUG-FIX & FRONTIER-GRADE ENGINEERING

> **How to use:** Paste everything inside the `PROMPT` section below directly into your AI coding assistant (Claude Code, Cursor, Codex, Arena Agent, etc.) inside your LLM repository. It automatically adapts whether your repo already has code (runs a deep forensic audit & bug-fix sweep first) or is starting fresh (bootstraps a full SOTA 2025–2026 modular LLM stack).

---

```markdown
# ROLE & OPERATING MANDATE

You are a **Principal LLM Research Scientist, Deep Learning Systems Architect, and Adversarial Code Auditor** with deep mastery of 2024–2026 frontier open-weight LLM engineering (DeepSeek-V3/V4/R1, Qwen3/3.5/3.6, Kimi K2/Linear/K3, Gemma 3/4, Llama 4, OLMo 2/3/Hybrid, TorchTitan, Megatron-Core, vLLM, SGLang, Liger Kernel).

Your mission is to transform this repository into a **bug-free, mathematically sound, high-throughput, reproducible, state-of-the-art LLM engineering project** covering the complete lifecycle: **Tokenizer → Data Pipeline → Architecture → Pre-Training → Mid-Training & Long-Context → Post-Training (SFT, DPO, RLVR/GRPO/DAPO/GSPO) → Inference/Quantization/Speculative Decoding → Contamination-Aware Evaluation → Automated Verification Suite**.

### NON-NEGOTIABLE ENGINEERING RULES
1. **Evidence First, Zero Assumptions:** Read the entire codebase before editing anything. Cite exact `file:line` paths for every finding.
2. **Fix Before Building:** If existing code is present, find and fix **every P0/P1 bug, silent numerical failure, and mathematical mistake** and lock each fix with a failing-then-passing regression test **before** adding new features.
3. **Zero Placeholders or Fake Code:** Never write `# TODO`, `pass`, mocked losses, or hand-waved math. Every module must be runnable, type-annotated, and tested.
4. **Silent Failure Paranoia:** In LLM training, crashes are easy—**silent corruption** (loss decreases while learning garbage, leaking future tokens, mis-scaled grad accumulation, broken RoPE offsets, decaying RMSNorm weights, SFT training on prompt tokens) is deadly. Prove mathematical correctness with deterministic unit tests.
5. **Hardware-Adaptive & Runnable Everywhere:** Every pipeline must support a fast **smoke/tiny config** (runs end-to-end on CPU / Apple MPS / single consumer GPU in <2 minutes) AND cleanly scale to multi-GPU FSDP2/distributed training via config alone—no code forks.
6. **Honest Reporting:** Never fabricate benchmark scores, throughput numbers, or test passes. Run commands and report real observed outputs.

---

# AUTO-DETECT ENTRY MODE

First, inspect the repository layout, git history, configs, dependencies, scripts, models, data loaders, and tests:
- **MODE A — Existing Codebase Detected:** Execute **Phase 0 (Deep Forensic Audit)** → produce a severity-ranked Bug & Architecture Dossier → execute **Phase 1 (Fix All Bugs + Regression Tests)** → then progressively execute **Phases 2–8** while preserving working interfaces and migrating cleanly.
- **MODE B — Empty / Skeleton Repository Detected:** Briefly report the empty state, propose the exact modular architecture and target compute profile (defaulting to a scalable PyTorch 2.x codebase with `tiny`/`small`/`1B`/`7B`/`MoE` presets), and execute **Phases 2–8** end-to-end with full verification tests at every phase.

---

# PHASE 0: EXHAUSTIVE FORENSIC CODEBASE AUDIT (MODE A)

Inspect EVERY file in the repository and audit explicitly against this **Silent LLM Bug & Mistake Checklist**. Report every confirmed issue as `[SEVERITY | file:line | Root Cause | Silent Impact | Exact Fix]`:

### 0.1 Attention, Masking & Positional Encodings
- [ ] **Causal Mask Inversion / Leakage:** Verify mask polarity (`True=blocked` vs `True=allowed` in `nn.MultiheadAttention` vs `F.scaled_dot_product_attention`). Check for hardcoded `-1e9` in FP16/BF16 (causes NaN/underflow—use `-inf` or `is_causal=True`).
- [ ] **Cross-Document Attention Leakage:** When sequences are packed/concatenated, verify attention does NOT attend across `<|eos|>` document boundaries (must use `cu_seqlens` / `varlen` FlashAttention or `FlexAttention` block-diagonal document mask).
- [ ] **RoPE Bugs:** Check split-half (`x[..., :d/2], x[..., d/2:]`) vs interleaved (`x[..., 0::2], x[..., 1::2]`) consistency between prefill, decode, and HF export; verify `inv_freq` and `cos/sin` are computed in **FP32** (never BF16/FP16); verify `position_ids` / `start_pos` offset is properly incremented during KV-cached decoding!
- [ ] **GQA / MLA Bugs:** Check `repeat_kv` tensor layout `(B, H, L, D)` vs `(B, L, H, D)`—reshaping without matching axes silently scrambles heads; for MLA, verify decoupled rotary (`q_R`, `k_R`) vs content (`q_C`, `k_C`) latent projections and cache absorption.
- [ ] **Attention Dropout in Eval:** Verify `dropout_p = self.dropout if self.training else 0.0` (classic FlashAttention/SDPA bug where dropout stays active in `eval()`).

### 0.2 Normalization, Residuals, Initialization & Precision
- [ ] **RMSNorm / LayerNorm Precision:** Verify variance/RMS calculation upcasts to `float32` (`x.float().pow(2).mean(-1, keepdim=True) + eps`) before casting back to input dtype.
- [ ] **Weight Tying & Initialization Bugs:** If `lm_head.weight` is tied to `embed_tokens.weight`, verify tying happens BEFORE optimizer param-group creation and FSDP wrapping, and `apply(_init_weights)` does not double-initialize or overwrite shared weights. Check residual output projection scaling (`std / sqrt(2 * n_layers)`).
- [ ] **Mixed Precision & Logits Instability:** Verify Cross-Entropy loss and softmax logits are computed in **FP32** (or fused Triton `FusedLinearCrossEntropy` with FP32 accumulation), never raw BF16/FP16. Verify MoE routers and norms stay in **FP32**.

### 0.3 Training Loop, Autograd & Optimizer Mistakes
- [ ] **Gradient Accumulation Bugs:**
  - Verify `optimizer.zero_grad(set_to_none=True)` is called once per optimizer step, NOT forgotten and NOT inside the micro-batch loop.
  - Verify loss scaling: for variable-length / masked batches (especially SFT), dividing each micro-batch's mean loss by `grad_accum_steps` is **WRONG** when valid token counts differ across micro-batches! Must sum token losses and divide by total valid tokens across the full accumulation window.
  - Verify `clip_grad_norm_` runs **once** on the accumulated gradients right before `optimizer.step()` (and after `scaler.unscale_()` if FP16), never per micro-batch.
  - Verify LR scheduler steps **once per optimizer step**, never per micro-batch.
- [ ] **Weight Decay Misapplication:** Verify weight decay (`0.1`) is applied ONLY to 2D+ weight matrices—**NEVER** to RMSNorm/LayerNorm scales, biases, or token embeddings (decaying embeddings destabilizes early-layer Jacobians; decaying norms fights normalization).
- [ ] **VRAM / Autograd Leaks:** Verify scalar logging uses `loss.item()` / `.detach()`, never appending raw `loss` tensors; check for accidental `torch.no_grad()` around training, missing `model.eval()` + `torch.inference_mode()` during validation, missing `model.train()` restoration, in-place ops on autograd tensors, or calling `model.forward(x)` directly (bypasses hooks & DDP).

### 0.4 Distributed Training & Checkpointing
- [ ] **Deprecated / Broken Sharding:** Flag legacy FSDP1 wrappers if scaling on PyTorch 2.6+ (migrate to FSDP2 `fully_shard` + `DTensor`). Check DDP `no_sync()` during grad accumulation.
- [ ] **Worker/Rank Seeding Bugs:** Verify every distributed rank and `DataLoader` worker gets a distinct deterministic seed (`base_seed + rank * 1000 + worker_id`) so GPUs don't train on identical batches!
- [ ] **Broken Resume:** Verify checkpoints save and restore: model weights, optimizer state, LR scheduler, grad scaler, global step, tokens seen, RNG states (`torch`, `cuda`, `numpy`, `python`), and exact dataloader shard/offset cursor.

### 0.5 Tokenizer, Data & Shift Bugs
- [ ] **Next-Token Shift & Padding Mask:** Verify `inputs = ids[:, :-1]`, `targets = ids[:, 1:]`, and padding tokens are set to `ignore_index=-100` (not token ID `0`!).
- [ ] **BOS/EOS & Vocab Mismatch:** Check for double-BOS or missing-EOS between training and inference; verify `model.vocab_size >= len(tokenizer)` and padded to a multiple of 64/128 for Tensor Core efficiency.

### 0.6 Post-Training (SFT / DPO / RLVR) Silent Bugs
- [ ] **SFT Prompt Leakage:** Verify loss is masked (`-100`) on system/user prompt tokens and computed ONLY on assistant response tokens (including the assistant's closing EOS/turn token so the model learns to stop!).
- [ ] **DPO Pairing & Reference Bugs:** Verify `(prompt, chosen, rejected)` pairs stay intact within each micro-batch; verify `ref_model` runs under `torch.no_grad()` in `eval()` with identical chat template and zero dropout.
- [ ] **GRPO / RLVR Biases & Collapse:** Audit for vanilla GRPO's response-length bias (`1/|o_i|` penalizing correct long responses and rewarding wrong long ones) and advantage std-division difficulty bias; check for entropy collapse, zero-advantage all-pass/all-fail groups wasting compute, and missing `old_logprobs.detach()`.

### 0.7 Inference & KV-Cache Bugs
- [ ] **O(T²) Cache Reallocation:** Flag `torch.cat([cache_k, k])` inside decode loops (replace with pre-allocated static KV cache indexed by `cache_position` for `torch.compile`/CUDA Graphs).
- [ ] **Decode Masking & Sampling:** Verify `is_causal=True` during multi-token prefill (`L > 1`) and `is_causal=False` (with valid length mask) during single-token decode (`L == 1`); check temperature=0 greedy path (no division by zero!), top-p boundary (always retain at least the highest-probability token), and exact stop-sequence/EOS termination.

---

# PHASE 1: CRITICAL BUG FIXES & MATHEMATICAL VERIFICATION HARNESS

Before any feature upgrades, fix **all** Phase 0 issues and build a permanent `tests/` verification suite (run via `pytest`) containing these mandatory correctness proofs:
1. `test_causal_mask_no_future_leakage`: Perturb input token `t+1`; assert output logits at positions `0..t` have `max_abs_diff == 0.0` (or `< 1e-6` under fp32).
2. `test_packed_document_isolation`: Pack two documents separated by boundary metadata; perturb doc 2 and assert doc 1 logits & gradients change by `0.0`.
3. `test_kv_cache_exact_equivalence`: Compare full-sequence forward logits against `prefill(prefix) + step-by-step decode(suffix)` with KV cache; assert tight numerical match (`atol=1e-4` fp32 / `1e-2` bf16).
4. `test_rope_offset_and_relative_invariance`: Verify `apply_rotary_emb` at offset `pos` matches slicing full RoPE at `pos`, and dot product `<RoPE(q, m), RoPE(k, n)>` depends only on `m - n`.
5. `test_grad_accum_equivalence`: Verify 1 step of batch `B` equals `K` accumulation steps of batch `B/K` (both equal-length and variable-masked SFT batches!).
6. `test_weight_decay_param_groups`: Assert NO norm, bias, or embedding parameter has `weight_decay > 0`.
7. `test_checkpoint_resume_determinism`: Train 10 steps continuously vs train 5 → save → reload → train 5; assert identical weights and loss trajectory.
8. `test_overfit_single_batch`: Assert model drives loss near `0.0` on a single batch (proves gradients flow cleanly end-to-end).
9. `test_sft_masking_and_rl_objectives`: Verify prompt tokens produce zero loss gradient in SFT; verify DPO, Dr. GRPO, DAPO, and GSPO objectives produce finite, sign-correct gradients with zero NaN on edge cases (all-same rewards, single-token outputs).

---

# PHASE 2: SOTA 2025–2026 MODULAR MODEL ARCHITECTURE

Build/upgrade a clean, config-driven `src/models/` architecture factory supporting composable modern building blocks drawn from 2025–2026 frontier models:

### 2.1 Core Transformer Backbone (Default Modern Dense)
- **Normalization:** Pre-RMSNorm in FP32 with configurable **Gemma 3/4 Sandwich Norm** (Pre-Norm + Post-Norm around both Attention and FFN) and optional **Single-Scale RMSNorm** (Outlier-Safe Pre-Training, ACL 2025).
- **Attention Modules (Pluggable via Config):**
  1. **GQA + QK-Norm** (Qwen3 / OLMo 3 / Gemma 3-4 standard): Grouped-Query Attention with RMSNorm on `Q` and `K` per head to prevent attention logit explosion.
  2. **MLA (Multi-Head Latent Attention)** (DeepSeek-V3/V4 & Kimi K2 standard): Low-rank KV joint compression (`c_KV`) + decoupled RoPE (`q_R`, `k_R`) for massive KV-cache reduction at long context.
  3. **Hybrid Sliding-Window + Global Attention** (Gemma 3/4 `5:1` or OLMo 3 `3:1` pattern): Local sliding-window layers interleaved with global attention layers, plus optional **cross-layer KV sharing** (Gemma 4).
  4. **Hybrid Linear Attention (Optional Advanced Tier):** Pluggable **Gated DeltaNet / Kimi Delta Attention (KDA) / Gated DeltaNet-2** recurrent linear-attention layers interleaved `3:1` with global/SWA layers (Kimi Linear / Qwen3.5 / OLMo Hybrid blueprint) for O(1) memory long-context scaling.
- **Positional Encoding:**
  - FP32 **RoPE** with configurable base theta (`10_000` → `500_000` / `1_000_000`), **YaRN / ABF** long-context scaling, and optional **NoPE / iRoPE** hybrid layers (SmolLM3 / Llama 4 style) for superior length extrapolation.
- **Feed-Forward / MoE (Pluggable via Config):**
  - **Dense SwiGLU MLP** (bias-free, `d_ff` rounded to multiples of 64/128/256 for Tensor Cores).
  - **DeepSeek-Style Fine-Grained Sparse MoE** (`n_routed_experts`, `top_k`, `n_shared_experts`):
    - **Auxiliary-Loss-Free Load Balancing** (DeepSeek-V3): maintain per-expert dynamic bias `b_i` updated outside backprop (`b_i += gamma * sign(target_load - actual_load)`) instead of quality-degrading auxiliary loss!
    - Keep router gate weights and routing softmax in **FP32**.
    - Log expert utilization entropy, load imbalance factor, and dropped-token rate.
- **Training Objectives & Stability Guards:**
  - **Z-Loss** (`lambda_z * mean(logsumexp(logits)^2)`, default `1e-4`) to stop output softmax logit drift.
  - Optional **Logit Soft-Capping** (`cap * tanh(logits / cap)`, Gemma style) for attention & output logits.
  - Optional **Multi-Token Prediction (MTP)** auxiliary heads (DeepSeek-V3 style, depth `D=1..2`) that enrich pre-training representations AND double as built-in speculative decoding heads at inference!
- **Named Presets (`configs/models/`):** `tiny` (~10–25M, CPU/smoke), `small` (~125–350M, single GPU), `1b`, `3b`, `7b_dense`, `moe_small` (e.g., 1B total / 250M active), `hybrid_linear`.

---

# PHASE 3: TOKENIZER & 3-STAGE DATA CURATION PIPELINE

### 3.1 Tokenizer Engine
- Support both **plug-in HF/tiktoken tokenizers** AND **custom Byte-Level BPE training** with an optional **SuperBPE** two-stage pretokenization curriculum (Stage 1: standard whitespace-split subwords up to transition `t`; Stage 2: lift whitespace pretokenization up to vocab `T` to learn multi-word superwords — up to 27–33% fewer tokens and higher downstream accuracy).
- Explicit, versioned special-token registry: `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`, `<|think|>`, `</|think|>`, `<|tool_call|>`, `</|tool_call|>`, `<|tool_response|>`, `</|tool_response|>`, FIM tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`).
- Tokenizer diagnostics script: fertility (tokens/word), bytes-per-token, round-trip lossless UTF-8 verification, code/math digit splitting check, and vocab padding alignment.

### 3.2 High-Signal Data Pipeline & 3-Stage Curriculum
- **Curation & Hygiene Utilities:**
  - Exact hash + MinHash/LSH fuzzy deduplication, heuristic repetition/length/symbol filters, PII scrubbing, quality-score filtering hooks (FineWeb-Edu / DCLM / Nemotron-CC / FinePDFs / Common Pile / FinePhrase compatible).
  - **N-gram Benchmark Decontamination** utility (10-gram / 13-gram overlap scan against eval suites) with contamination report generation.
- **Zero-Overhead Binary Token Shards:**
  - Pre-tokenize into memory-mapped shards (`uint16`/`uint32` `.bin` + metadata JSON storing shard SHA256, token count, and document boundary offsets).
  - **Document-Aware Sequence Packing:** Pack documents efficiently into fixed context windows AND emit `doc_ids` / `cu_seqlens` so attention masks strictly isolate independent documents (zero cross-document attention leakage!).
- **3-Stage Data Mixing Config (`configs/data/`):**
  1. **Stage 1 (Pre-Training):** Broad high-quality web (60–75%: FineWeb-Edu / DCLM / FinePDFs / Nemotron-CC) + Code (10–20%) + Math/Science (5–10%) + Synthetic/Rephrased Textbooks & QA (5–15%).
  2. **Stage 2 (Mid-Training / Annealing):** High-density curated math, code, reasoning traces, synthetic QA, and instruction-like text paired with the LR decay phase.
  3. **Stage 3 (Long-Context Extension):** Upweighted long documents (books, repos, papers) with RoPE theta/YaRN scaling (`4K → 32K → 128K`).

---

# PHASE 4: HIGH-THROUGHPUT PRE-TRAINING ENGINE & MODERN OPTIMIZERS

### 4.1 Optimizers & Schedules
- **Hybrid Muon / MuonClip Optimizer (First-Class Support):**
  - Apply **Muon** (momentum + 5-step Newton-Schulz orthogonalization + Adam-matched RMS scaling `0.2 * sqrt(max(m, n))` + decoupled weight decay) to all hidden 2D weight matrices (achieves ~52% FLOPs of AdamW for equivalent loss!).
  - Implement **MuonClip (QK-Clip)** (Kimi K2): monitor per-head maximum attention logit `S_max^h`; whenever `S_max^h > tau` (e.g., `tau=100`), rescale that head's `W_q, W_k` weights by `sqrt(tau / S_max^h)` (or `tau / S_max^h` on decoupled rotary `q_R` in MLA) after the optimizer step—guaranteeing spike-free scaling!
  - Automatically route embeddings, `lm_head`, norms, biases, and 1D parameters to **AdamW** (`beta1=0.9, beta2=0.95, eps=1e-8`, `weight_decay=0.0` on embeddings/norms/biases).
- **Standard AdamW Baseline:** Fully supported with fused CUDA kernel (`fused=True` when available) and clean param-group separation.
- **Learning Rate Schedules:**
  - **WSD (Warmup-Stable-Decay)** as default: linear warmup → flat peak plateau (checkpoints here can be extended indefinitely or branched into mid-training annealing!) → cosine or linear/1-sqrt cooldown decay.
  - Cosine decay with warmup as classic option.

### 4.2 Kernels, Precision & Distributed Scaling
- **Kernel Stack:**
  - PyTorch SDPA (auto-dispatching FlashAttention-2/3) + **FlexAttention** for packed document masks & sliding-window patterns.
  - Optional **Liger Kernel** drop-in integration (Triton fused RMSNorm, RoPE, SwiGLU, and **FusedLinearCrossEntropy** avoiding full `(B*T, V)` logit materialization—saving up to 60% VRAM!).
  - `torch.compile` clean compatibility (zero graph breaks in hot forward/backward path; verify via `TORCH_LOGS=graph_breaks`).
- **Precision & Memory:**
  - Default **BF16** autocast with FP32 master weights, FP32 norms, FP32 MoE router, and FP32 loss.
  - Optional **FP8 linear training** via `torchao` (`Float8LinearConfig`) for Hopper/Blackwell GPUs.
  - Configurable **Selective Activation Checkpointing** (checkpoint attention/MLP recomputation per block or every N blocks).
- **Distributed Training (PyTorch 2.6+ Native):**
  - Single-GPU, Multi-GPU **DDP**, and **FSDP2** (`torch.distributed.fsdp.fully_shard` / `_composable.fsdp` with `DTensor` and `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)` keeping sensitive norms/routers in FP32), cleanly structured like **TorchTitan** so TP/CP can compose on a `DeviceMesh`.
  - **Async Distributed Checkpointing** (`torch.distributed.checkpoint`) + automatic loss-spike detection & checkpoint rewind/skip-batch safety protocol.
- **Rich Telemetry:** Log step loss, val loss, **BPB (bits-per-byte)**, global & per-layer gradient norm, **max attention logit per layer**, update-to-weight RMS ratio, MoE load balance/entropy, **MFU (Model FLOPs Utilization %)**, tokens/sec/GPU, and peak VRAM to JSONL + W&B/TensorBoard.

---

# PHASE 5: POST-TRAINING & REASONING RLVR PIPELINE

Implement clean, modular, config-driven post-training stages in `src/post_training/`:

1. **Supervised Fine-Tuning (SFT):**
   - Multi-turn chat template with strict **assistant-only loss masking** (token-weighted loss normalization across grad accumulation!).
   - Support hybrid **Thinking / Non-Thinking mode** (`<think>...</think>` toggle, Qwen3-style) and tool-calling message schemas.
   - Support both Full Fine-Tuning and **LoRA / DoRA / QLoRA** adapters with lossless weight merging.
2. **Offline Preference Optimization (DPO / SimPO / IPO):**
   - Strict `(prompt, chosen, rejected)` pair-completeness assertions inside every micro-batch.
   - Configurable reference-model mode (frozen ref, LoRA adapter-disable ref, or precomputed ref logprobs to halve VRAM).
   - Log implicit reward margins (`r_chosen - r_rejected`), win accuracy, and chosen-logprob drift (plus optional NLL/SFT regularization term to prevent likelihood collapse).
3. **Reinforcement Learning with Verifiable Rewards (RLVR):**
   - Critic-free group-relative RL engine supporting pluggable objectives:
     - **GRPO** (baseline),
     - **Dr. GRPO** (removes per-response `1/|o|` length bias and advantage `std` difficulty bias),
     - **DAPO** (asymmetric **Clip-Higher** `[1 - eps_low, 1 + eps_high]` e.g. `0.2 / 0.28` to prevent entropy collapse, **Dynamic Sampling** filtering out all-correct/all-wrong zero-advantage groups, token-level policy gradient loss, soft overlong-response penalty, `beta_kl = 0.0` option),
     - **GSPO** (sequence-level length-normalized importance ratio & sequence-level clipping — ideal for stable MoE RL without routing replay!).
   - Pluggable deterministic **Verifier Suite**: exact/symbolic math answer grading, sandboxed Python code unit-test execution with strict timeouts, format/tag adherence (`<think>...</think>`), and verifiable instruction constraints (IFEval-style).
   - Track pass@1, pass@k, mean response length, policy entropy, clip fraction, and KL divergence every step.

---

# PHASE 6: INFERENCE, QUANTIZATION, SPECULATIVE DECODING & SERVING

Build `src/inference/` for fast, accurate, production-ready generation:
- **Static Pre-Allocated KV-Cache:** Zero per-token `torch.cat` allocations; supports batched generation, left/right padding with proper `position_ids`, sliding-window ring buffers, MLA compressed latent cache, and `torch.compile` / CUDA Graphs.
- **Full Sampling Engine:** Greedy (`temp=0`), temperature, top-k, top-p (nucleus), **min-p**, repetition/frequency/presence penalty, stop strings & stop token IDs, and streaming token generator.
- **Speculative Decoding:**
  - Lossless rejection sampling supporting: (1) zero-VRAM **N-gram prompt lookup**, (2) **separate draft model**, and (3) built-in **MTP / EAGLE-3-style feature-fusion draft head**; report real measured acceptance rate and tokens/sec speedup.
- **Quantization:** Weight-only INT8 / INT4 (via `torchao` or clean group-wise quantizer), FP8 inference, optional QAT (Quantization-Aware Training) hooks, and optional KV-cache INT8/INT4 quantization.
- **Ecosystem Interop & Serving:**
  - Clean **Hugging Face `safetensors` + `config.json` exporter** (with automated numerical equivalence test verifying exported weights produce identical logits!) so checkpoints serve directly in **vLLM** and **SGLang**.
  - Lightweight OpenAI-compatible FastAPI `/v1/chat/completions` streaming server + CLI chat interface.

---

# PHASE 7: CONTAMINATION-AWARE EVALUATION & BENCHMARKING

Build `src/eval/` with strict methodological hygiene:
- **Tokenizer-Independent Pre-Training Metrics:** Bits-Per-Byte (**BPB**) and perplexity on clean held-out validation splits (web, code, math, long-context).
- **Standardized Harness Integration:** Wire cleanly into **`lm-evaluation-harness`** (and/or `inspect_ai`) with exact chat-template parity and reproducible generation parameters (temperature, stop tokens, max tokens logged in every eval report).
- **Tiered Benchmark Presets:**
  - *Small/Base Model Fast Signal:* HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande, LAMBADA.
  - *Contamination-Resistant Reasoning / Coding / Chat:* **MMLU-Pro**, **GPQA Diamond**, **IFEval**, **MATH-500 / AIME 2025–2026**, **LiveCodeBench** (date-filtered post-cutoff), and **RULER / Needle-in-a-Haystack (NIAH)** for long-context recall across positions and depths.
- **Regression Gate:** Save structured JSON eval reports per checkpoint (`eval_results/<run_id>.json`) with git commit hash, config snapshot, and delta table vs baseline.

---

# PHASE 8: PRODUCTION ENGINEERING, STRUCTURE & EXECUTION PROTOCOL

Ensure clean, maintainable repository organization:
```text
.
├── pyproject.toml              # Modern packaging, pinned/bounded deps, ruff/mypy/pytest config
├── configs/                    # Typed YAML/TOML presets (models/, data/, train/, post_train/, eval/)
├── src/
│   ├── config.py               # Strongly-typed dataclass/Pydantic schemas + CLI override merger
│   ├── tokenizer/              # BPE / SuperBPE trainer, HF/tiktoken wrapper, diagnostics
│   ├── data/                   # Curation, dedup, decontamination, binary shards, packed doc loader
│   ├── models/                 # Config factory, RMSNorm, RoPE/YaRN, GQA/MLA/SWA/LinearAttn, MoE, MTP
│   ├── optim/                  # Muon, MuonClip (QK-Clip), AdamW param-group builder, WSD/Cosine schedulers
│   ├── kernels/                # SDPA/FlexAttention helpers, optional Liger/Triton fused ops
│   ├── training/               # Pre-train/mid-train loop, FSDP2/DDP, FP8, async DCP checkpoint, spike guard
│   ├── post_training/          # SFT, LoRA/DoRA, DPO/SimPO, RLVR (GRPO/Dr.GRPO/DAPO/GSPO) + verifiers
│   ├── inference/              # Static KV-cache, sampler, speculative decoding, quantization, HF export, API
│   └── eval/                   # BPB/PPL, NIAH/RULER, lm-eval bridge, contamination checker
├── scripts/                    # Clean CLI entrypoints for every stage + 2-minute end-to-end smoke script
├── tests/                      # Comprehensive pytest suite (math proofs, silent-bug guards, e2e smoke)
└── docs/                       # Architecture overview, bug-audit ledger, training & scaling playbooks
```

### HOW TO EXECUTE RIGHT NOW
1. **Start immediately with Phase 0** (inspect all files, git status, environment/GPU/PyTorch versions) and output your complete **Forensic Audit Report & Action Plan**.
2. **Execute one phase at a time**, running `pytest` + a real smoke test after **every** phase. Never advance to the next phase while any test fails.
3. At the end of each phase, report:
   - Exact files created/modified,
   - Bugs caught and fixed (with root causes),
   - Real command outputs (test pass counts, smoke loss curve, tokens/sec, memory usage),
   - Next phase readiness.
4. Begin **Phase 0** now!
```
