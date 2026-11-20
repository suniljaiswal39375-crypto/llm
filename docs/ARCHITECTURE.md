# Nova-LLM Architecture & Silent-Bug Prevention Ledger

This document summarizes the engineering invariants enforced throughout `src/nova_llm/` and verified continuously by `tests/`.

## 1. Pluggable Model Topologies (`src/nova_llm/models/`)

`NovaLM` is driven by a single validated `ModelConfig` (`src/nova_llm/config.py`):

1. **Modern Dense (`attention_type="gqa"`):**
   - Grouped-Query Attention with per-head **QK-Norm**, FP32 **RoPE** (with optional **YaRN** context scaling and periodic **NoPE** layers), optional **Sliding-Window / Global** interleaving (`swa_ratio`), and bias-free **SwiGLU** MLP.
   - Supports Pre-RMSNorm, Gemma 3/4 **Sandwich Norm** (`sandwich_norm=True`), and ACL 2025 **Outlier-Safe Pre-Training** (`norm_type="single_scale_rmsnorm"`, `use_osp_embed_proj=True`).
2. **DeepSeek-V3/V4 Style Sparse MoE + MLA (`attention_type="mla"`, `use_moe=True`):**
   - **Multi-Head Latent Attention (MLA)** compresses keys/values into a low-rank normalized latent `c_kv` plus a shared decoupled rotary key `k_rope`. Only `(c_kv, k_rope)` are stored in `LayerMLACache`.
   - **DeepSeekMoE** combines `n_shared_experts` with top-k `n_routed_experts`, runs router affinities strictly in **FP32**, and updates a non-gradient `expert_bias` buffer after each step (**Auxiliary-Loss-Free Load Balancing**).
3. **Hybrid Linear Attention (`attention_type="hybrid_linear"`):**
   - Interleaves **Gated DeltaNet-2** recurrent linear-attention blocks (decoupled decay $\alpha_t$, erase $\beta_t$, and write $\omega_t$ gates with causal depthwise short convolution and L2-normalized Q/K) with global GQA layers at configurable ratio (default `3:1`).
   - Maintains O(1)-memory recurrent matrix state + short-conv history in `LayerLinearAttnState` with exact parity to full-sequence forward passes.
4. **Multi-Token Prediction (`mtp_depth >= 1`):**
   - DeepSeek-V3 MTP modules fuse previous hidden state with shifted token embeddings through lightweight decoder blocks sharing `lm_head`, boosting pre-training signal AND serving as zero-extra-model speculative decoding draft heads (`speculative_generate(..., method="mtp")`).

## 2. Silent-Bug Prevention Matrix (All Covered by `pytest`)

| Silent Failure Class | Root Cause in Naive Implementations | Nova-LLM Invariant | Automated Proof |
|---|---|---|---|
| Future token leakage | Inverted causal mask or unmasked SDPA | Exact `k_pos <= start_pos + q_pos` boolean mask | `test_causal_mask_no_future_leakage` |
| Packed cross-doc leakage | Concatenating docs without boundary masks | `doc_ids` masks both attention, conv taps, recurrent state, AND transition targets (`-100`) | `test_packed_document_isolation`, `test_binary_shards_*` |
| KV-cache / RoPE drift | Missing `start_pos` offset or `torch.cat` bugs | Pre-allocated static `InferenceCache` + FP32 RoPE offset indexing | `test_kv_cache_exact_equivalence`, `test_rope_offset_and_relative_invariance` |
| Grad accumulation bias | Dividing per-microbatch mean by `K` with uneven `-100` masks | Exact token-weighted `loss_reduction="sum" / total_valid_tokens` | `test_grad_accum_exact_equivalence_with_variable_masks` |
| Norm/Embedding decay | Applying AdamW `weight_decay` globally | Explicit param classifier guarantees `wd=0.0` on all norms, biases, 1D params, embeddings | `test_weight_decay_param_groups_*` |
| Attention logit explosion | Unbounded `q^T k` under Muon at scale | QK-Norm + Kimi K2 **MuonClip** per-head QK-Clip (`apply_qk_clip_`) | `test_muonclip_qk_clip_rescales_exploding_heads` |
| Broken checkpoint resume | Missing RNG or dataloader cursor state | Atomic save/load of weights, optimizer, scheduler, RNG, and shard cursor | `test_checkpoint_resume_determinism` |
| SFT prompt supervision | Computing CE on user/system prompt tokens | `encode_chat_for_sft` sets `-100` strictly on non-assistant turns | `test_sft_prompt_tokens_receive_zero_loss_contribution` |
| RLVR length/std bias | Vanilla GRPO `1/\|o\|` & std normalization | Pluggable `dr_grpo`, `dapo` (Clip-Higher + Dynamic Sampling), `gspo` (sequence IS) | `test_rlvr_algorithms_and_trainer` |
| Speculative drift | Off-by-one verify logits or broken greedy match | Exact verification + Leviathan rejection sampling (`ngram`, `draft_model`, `mtp`) | `test_speculative_decoding_exact_greedy_parity` |
