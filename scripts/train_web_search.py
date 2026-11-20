#!/usr/bin/env python3
"""4-Part Staged Trainer for Nova-Search: Web-Search Native Agentic LLM.

Trains NovaLM end-to-end to master all 4 pillars of frontier web search:
  1. Smart Search Routing: Emit `<|tool_call|>web_search` for factual/research queries,
     answer greetings & simple arithmetic directly without wasteful searches.
  2. High-Precision Search Query Planning: Generate concise, keyword-dense queries.
  3. Evidence-Grounded Synthesis & Inline Citations (`[1]`, `[2]`): Extract exact facts
     from `<|tool_result|>` snippets with zero fabricated citations.
  4. Honest Abstention: Explicitly state when retrieved search snippets lack the answer.

Stages (independently resumable):
  --part 1 : SuperBPE tokenizer + Web & Search Trace Pre-Training (MuonClip + WSD)
  --part 2 : Multi-Turn Agentic Search + Citation SFT (assistant-only masked)
  --part 3 : Groundedness & Tool-Use DPO Alignment (cited vs. hallucinated/broken)
  --part 4 : Web-Search RLVR (DAPO + WebSearchRLVRVerifier) + SafeTensors Export & Eval
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path
from typing import Any

import torch

from nova_llm.agent import (
    BM25LocalWebIndex,
    HybridSearchBackend,
    SEARCH_SYSTEM_PROMPT,
    SearchAgent,
    WebSearchRLVRVerifier,
    format_search_results,
    parse_tool_calls,
    verify_grounded_synthesis,
    verify_search_tool_call,
)
from nova_llm.config import ModelConfig, OptimConfig, PostTrainConfig, TrainConfig
from nova_llm.data import PackedShardLoader, TokenShardReader, write_token_shard
from nova_llm.eval import evaluate_bpb_and_ppl
from nova_llm.inference import export_safetensors, load_safetensors_model
from nova_llm.models import NovaLM
from nova_llm.post_training import DPOExample, DPOTrainer, RLVRPrompt, RLVRTrainer, SFTTrainer
from nova_llm.tokenizer import ByteBPETokenizer, evaluate_tokenizer
from nova_llm.training import PreTrainer
from nova_llm.utils import seed_everything

# Each entry: (question_variants, optimal_search_query, expected_keywords, grounded_answer, key_facts)
SEARCH_SCENARIOS: list[dict[str, Any]] = [
    {
        "questions": ["What is DeepSeek-V3?", "Explain DeepSeek-V3 architecture."],
        "query": "DeepSeek-V3 MLA MoE MTP architecture",
        "keywords": ["deepseek", "mla"],
        "answer": "DeepSeek-V3 is a 671B total / 37B active MoE model featuring Multi-Head Latent Attention (MLA), auxiliary-loss-free load balancing via dynamic router biases, and Multi-Token Prediction (MTP) heads [1].",
        "facts": ["671b", "mla", "mtp"],
    },
    {
        "questions": ["What is MuonClip?", "How does Kimi K2 use MuonClip?"],
        "query": "Kimi K2 MuonClip optimizer QK-Clip",
        "keywords": ["muonclip", "qk-clip"],
        "answer": "MuonClip is Kimi K2's optimizer combining Newton-Schulz orthogonalized momentum with per-head QK-Clip to eliminate attention logit explosions across 15.5T tokens [1].",
        "facts": ["newton-schulz", "qk-clip", "15.5t"],
    },
    {
        "questions": ["What is SuperBPE?", "How much does SuperBPE reduce tokens?"],
        "query": "SuperBPE tokenizer superwords token reduction",
        "keywords": ["superbpe", "tokenizer"],
        "answer": "SuperBPE uses a two-stage tokenizer curriculum transitioning from subwords to cross-whitespace superwords, cutting sequence token counts by 27-33% while improving downstream accuracy [1].",
        "facts": ["two-stage", "27-33%"],
    },
    {
        "questions": ["What is Gated DeltaNet-2?", "How does hybrid linear attention save KV cache?"],
        "query": "Gated DeltaNet hybrid linear attention KV cache",
        "keywords": ["deltanet", "linear"],
        "answer": "A 3:1 hybrid of Gated DeltaNet linear attention and global attention reduces KV-cache memory by roughly 75% while matching or outperforming pure softmax transformers on long context [1].",
        "facts": ["75%", "linear attention"],
    },
    {
        "questions": ["What is DAPO in RLVR?", "How does DAPO improve over GRPO?"],
        "query": "DAPO Dr GRPO RLVR reasoning Clip-Higher",
        "keywords": ["dapo", "grpo"],
        "answer": "DAPO improves critic-free RLVR over GRPO with Clip-Higher asymmetric clipping, Dynamic Sampling of non-zero-advantage groups, token-level loss, and overlong reward shaping, while Dr. GRPO removes length and std bias [1].",
        "facts": ["clip-higher", "dynamic sampling"],
    },
    {
        "questions": ["What is PyTorch FSDP2?", "How does FSDP2 differ from FSDP1?"],
        "query": "PyTorch FSDP2 fully_shard DTensor",
        "keywords": ["fsdp2", "dtensor"],
        "answer": "PyTorch FSDP2 uses DTensor-based per-parameter sharding via fully_shard(), replacing legacy FlatParameter FSDP1 with composable tensor parallelism and native torch.compile support [1].",
        "facts": ["dtensor", "fully_shard"],
    },
    {
        "questions": ["What is YaRN RoPE scaling?", "How does YaRN extend context length?"],
        "query": "YaRN RoPE context window extension mscale",
        "keywords": ["yarn", "rope"],
        "answer": "YaRN extends RoPE context windows by interpolating low-frequency dimensions, preserving high-frequency local bands, and scaling attention temperature by mscale = 0.1*ln(s)+1 [1].",
        "facts": ["interpolating", "mscale"],
    },
    {
        "questions": ["What is speculative decoding?", "How does EAGLE-3 or MTP speed up inference?"],
        "query": "speculative decoding EAGLE MTP rejection sampling",
        "keywords": ["speculative", "mtp"],
        "answer": "Speculative decoding drafts K candidate tokens with a draft model or MTP head and verifies them in one parallel target forward pass using rejection sampling, delivering 2-3x faster lossless generation [1].",
        "facts": ["rejection sampling", "2-3x"],
    },
    {
        "questions": ["What is DoRA?", "How is DoRA better than LoRA?"],
        "query": "DoRA weight decomposed low rank adaptation LoRA",
        "keywords": ["dora", "lora"],
        "answer": "DoRA decomposes pretrained weights into magnitude and directional components and applies LoRA updates to the direction, outperforming standard LoRA with zero extra inference latency after merging [1].",
        "facts": ["magnitude", "direction"],
    },
    {
        "questions": ["What is Nova-LLM?", "Tell me about the Nova-LLM project."],
        "query": "Nova-LLM PyTorch architecture web search",
        "keywords": ["nova-llm", "pytorch"],
        "answer": "Nova-LLM is an open-source PyTorch LLM suite featuring GQA+QK-Norm, DeepSeek MLA, Gated DeltaNet-2, aux-free MoE, SuperBPE, MuonClip, FSDP2, SFT/DPO/RLVR, speculative decoding, and native web search [1].",
        "facts": ["nova-llm", "muonclip"],
    },
    {
        "questions": ["What is new in Python 3.13?", "Does Python 3.13 remove the GIL?"],
        "query": "Python 3.13 free-threaded GIL JIT PEP 703",
        "keywords": ["python", "3.13"],
        "answer": "Python 3.13 introduces an experimental free-threaded build (PEP 703) disabling the GIL, an experimental JIT compiler, and improved interactive REPL error messages [1].",
        "facts": ["pep 703", "gil", "jit"],
    },
    {
        "questions": ["What is FastAPI?", "Why use FastAPI for AI APIs?"],
        "query": "FastAPI Starlette Pydantic async SSE AI API",
        "keywords": ["fastapi", "sse"],
        "answer": "FastAPI is a modern Python web framework built on Starlette and Pydantic v2 offering automatic OpenAPI docs, async SSE streaming, and high throughput for AI inference APIs [1].",
        "facts": ["starlette", "pydantic", "sse"],
    },
    {
        "questions": ["How does Docker differ from a virtual machine?", "What is Docker?"],
        "query": "Docker containers vs virtual machines kernel",
        "keywords": ["docker", "containers"],
        "answer": "Docker packages applications and dependencies into lightweight Linux containers sharing the host OS kernel, starting in milliseconds compared to full hardware-virtualized VMs [1].",
        "facts": ["containers", "kernel"],
    },
    {
        "questions": ["What is pgvector in PostgreSQL?", "Can PostgreSQL do hybrid search?"],
        "query": "PostgreSQL pgvector HNSW hybrid search",
        "keywords": ["postgresql", "pgvector"],
        "answer": "PostgreSQL is an ACID-compliant relational database supporting full-text BM25/tsvector search and dense vector similarity via the pgvector HNSW extension [1].",
        "facts": ["acid", "pgvector", "hnsw"],
    },
    {
        "questions": ["What has the James Webb Space Telescope discovered?", "Where does JWST orbit?"],
        "query": "James Webb Space Telescope JWST L2 discoveries",
        "keywords": ["webb", "jwst"],
        "answer": "Orbiting the Sun-Earth L2 point 1.5 million km from Earth, the James Webb Space Telescope observes infrared light from the earliest galaxies formed 300 million years after the Big Bang and analyzes exoplanet atmospheres [1].",
        "facts": ["l2", "1.5 million", "infrared"],
    },
    {
        "questions": ["When did Chandrayaan-3 land on the Moon?", "Tell me about ISRO Chandrayaan-3."],
        "query": "ISRO Chandrayaan-3 lunar south pole landing date",
        "keywords": ["chandrayaan", "isro"],
        "answer": "India's ISRO Chandrayaan-3 achieved a historic soft landing near the Moon's south pole on August 23, 2023, making India the fourth nation to soft-land on the Moon and the first near the lunar south pole [1].",
        "facts": ["august 23, 2023", "south pole"],
    },
    {
        "questions": ["What is the exact speed of light?", "What is Einstein's general relativity?"],
        "query": "speed of light meters per second general relativity",
        "keywords": ["speed of light", "relativity"],
        "answer": "The speed of light in vacuum is exactly 299,792,458 meters per second (~300,000 km/s), and Einstein's general relativity describes gravity as the curvature of spacetime caused by mass and energy [1].",
        "facts": ["299,792,458", "spacetime"],
    },
    {
        "questions": ["What is photosynthesis?", "How do plants perform photosynthesis?"],
        "query": "photosynthesis chloroplasts glucose oxygen CO2",
        "keywords": ["photosynthesis", "glucose"],
        "answer": "Photosynthesis converts solar energy, water (H2O), and carbon dioxide (CO2) inside chloroplasts into glucose (C6H12O6) and releases molecular oxygen (O2) [1].",
        "facts": ["chloroplasts", "glucose", "oxygen"],
    },
    {
        "questions": ["How does CRISPR-Cas9 work?", "What is CRISPR?"],
        "query": "CRISPR-Cas9 guide RNA DNA gene editing",
        "keywords": ["crispr", "dna"],
        "answer": "CRISPR-Cas9 uses a programmable guide RNA to direct the Cas9 enzyme to cut specific DNA sequences, enabling precise gene knockout, repair, and therapeutic editing [1].",
        "facts": ["guide rna", "cas9", "dna"],
    },
    {
        "questions": ["How does quantum computing work?", "What are qubits?"],
        "query": "quantum computing qubits superposition error correction",
        "keywords": ["quantum", "qubits"],
        "answer": "Quantum computers encode information in qubits using superposition and entanglement, while surface codes and logical qubits suppress hardware decoherence errors [1].",
        "facts": ["superposition", "entanglement"],
    },
    {
        "questions": ["What is the capital of India?", "Tell me key facts about India."],
        "query": "India capital New Delhi population economy",
        "keywords": ["india", "new delhi"],
        "answer": "India's capital is New Delhi, its financial center is Mumbai, and Bengaluru is a global IT hub; India is the world's most populous nation (over 1.4 billion people) and largest democracy [1].",
        "facts": ["new delhi", "1.4 billion"],
    },
    {
        "questions": ["What are the capitals of France and Japan?", "What is the capital of Japan?"],
        "query": "world capitals France Paris Japan Tokyo",
        "keywords": ["france", "japan"],
        "answer": "The capital of France is Paris and the capital of Japan is Tokyo (along with London for the UK and Washington, D.C. for the United States) [1].",
        "facts": ["paris", "tokyo"],
    },
    {
        "questions": ["How tall is Mount Everest?", "What is the tallest mountain on Earth?"],
        "query": "Mount Everest height meters Himalayas",
        "keywords": ["everest", "height"],
        "answer": "Mount Everest in the Himalayas on the Nepal-Tibet border is Earth's highest mountain above sea level at 8,848.86 meters (29,031.7 feet) [1].",
        "facts": ["8,848.86", "himalayas"],
    },
    {
        "questions": ["What is the largest ocean on Earth?", "How deep is the Mariana Trench?"],
        "query": "largest ocean Pacific Mariana Trench depth",
        "keywords": ["pacific", "ocean"],
        "answer": "The Pacific Ocean is Earth's largest and deepest ocean (~165 million sq km) and contains the Mariana Trench (~10,984 m deep) [1].",
        "facts": ["pacific", "10,984"],
    },
    {
        "questions": ["Which planet is the largest in our solar system?", "Why is Mars red?"],
        "query": "solar system planets Jupiter Mars Saturn",
        "keywords": ["jupiter", "mars"],
        "answer": "Jupiter is the largest planet in our solar system (famous for its Great Red Spot), Mars is the Red Planet due to iron oxide dust, and Saturn is renowned for its icy ring system [1].",
        "facts": ["jupiter", "mars", "iron oxide"],
    },
    {
        "questions": ["What is HTTP/3 and QUIC?", "How does HTTP/3 improve web speed?"],
        "query": "HTTP/3 QUIC UDP TLS 1.3 web latency",
        "keywords": ["http/3", "quic"],
        "answer": "HTTP/3 runs over the UDP-based QUIC transport protocol with built-in TLS 1.3 encryption, eliminating TCP head-of-line blocking and reducing web connection latency [1].",
        "facts": ["quic", "udp", "tls 1.3"],
    },
    {
        "questions": ["What is OAuth 2.0 and JWT?", "How does OAuth 2.0 work?"],
        "query": "OAuth 2.0 PKCE JWT authentication tokens",
        "keywords": ["oauth", "jwt"],
        "answer": "OAuth 2.0 delegates authorization using short-lived access tokens and PKCE for public clients, while signed JWTs verify stateless API claims without exposing user passwords [1].",
        "facts": ["pkce", "jwt"],
    },
    {
        "questions": ["Who invented the World Wide Web?", "When was the World Wide Web created?"],
        "query": "World Wide Web Tim Berners-Lee CERN 1989",
        "keywords": ["world wide web", "berners-lee"],
        "answer": "Tim Berners-Lee invented the World Wide Web in 1989 at CERN, creating HTTP, HTML, and the first web browser to help researchers share documents globally [1].",
        "facts": ["tim berners-lee", "1989", "cern"],
    },
    {
        "questions": ["What is NVIDIA Blackwell?", "What precision does NVIDIA Blackwell support?"],
        "query": "NVIDIA Blackwell GPU FP4 FP8 NVLink 5",
        "keywords": ["blackwell", "nvidia"],
        "answer": "NVIDIA Blackwell GPUs introduce 5th-generation Tensor Cores with native FP4/FP8 precision and NVLink 5 interconnects, delivering major speedups for trillion-parameter LLM training and inference [1].",
        "facts": ["fp4", "fp8", "nvlink 5"],
    },
]

# Direct-answer (no wasteful search) & honest-abstention scenarios
DIRECT_SCENARIOS: list[tuple[list[str], str, str]] = [
    (
        ["Hello", "Hi", "Hey!"],
        "Hello! I am Nova-Search, an agentic web-searching LLM. Ask me any question and I will search, verify sources, and cite my answer!",
        "hello",
    ),
    (
        ["Who are you?", "What can you do?"],
        "I am Nova-Search, a web-search specialist LLM trained with MuonClip, tool-calling SFT, groundedness DPO, and search RLVR. I emit web_search queries, read live snippets, and synthesize answers with inline citations.",
        "nova-search",
    ),
    (
        ["What is 2 + 2?", "What is 2+2?"],
        "2 + 2 equals 4.",
        "4",
    ),
    (
        ["What is 12 * 12?"],
        "12 * 12 equals 144.",
        "144",
    ),
    (
        ["Thanks!", "Thank you"],
        "You are welcome! Ask me anytime you need verified, cited answers.",
        "welcome",
    ),
]

ABSTENTION_SCENARIOS: list[tuple[str, str, str]] = [
    (
        "What is the secret password of Project Xylophone-99?",
        "Project Xylophone-99 secret password",
        "I could not find any relevant information about Project Xylophone-99 in the web search results, so I cannot verify an answer.",
    ),
    (
        "Who won the 2099 Mars Olympics swimming gold medal?",
        "2099 Mars Olympics swimming gold medal winner",
        "The web search results do not contain any information about a 2099 Mars Olympics event, so I cannot verify that claim.",
    ),
]


def _make_tool_call(query: str) -> str:
    payload = json.dumps({"name": "web_search", "arguments": {"query": query}})
    return f"<|tool_call|>{payload}</|tool_call|>"


def build_web_search_datasets(index: BM25LocalWebIndex) -> tuple[
    list[str],
    list[list[dict[str, str]]],
    list[DPOExample],
    list[RLVRPrompt],
]:
    pretrain_docs: list[str] = []
    sft_convs: list[list[dict[str, str]]] = []
    dpo_pairs: list[DPOExample] = []
    rlvr_prompts: list[RLVRPrompt] = []

    # Include raw web corpus docs for Stage 1 domain pre-training
    for d in index.docs:
        pretrain_docs.append(f"Title: {d.title}\nSource: {d.domain} ({d.date})\nSummary: {d.snippet}\n")

    # Build exact multi-turn Search -> Tool Result -> Cited Synthesis SFT traces
    for sc in SEARCH_SCENARIOS:
        tc_str = _make_tool_call(sc["query"])
        results = index.search(sc["query"], top_k=2)
        res_block = f"<|tool_result|>{format_search_results(results)}</|tool_result|>"

        for q in sc["questions"]:
            pretrain_docs.append(f"Query: {q}\nTool: {tc_str}\nResults:\n{res_block}\nAnswer: {sc['answer']}\n")
            sft_convs.append(
                [
                    {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": tc_str},
                    {"role": "tool", "content": res_block},
                    {"role": "assistant", "content": sc["answer"]},
                ]
            )

        # Groundedness DPO Pair 1: Valid tool call vs. hallucinating without searching
        q0 = sc["questions"][0]
        dpo_pairs.append(
            DPOExample(
                prompt=q0,
                chosen=tc_str,
                rejected="I am guessing the answer without checking any web sources.",
            )
        )
        # Groundedness DPO Pair 2: Cited factual synthesis vs. fabricated citation
        dpo_pairs.append(
            DPOExample(
                prompt=f"{q0}\n{res_block}",
                chosen=sc["answer"],
                rejected="According to an unknown source [9], none of those facts are real.",
            )
        )

        # RLVR verifiable tool-calling prompt
        rlvr_prompts.append(
            RLVRPrompt(
                prompt=q0,
                metadata={"task": "tool_call", "keywords": sc["keywords"]},
            )
        )

    # Direct-answer (no-search) turns
    for qs, ans, key in DIRECT_SCENARIOS:
        for q in qs:
            pretrain_docs.append(f"User: {q}\nAssistant: {ans}\n")
            sft_convs.append(
                [
                    {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans},
                ]
            )
        rlvr_prompts.append(RLVRPrompt(prompt=qs[0], metadata={"task": "direct_no_search", "expected": key}))

    # Abstention traces (empty search results -> honest refusal)
    for q, query, ans in ABSTENTION_SCENARIOS:
        tc_str = _make_tool_call(query)
        empty_res = "<|tool_result|>No relevant web search results found.</|tool_result|>"
        sft_convs.append(
            [
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": tc_str},
                {"role": "tool", "content": empty_res},
                {"role": "assistant", "content": ans},
            ]
        )

    return pretrain_docs, sft_convs, dpo_pairs, rlvr_prompts


def _model_cfg(vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=160,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=512,
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
        raise FileNotFoundError(f"Missing prior checkpoint {path}. Run earlier parts first.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = NovaLM(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    return model, payload.get("metrics", {})


def run_part1_pretrain(base_dir: Path, index: BM25LocalWebIndex, steps: int = 100) -> dict:
    t0 = time.perf_counter()
    seed_everything(101)
    pretrain_docs, _, _, _ = build_web_search_datasets(index)

    tokenizer = ByteBPETokenizer.train(
        pretrain_docs * 2, vocab_size=1536, superbpe_transition_vocab=1152, min_pair_freq=2
    )
    tokenizer.save(base_dir / "tokenizer.json")
    tok_stats = evaluate_tokenizer(tokenizer, pretrain_docs[:25])

    shard_dir = base_dir / "shards"
    write_token_shard(pretrain_docs * 6, tokenizer, shard_dir / "train")
    write_token_shard(pretrain_docs[:20], tokenizer, shard_dir / "val")

    cfg = _model_cfg(tokenizer.vocab_size)
    model = NovaLM(cfg)
    train_cfg = TrainConfig(
        output_dir=str(base_dir / "part1_logs"),
        seed=101,
        max_steps=steps,
        micro_batch_size=8,
        grad_accum_steps=1,
        seq_len=96,
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
    train_loader = PackedShardLoader(TokenShardReader(shard_dir / "train"), batch_size=8, seq_len=96, seed=101)
    val_loader = PackedShardLoader(TokenShardReader(shard_dir / "val"), batch_size=4, seq_len=96, seed=202)
    hist = PreTrainer(model, train_loader, train_cfg, val_loader=val_loader, device="cpu", quiet=False).fit()

    bpb = evaluate_bpb_and_ppl(model, tokenizer, pretrain_docs[:15])
    metrics = {
        "part": 1,
        "stage": "web_pretrain",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "params": model.count_parameters(),
        "tokenizer": tok_stats,
        "loss_start": hist[0]["loss"],
        "loss_end": hist[-1]["loss"],
        "eval_bpb": bpb["bits_per_byte"],
        "eval_ppl": bpb["perplexity"],
    }
    _save_stage(base_dir / "checkpoints" / "part1_pretrain.pt", model, "part1_pretrain", metrics)
    print(f"[PART 1 DONE] Web Pre-Training: loss {metrics['loss_start']:.4f} -> {metrics['loss_end']:.4f} | BPB={bpb['bits_per_byte']:.3f} ({metrics['elapsed_s']}s)")
    return metrics


def run_part2_sft(base_dir: Path, index: BM25LocalWebIndex, steps: int = 440) -> dict:
    t0 = time.perf_counter()
    seed_everything(102)
    tokenizer = ByteBPETokenizer.load(base_dir / "tokenizer.json")
    model, _ = _load_stage(base_dir / "checkpoints" / "part1_pretrain.pt")
    _, sft_convs, _, _ = build_web_search_datasets(index)

    sft_cfg = PostTrainConfig(
        stage="sft",
        output_dir=str(base_dir / "part2_logs"),
        seed=102,
        max_steps=steps,
        micro_batch_size=6,
        grad_accum_steps=1,
        max_seq_len=512,
        lr=3e-3,
        weight_decay=0.0,
        use_lora=False,
    )
    hist = SFTTrainer(model, tokenizer, sft_convs, sft_cfg, device="cpu").fit()
    metrics = {
        "part": 2,
        "stage": "agentic_search_sft",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "conversations": len(sft_convs),
        "sft_loss_start": hist[0]["sft_loss"],
        "sft_loss_end": hist[-1]["sft_loss"],
    }
    _save_stage(base_dir / "checkpoints" / "part2_sft.pt", model, "part2_sft", metrics)
    print(f"[PART 2 DONE] Agentic Search SFT ({len(sft_convs)} multi-turn traces): loss {metrics['sft_loss_start']:.4f} -> {metrics['sft_loss_end']:.4f} ({metrics['elapsed_s']}s)")
    return metrics


def run_part3_dpo(base_dir: Path, index: BM25LocalWebIndex, steps: int = 16) -> dict:
    t0 = time.perf_counter()
    seed_everything(103)
    tokenizer = ByteBPETokenizer.load(base_dir / "tokenizer.json")
    model, _ = _load_stage(base_dir / "checkpoints" / "part2_sft.pt")
    _, _, dpo_pairs, _ = build_web_search_datasets(index)

    dpo_cfg = PostTrainConfig(
        stage="dpo",
        output_dir=str(base_dir / "part3_logs"),
        seed=103,
        max_steps=steps,
        micro_batch_size=4,
        grad_accum_steps=1,
        max_seq_len=256,
        lr=6e-5,
        dpo_beta=0.1,
        dpo_nll_weight=0.5,
    )
    hist = DPOTrainer(model, tokenizer, dpo_pairs, dpo_cfg, device="cpu").fit()
    metrics = {
        "part": 3,
        "stage": "groundedness_dpo",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "pairs": len(dpo_pairs),
        "pref_loss_start": hist[0]["pref_loss"],
        "pref_loss_end": hist[-1]["pref_loss"],
        "win_acc": hist[-1]["win_acc"],
    }
    _save_stage(base_dir / "checkpoints" / "part3_dpo.pt", model, "part3_dpo", metrics)
    print(f"[PART 3 DONE] Groundedness DPO: loss {metrics['pref_loss_start']:.4f} -> {metrics['pref_loss_end']:.4f} | win_acc={metrics['win_acc']:.2f} ({metrics['elapsed_s']}s)")
    return metrics


def run_part4_rlvr_and_eval(base_dir: Path, index: BM25LocalWebIndex, steps: int = 6) -> dict:
    t0 = time.perf_counter()
    seed_everything(104)
    tokenizer = ByteBPETokenizer.load(base_dir / "tokenizer.json")
    model, _ = _load_stage(base_dir / "checkpoints" / "part3_dpo.pt")
    _, _, _, rlvr_prompts = build_web_search_datasets(index)

    rl_cfg = PostTrainConfig(
        stage="dapo",
        output_dir=str(base_dir / "part4_logs"),
        seed=104,
        max_steps=steps,
        lr=2e-5,
        rl_group_size=4,
        rl_max_new_tokens=36,
        rl_kl_beta=0.02,
        rl_dynamic_sampling=True,
        rl_ppo_epochs=1,
    )
    verifier = WebSearchRLVRVerifier()
    hist = RLVRTrainer(model, tokenizer, rlvr_prompts, rl_cfg, verifier=verifier, device="cpu").fit()

    export_dir = base_dir / "model"
    export_safetensors(model, export_dir, tokenizer=tokenizer)
    reloaded = load_safetensors_model(export_dir, device="cpu").eval()

    # Dedicated End-to-End Agentic Web-Search Benchmark!
    agent = SearchAgent(reloaded, tokenizer, search_backend=HybridSearchBackend(local_index=index), top_k_results=2)
    eval_cases = [
        ("What is DeepSeek-V3?", True, ["deepseek", "mla"], ["671b", "mla"]),
        ("What is MuonClip?", True, ["muonclip", "qk-clip"], ["newton-schulz", "qk-clip"]),
        ("When did Chandrayaan-3 land on the Moon?", True, ["chandrayaan", "isro"], ["august 23, 2023", "south pole"]),
        ("What has the James Webb Space Telescope discovered?", True, ["webb", "jwst"], ["l2", "infrared"]),
        ("What is new in Python 3.13?", True, ["python", "3.13"], ["pep 703", "gil"]),
        ("How tall is Mount Everest?", True, ["everest"], ["8,848.86"]),
        ("What is HTTP/3 and QUIC?", True, ["http/3", "quic"], ["quic", "tls 1.3"]),
        ("What is 12 * 12?", False, [], ["144"]),
    ]

    valid_tool_calls = 0
    search_cases = 0
    grounded_scores: list[float] = []
    routing_correct = 0
    samples = []

    print("\n=== NOVA-SEARCH END-TO-END AGENTIC WEB-SEARCH BENCHMARK ===")
    for q, expect_search, exp_kw, exp_facts in eval_cases:
        resp = agent.run(q, max_tokens=112, temperature=0.0)
        if resp.used_search == expect_search:
            routing_correct += 1
        if expect_search:
            search_cases += 1
            if resp.tool_calls and verify_search_tool_call(resp.raw_turns[0], exp_kw) >= 0.6:
                valid_tool_calls += 1
            g_score = verify_grounded_synthesis(resp.answer, exp_facts, allowed_citation_ids=[1, 2])
            grounded_scores.append(g_score)
        else:
            g_score = 1.0 if all(f in resp.answer.lower() for f in exp_facts) else 0.0
            grounded_scores.append(g_score)

        tc_display = resp.tool_calls[0]["arguments"]["query"] if resp.tool_calls else "(direct answer - no search)"
        print(f"Q: {q}\n  🔍 Query: {tc_display}\n  ✅ Answer (score={g_score:.2f}): {resp.answer}\n")
        samples.append(
            {
                "question": q,
                "used_search": resp.used_search,
                "query": tc_display,
                "answer": resp.answer,
                "grounded_score": g_score,
            }
        )

    bench = {
        "routing_accuracy": round(routing_correct / len(eval_cases), 4),
        "tool_call_precision": round(valid_tool_calls / max(1, search_cases), 4),
        "mean_grounded_citation_score": round(sum(grounded_scores) / max(1, len(grounded_scores)), 4),
    }
    metrics = {
        "part": 4,
        "stage": "web_search_rlvr_dapo_export",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "rl_steps": len(hist),
        "benchmark": bench,
        "export_dir": str(export_dir),
        "samples": samples,
    }
    _save_stage(base_dir / "checkpoints" / "part4_rlvr.pt", model, "part4_rlvr", metrics)
    print(f"[PART 4 DONE] RLVR (DAPO) + Benchmark: {bench} | Exported to {export_dir} ({metrics['elapsed_s']}s)")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Nova-Search (Web-Search Native Agentic LLM)")
    parser.add_argument("--part", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--out-dir", default="runs/nova_web_search")
    args = parser.parse_args()

    base_dir = Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    report_path = base_dir / "web_search_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    index = BM25LocalWebIndex()
    parts = ["1", "2", "3", "4"] if args.part == "all" else [args.part]
    for p in parts:
        if p == "1":
            report["part1"] = run_part1_pretrain(base_dir, index)
        elif p == "2":
            report["part2"] = run_part2_sft(base_dir, index)
        elif p == "3":
            report["part3"] = run_part3_dpo(base_dir, index)
        elif p == "4":
            report["part4"] = run_part4_rlvr_and_eval(base_dir, index)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
