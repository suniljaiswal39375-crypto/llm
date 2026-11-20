"""Verification tests for Byte-Level BPE, SuperBPE, SFT chat masking, curation & packed shards."""

from __future__ import annotations

from pathlib import Path

import torch

from nova_llm.data import (
    MultiSourceMixer,
    PackedShardLoader,
    TokenShardReader,
    curate_corpus,
    decontaminate_against_benchmarks,
    scrub_pii,
    write_token_shard,
)
from nova_llm.tokenizer import ByteBPETokenizer, evaluate_tokenizer


def test_bpe_and_superbpe_roundtrip_and_encoding_efficiency(tmp_path: Path) -> None:
    """SuperBPE (2-stage curriculum) must be 100% lossless UTF-8 and use fewer tokens than BPE."""
    corpus = [
        "by the way, the large language model is trained in the cloud with high efficiency.\n",
        "by the way, the large language model uses attention in the transformer layers.\n",
        "in the end, the large language model achieves strong accuracy by the way.\n",
        "Unicode check: नमस्ते दुनिया! 🚀 α+β=γ — café naïve.\n",
    ] * 8

    bpe = ByteBPETokenizer.train(corpus, vocab_size=360, superbpe_transition_vocab=None, min_pair_freq=2)
    superbpe = ByteBPETokenizer.train(corpus, vocab_size=360, superbpe_transition_vocab=320, min_pair_freq=2)

    bpe_stats = evaluate_tokenizer(bpe, corpus[:4])
    sbpe_stats = evaluate_tokenizer(superbpe, corpus[:4])

    assert bpe_stats["roundtrip_failures"] == 0
    assert sbpe_stats["roundtrip_failures"] == 0
    assert sbpe_stats["superbpe_enabled"] is True
    assert sbpe_stats["bytes_per_token"] > bpe_stats["bytes_per_token"], (
        f"Expected SuperBPE ({sbpe_stats['bytes_per_token']:.2f} B/tok) > BPE ({bpe_stats['bytes_per_token']:.2f} B/tok)"
    )

    # Verify save / load exact parity
    save_p = tmp_path / "superbpe.json"
    superbpe.save(save_p)
    loaded = ByteBPETokenizer.load(save_p)
    sample = "by the way, the large language model works! 🚀"
    assert loaded.encode(sample) == superbpe.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


def test_chat_sft_masking_hides_prompt_and_keeps_assistant_eos() -> None:
    """Verify prompt tokens receive -100 while assistant tokens AND <|im_end|> are supervised."""
    tok = ByteBPETokenizer.train(["hello world assistant response"], vocab_size=300)
    msgs = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "thinking": "2+2=4", "content": "4"},
    ]
    inp, lbl = tok.encode_chat_for_sft(msgs)
    assert len(inp) == len(lbl)
    # Leading prompt positions must be masked (-100)
    assert lbl[0] == -100 and lbl[5] == -100
    # Unmasked region must decode to the assistant response + closing <|im_end|>
    unmasked_ids = [t for t in lbl if t != -100]
    decoded_target = tok.decode(unmasked_ids, skip_special_tokens=False)
    assert "<|think|>" in decoded_target and "4" in decoded_target and "<|im_end|>" in decoded_target
    assert "What is 2+2?" not in decoded_target


def test_curation_pii_dedup_and_decontamination() -> None:
    """Verify PII scrubbing, exact + MinHash fuzzy dedup, and n-gram benchmark decontamination."""
    assert "<EMAIL>" in scrub_pii("Reach me at alice@example.org or 192.168.1.10")
    assert "<IP>" in scrub_pii("Server at 192.168.1.10 failed")

    docs = [
        "A completely clean scientific article about quantum error correction and stabilizer codes in physics.",
        "A completely clean scientific article about quantum error correction and stabilizer codes in physics.",  # exact dup
        "A completely clean scientific article about quantum error correction and stabilizer codes in modern physics.",  # near dup
        "Short",  # too short
        "Here is the leaked benchmark question: what is the exact orbital velocity of Jupiter around the sun in km/s?",
    ]
    bench = ["what is the exact orbital velocity of Jupiter around the sun in km/s?"]
    cleaned, stats = curate_corpus(docs, benchmark_texts=bench, min_chars=20, fuzzy_dedup=True, decontam_ngram=6)
    assert stats["after_exact_dedup"] < stats["after_heuristics"]
    assert stats["decontamination"]["contaminated_docs"] == 1
    assert len(cleaned) == 1


def test_binary_shards_cross_doc_target_masking_and_rank_disjointness(tmp_path: Path) -> None:
    """Verify cross-document target transitions are masked to -100 and multi-rank loaders are disjoint."""
    docs = [f"Document {i}: concise text about topic {i} for shard boundary check." for i in range(30)]
    tok = ByteBPETokenizer.train(docs, vocab_size=300)
    write_token_shard(docs, tok, tmp_path / "shard_a")
    reader = TokenShardReader(tmp_path / "shard_a")

    loader_r0 = PackedShardLoader(reader, batch_size=2, seq_len=24, rank=0, world_size=2, seed=42)
    loader_r1 = PackedShardLoader(reader, batch_size=2, seq_len=24, rank=1, world_size=2, seed=42)
    b0 = loader_r0.next_batch()
    b1 = loader_r1.next_batch()
    assert not torch.equal(b0["input_ids"], b1["input_ids"]), "Rank 0 and Rank 1 fetched identical batches!"

    # Verify every cross-doc transition has target == -100
    d = b0["doc_ids"]
    y = b0["targets"]
    transitions = d[:, :-1] != d[:, 1:]
    # Where doc_id changes between pos t and t+1, y[:, t] is predicting token t+1 (different doc) -> must be -100!
    assert (y[:, :-1][transitions] == -100).all(), "Cross-document transition was NOT masked to -100!"

    # Verify MultiSourceMixer deterministic state restore
    mixer = MultiSourceMixer({"a": loader_r0, "b": loader_r1}, {"a": 0.7, "b": 0.3}, seed=9)
    st = mixer.state_dict()
    ref_b = mixer.next_batch()
    mixer.load_state_dict(st)
    replayed_b = mixer.next_batch()
    assert torch.equal(ref_b["input_ids"], replayed_b["input_ids"])
