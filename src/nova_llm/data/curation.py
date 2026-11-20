"""Pre-training data hygiene: heuristic filtering, PII scrub, exact + MinHash dedup, decontamination."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def scrub_pii(text: str) -> str:
    """Redact common PII patterns (emails, phone numbers, IPv4 addresses, SSNs)."""
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _SSN_RE.sub("<SSN>", text)
    text = _IPV4_RE.sub("<IP>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    return text


def passes_heuristics(
    text: str,
    min_chars: int = 20,
    max_line_dup_ratio: float = 0.5,
    min_alpha_ratio: float = 0.3,
) -> bool:
    """Apply FineWeb/DCLM-style lightweight heuristic quality filters."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    alpha = sum(1 for c in stripped if c.isalpha())
    if alpha / len(stripped) < min_alpha_ratio:
        return False
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if len(lines) >= 4:
        counts = Counter(lines)
        dup_lines = sum(c - 1 for c in counts.values() if c > 1)
        if dup_lines / len(lines) > max_line_dup_ratio:
            return False
    return True


def exact_deduplicate(docs: list[str]) -> list[str]:
    """Exact SHA-256 document deduplication preserving first occurrence order."""
    seen: set[str] = set()
    kept: list[str] = []
    for doc in docs:
        h = hashlib.sha256(doc.strip().encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            kept.append(doc)
    return kept


def _word_shingles(text: str, n: int = 5) -> set[bytes]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return {" ".join(words).encode("utf-8")} if words else set()
    return {" ".join(words[i : i + n]).encode("utf-8") for i in range(len(words) - n + 1)}


def _minhash_signature(shingles: set[bytes], num_perm: int = 64) -> list[int]:
    """Deterministic MinHash signature using salted BLAKE2b hashes."""
    if not shingles:
        return [0] * num_perm
    sig = []
    for perm in range(num_perm):
        salt = perm.to_bytes(4, "little")
        min_h = min(
            int.from_bytes(hashlib.blake2b(salt + sh, digest_size=8).digest(), "little")
            for sh in shingles
        )
        sig.append(min_h)
    return sig


def minhash_deduplicate(
    docs: list[str],
    ngram_size: int = 5,
    num_perm: int = 64,
    num_bands: int = 16,
    jaccard_threshold: float = 0.75,
) -> list[str]:
    """Fuzzy near-duplicate removal using MinHash + LSH banding."""
    rows_per_band = max(1, num_perm // num_bands)
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    kept_docs: list[str] = []
    kept_shingles: list[set[bytes]] = []

    for doc in docs:
        shingles = _word_shingles(doc, n=ngram_size)
        sig = _minhash_signature(shingles, num_perm=num_perm)
        candidates: set[int] = set()
        band_keys = []
        for b in range(num_bands):
            start = b * rows_per_band
            key = (b, tuple(sig[start : start + rows_per_band]))
            band_keys.append(key)
            if key in buckets:
                candidates.update(buckets[key])

        is_dup = False
        for cand_idx in candidates:
            other = kept_shingles[cand_idx]
            union = len(shingles | other)
            if union > 0 and (len(shingles & other) / union) >= jaccard_threshold:
                is_dup = True
                break

        if not is_dup:
            new_idx = len(kept_docs)
            kept_docs.append(doc)
            kept_shingles.append(shingles)
            for key in band_keys:
                buckets.setdefault(key, []).append(new_idx)

    return kept_docs


def decontaminate_against_benchmarks(
    docs: list[str],
    benchmark_texts: list[str],
    ngram_size: int = 10,
) -> tuple[list[str], dict[str, Any]]:
    """Remove training documents containing any n-gram overlap with benchmark evaluation texts."""
    bench_ngrams: set[tuple[str, ...]] = set()
    for btext in benchmark_texts:
        words = re.findall(r"\w+", btext.lower())
        if len(words) >= ngram_size:
            for i in range(len(words) - ngram_size + 1):
                bench_ngrams.add(tuple(words[i : i + ngram_size]))
        elif words:
            bench_ngrams.add(tuple(words))

    clean: list[str] = []
    removed_indices: list[int] = []
    for idx, doc in enumerate(docs):
        words = re.findall(r"\w+", doc.lower())
        contaminated = False
        if len(words) >= ngram_size:
            for i in range(len(words) - ngram_size + 1):
                if tuple(words[i : i + ngram_size]) in bench_ngrams:
                    contaminated = True
                    break
        elif tuple(words) in bench_ngrams:
            contaminated = True

        if contaminated:
            removed_indices.append(idx)
        else:
            clean.append(doc)

    report = {
        "input_docs": len(docs),
        "kept_docs": len(clean),
        "contaminated_docs": len(removed_indices),
        "contamination_rate": len(removed_indices) / max(1, len(docs)),
        "ngram_size": ngram_size,
    }
    return clean, report


def curate_corpus(
    docs: list[str],
    benchmark_texts: list[str] | None = None,
    min_chars: int = 20,
    fuzzy_dedup: bool = True,
    jaccard_threshold: float = 0.75,
    decontam_ngram: int = 10,
) -> tuple[list[str], dict[str, Any]]:
    """Full curation pipeline: PII scrub -> heuristic filter -> exact dedup -> MinHash -> decontam."""
    n_raw = len(docs)
    scrubbed = [scrub_pii(d) for d in docs]
    filtered = [d for d in scrubbed if passes_heuristics(d, min_chars=min_chars)]
    n_after_heur = len(filtered)
    deduped = exact_deduplicate(filtered)
    n_after_exact = len(deduped)
    if fuzzy_dedup and len(deduped) > 1:
        deduped = minhash_deduplicate(deduped, jaccard_threshold=jaccard_threshold)
    n_after_fuzzy = len(deduped)

    decontam_report: dict[str, Any] = {"contaminated_docs": 0}
    if benchmark_texts:
        deduped, decontam_report = decontaminate_against_benchmarks(
            deduped, benchmark_texts, ngram_size=decontam_ngram
        )

    stats = {
        "raw_docs": n_raw,
        "after_heuristics": n_after_heur,
        "after_exact_dedup": n_after_exact,
        "after_fuzzy_dedup": n_after_fuzzy,
        "final_docs": len(deduped),
        "decontamination": decontam_report,
    }
    return deduped, stats
