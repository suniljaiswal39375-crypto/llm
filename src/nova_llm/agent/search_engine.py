"""Pluggable Web Search Backends: genuine BM25 local web index + live HTTP provider adapters."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

import httpx


@dataclass
class WebDocument:
    title: str
    url: str
    snippet: str
    date: str = "2026-11"
    tags: tuple[str, ...] = ()

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc or "web"


@dataclass
class SearchResult:
    id: int
    title: str
    url: str
    snippet: str
    date: str
    domain: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "by", "from", "as", "it", "be", "this", "that",
    "what", "who", "when", "where", "how", "why", "which", "do", "does", "did",
}


def _tokenize_search(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    filtered = [t for t in toks if t not in _STOPWORDS]
    return filtered or toks


def build_default_web_corpus() -> list[WebDocument]:
    """Rich multi-domain web corpus for offline/sandbox BM25 search & agentic training."""
    return [
        # Frontier AI & LLMs (2025-2026)
        WebDocument(
            title="DeepSeek-V3 Technical Report: MLA, Aux-Free MoE & MTP",
            url="https://arxiv.org/abs/2412.19437",
            snippet="DeepSeek-V3 is a 671B total / 37B active MoE model introducing Multi-Head Latent Attention (MLA), auxiliary-loss-free load balancing via dynamic router biases, and Multi-Token Prediction (MTP) heads.",
            date="2025-01",
            tags=("deepseek", "mla", "moe", "mtp", "llm"),
        ),
        WebDocument(
            title="Kimi K2 & MuonClip: Scaling Muon to 15.5T Tokens With Zero Spikes",
            url="https://arxiv.org/abs/2507.20534",
            snippet="Kimi K2 trains a 1T MoE model using the MuonClip optimizer, combining Newton-Schulz orthogonalized momentum with per-head QK-Clip to eliminate attention logit explosions across 15.5T tokens.",
            date="2025-07",
            tags=("kimi", "muon", "muonclip", "qk-clip", "optimizer"),
        ),
        WebDocument(
            title="SuperBPE: Space Travel for Language Models",
            url="https://arxiv.org/abs/2503.13423",
            snippet="SuperBPE uses a two-stage tokenizer curriculum transitioning from subwords to cross-whitespace superwords, cutting sequence token counts by 27-33% and improving downstream accuracy.",
            date="2025-03",
            tags=("superbpe", "tokenizer", "bpe", "compression"),
        ),
        WebDocument(
            title="Gated DeltaNet-2 & Hybrid Linear Attention Architectures",
            url="https://arxiv.org/abs/2412.06464",
            snippet="Hybrid 3:1 Gated DeltaNet linear attention interleaved with global attention reduces KV-cache memory by roughly 75% while matching or outperforming pure softmax transformers on long context.",
            date="2026-05",
            tags=("deltanet", "linear attention", "hybrid", "kv cache"),
        ),
        WebDocument(
            title="DAPO & Dr. GRPO: Open-Source RLVR Recipes for Reasoning LLMs",
            url="https://arxiv.org/abs/2503.14476",
            snippet="DAPO improves critic-free RLVR over GRPO with Clip-Higher asymmetric clipping, Dynamic Sampling of non-zero-advantage groups, token-level loss, and overlong reward shaping; Dr. GRPO removes length and std bias.",
            date="2025-03",
            tags=("dapo", "grpo", "rlvr", "reasoning", "reinforcement learning"),
        ),
        WebDocument(
            title="PyTorch FSDP2 & TorchTitan Distributed Training Guide",
            url="https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html",
            snippet="PyTorch FSDP2 uses DTensor-based per-parameter sharding via fully_shard(), replacing legacy FlatParameter FSDP1 with composable tensor parallelism and native torch.compile support.",
            date="2026-04",
            tags=("pytorch", "fsdp2", "distributed", "dtensor"),
        ),
        WebDocument(
            title="YaRN: Efficient Context Window Extension of Large Language Models",
            url="https://arxiv.org/abs/2309.00071",
            snippet="YaRN extends RoPE context windows by interpolating low-frequency dimensions, preserving high-frequency local bands, and scaling attention temperature by mscale = 0.1*ln(s)+1.",
            date="2024-02",
            tags=("yarn", "rope", "long context", "attention"),
        ),
        WebDocument(
            title="Speculative Decoding & EAGLE-3 / MTP Verification",
            url="https://huggingface.co/blog/speculative-decoding",
            snippet="Speculative decoding drafts K candidate tokens with a draft model or MTP head and verifies them in one parallel target forward pass using rejection sampling, yielding 2-3x faster lossless generation.",
            date="2026-02",
            tags=("speculative decoding", "eagle", "mtp", "inference"),
        ),
        WebDocument(
            title="DoRA: Weight-Decomposed Low-Rank Adaptation",
            url="https://arxiv.org/abs/2402.09353",
            snippet="DoRA decomposes pretrained weights into magnitude and directional components and applies LoRA updates to the direction, consistently outperforming standard LoRA with zero extra inference latency after merging.",
            date="2024-06",
            tags=("lora", "dora", "peft", "fine-tuning"),
        ),
        WebDocument(
            title="Nova-LLM Project Documentation",
            url="https://github.com/suniljaiswal39375-crypto/llm",
            snippet="Nova-LLM is an open-source PyTorch LLM suite featuring GQA+QK-Norm, DeepSeek MLA, Gated DeltaNet-2, aux-free MoE, SuperBPE, MuonClip, FSDP2, SFT/DPO/RLVR, speculative decoding, and native web search.",
            date="2026-11",
            tags=("nova-llm", "github", "pytorch", "search"),
        ),
        # Programming, Frameworks & Cloud
        WebDocument(
            title="Python 3.13 & Free-Threaded CPython Overview",
            url="https://docs.python.org/3/whatsnew/3.13.html",
            snippet="Python 3.13 introduces an experimental free-threaded build (PEP 703) disabling the GIL, an experimental JIT compiler, and improved interactive REPL error messages.",
            date="2025-10",
            tags=("python", "gil", "jit", "programming"),
        ),
        WebDocument(
            title="FastAPI High-Performance Async Web Framework",
            url="https://fastapi.tiangolo.com/",
            snippet="FastAPI is a modern Python web framework built on Starlette and Pydantic v2 offering automatic OpenAPI docs, async SSE streaming, and high throughput for AI inference APIs.",
            date="2026-06",
            tags=("fastapi", "python", "api", "sse", "backend"),
        ),
        WebDocument(
            title="Git Branching, Pull Requests & Merge Workflows",
            url="https://git-scm.com/book/en/v2",
            snippet="Git tracks repository snapshots via immutable commits; developers isolate features on branches, review diffs in Pull Requests, and integrate changes using merge, squash, or rebase.",
            date="2026-01",
            tags=("git", "github", "pull request", "version control"),
        ),
        WebDocument(
            title="Docker Containers vs Virtual Machines",
            url="https://docs.docker.com/get-started/overview/",
            snippet="Docker packages applications and dependencies into lightweight Linux containers sharing the host OS kernel, starting in milliseconds compared to full hardware-virtualized VMs.",
            date="2026-03",
            tags=("docker", "containers", "devops", "cloud"),
        ),
        WebDocument(
            title="PostgreSQL & pgvector for Hybrid Search",
            url="https://www.postgresql.org/docs/current/",
            snippet="PostgreSQL is an ACID-compliant relational database supporting full-text BM25/tsvector search and dense vector similarity via the pgvector HNSW extension.",
            date="2026-05",
            tags=("postgresql", "database", "sql", "vector search"),
        ),
        # Science, Space & Climate
        WebDocument(
            title="James Webb Space Telescope (JWST) Key Discoveries",
            url="https://science.nasa.gov/mission/webb/",
            snippet="Orbiting the Sun-Earth L2 point 1.5 million km from Earth, the James Webb Space Telescope observes infrared light from the earliest galaxies formed 300 million years after the Big Bang and analyzes exoplanet atmospheres.",
            date="2026-08",
            tags=("jwst", "nasa", "space", "astronomy", "telescope"),
        ),
        WebDocument(
            title="ISRO Chandrayaan-3 Lunar South Pole Landing",
            url="https://www.isro.gov.in/Chandrayaan3.html",
            snippet="India's ISRO Chandrayaan-3 mission achieved a historic soft landing near the Moon's south pole on August 23, 2023, making India the fourth nation to soft-land on the Moon and the first near the lunar south pole.",
            date="2025-08",
            tags=("isro", "india", "chandrayaan", "moon", "space"),
        ),
        WebDocument(
            title="Speed of Light & General Relativity Basics",
            url="https://physics.nist.gov/cgi-bin/cuu/Value?c",
            snippet="The speed of light in vacuum is exactly 299,792,458 meters per second (~300,000 km/s). Einstein's general relativity describes gravity as the curvature of spacetime caused by mass and energy.",
            date="2026-01",
            tags=("physics", "speed of light", "relativity", "gravity"),
        ),
        WebDocument(
            title="Photosynthesis Light and Calvin Cycle Reactions",
            url="https://www.britannica.com/science/photosynthesis",
            snippet="Photosynthesis converts solar energy, water (H2O), and carbon dioxide (CO2) inside chloroplasts into glucose (C6H12O6) and releases molecular oxygen (O2).",
            date="2026-02",
            tags=("biology", "photosynthesis", "plants", "oxygen"),
        ),
        WebDocument(
            title="CRISPR-Cas9 Gene Editing Mechanism",
            url="https://www.genome.gov/genetics-glossary/CRISPR",
            snippet="CRISPR-Cas9 uses a programmable guide RNA to direct the Cas9 enzyme to cut specific DNA sequences, enabling precise gene knockout, repair, and therapeutic editing.",
            date="2026-04",
            tags=("crispr", "dna", "genetics", "biology"),
        ),
        WebDocument(
            title="Quantum Computing Qubits & Error Correction",
            url="https://quantum.ibm.com/docs",
            snippet="Quantum computers encode information in qubits using superposition and entanglement; surface codes and logical qubits suppress hardware decoherence errors.",
            date="2026-07",
            tags=("quantum", "qubits", "computing", "physics"),
        ),
        # India, Geography & Global Facts
        WebDocument(
            title="India Overview: Capital, Population, Economy & Tech Hub",
            url="https://www.india.gov.in/india-glance/profile",
            snippet="India's capital is New Delhi and its financial center is Mumbai; Bengaluru is a global IT hub. India is the world's most populous nation (over 1.4 billion people) and the world's largest democracy.",
            date="2026-09",
            tags=("india", "new delhi", "geography", "economy"),
        ),
        WebDocument(
            title="World Capitals Reference: France, Japan, UK, USA",
            url="https://www.cia.gov/the-world-factbook/",
            snippet="Major national capitals include Paris (France), Tokyo (Japan), London (United Kingdom), Washington, D.C. (United States), Ottawa (Canada), and Canberra (Australia).",
            date="2026-06",
            tags=("capitals", "france", "japan", "uk", "usa", "geography"),
        ),
        WebDocument(
            title="Mount Everest & Earth's Highest Peaks",
            url="https://www.nationalgeographic.com/adventure/article/mount-everest",
            snippet="Mount Everest in the Himalayas on the Nepal-Tibet border is Earth's highest mountain above sea level at 8,848.86 meters (29,031.7 feet).",
            date="2026-03",
            tags=("everest", "mountain", "himalayas", "geography"),
        ),
        WebDocument(
            title="World Oceans & Continents Geographic Data",
            url="https://oceanservice.noaa.gov/facts/biggestocean.html",
            snippet="The Pacific Ocean is Earth's largest and deepest ocean, spanning ~165 million square kilometers and containing the Mariana Trench (~10,984 m deep). Earth has seven continents.",
            date="2026-05",
            tags=("pacific", "ocean", "continents", "mariana trench"),
        ),
        WebDocument(
            title="Solar System Planets: Jupiter, Mars, Earth & Saturn",
            url="https://science.nasa.gov/solar-system/planets/",
            snippet="Jupiter is the largest planet in our solar system (famous for its Great Red Spot), Mars is the Red Planet due to iron oxide dust, and Saturn is renowned for its icy ring system.",
            date="2026-07",
            tags=("planets", "jupiter", "mars", "saturn", "solar system"),
        ),
        # Tech Industry, Web & Security
        WebDocument(
            title="HTTP/3, QUIC & Modern Web Performance",
            url="https://developer.mozilla.org/en-US/docs/Glossary/HTTP_3",
            snippet="HTTP/3 runs over the UDP-based QUIC transport protocol with built-in TLS 1.3 encryption, eliminating TCP head-of-line blocking and reducing web connection latency.",
            date="2026-04",
            tags=("http3", "quic", "networking", "web"),
        ),
        WebDocument(
            title="OAuth 2.0 & JWT Authentication Best Practices",
            url="https://oauth.net/2/",
            snippet="OAuth 2.0 delegates authorization using short-lived access tokens and PKCE for public clients, while signed JWTs verify stateless API claims without exposing user passwords.",
            date="2026-05",
            tags=("oauth", "jwt", "security", "authentication"),
        ),
        WebDocument(
            title="History of the World Wide Web (CERN, 1989)",
            url="https://home.cern/science/computing/birth-web",
            snippet="British computer scientist Tim Berners-Lee invented the World Wide Web in 1989 at CERN, creating HTTP, HTML, and the first web browser to help researchers share documents globally.",
            date="2026-01",
            tags=("web", "tim berners-lee", "cern", "history"),
        ),
        WebDocument(
            title="NVIDIA Blackwell & Hopper GPU Architecture for AI",
            url="https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/",
            snippet="NVIDIA Blackwell GPUs introduce 5th-generation Tensor Cores with native FP4/FP8 precision and NVLink 5 interconnects, delivering major speedups for trillion-parameter LLM training and inference.",
            date="2026-03",
            tags=("nvidia", "blackwell", "gpu", "fp8", "fp4", "hardware"),
        ),
    ]


class BM25LocalWebIndex:
    """Genuine BM25 full-text search index over WebDocuments (fast, deterministic, offline-safe)."""

    def __init__(self, docs: list[WebDocument] | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs if docs is not None else build_default_web_corpus()
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = []
        self._doc_tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()

        for d in self.docs:
            # Title and tags carry higher natural weight by including them in indexed text
            text = f"{d.title} {d.title} {' '.join(d.tags)} {d.snippet}"
            toks = _tokenize_search(text)
            self._doc_tokens.append(toks)
            tf = Counter(toks)
            self._doc_tf.append(tf)
            for term in tf:
                self._df[term] += 1

        self._avg_dl = sum(len(t) for t in self._doc_tokens) / max(1, len(self._doc_tokens))

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        q_tokens = _tokenize_search(query)
        if not q_tokens:
            return []
        n_docs = len(self.docs)
        scored: list[tuple[float, int]] = []
        for idx, (doc, tf) in enumerate(zip(self.docs, self._doc_tf)):
            dl = len(self._doc_tokens[idx])
            score = 0.0
            for term in set(q_tokens):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                df = self._df[term]
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / max(1.0, self._avg_dl)))
                score += idf * (freq * (self.k1 + 1.0)) / denom
            if score > 0.0:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            d = self.docs[idx]
            results.append(
                SearchResult(
                    id=rank,
                    title=d.title,
                    url=d.url,
                    snippet=d.snippet,
                    date=d.date,
                    domain=d.domain,
                    score=round(score, 4),
                )
            )
        return results


class LiveHTTPWebBackend:
    """Production live web-search adapter (Tavily / Serper / Brave / GitHub Repo Search).

    Automatically activates when an API key env var is present (`TAVILY_API_KEY`,
    `SERPER_API_KEY`, `BRAVE_API_KEY`, or `NOVA_ENABLE_GITHUB_SEARCH=1`).
    """

    def __init__(self, timeout: float = 8.0) -> None:
        self.timeout = timeout

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            return self._search_tavily(query, tavily_key, top_k)

        serper_key = os.getenv("SERPER_API_KEY")
        if serper_key:
            return self._search_serper(query, serper_key, top_k)

        brave_key = os.getenv("BRAVE_API_KEY")
        if brave_key:
            return self._search_brave(query, brave_key, top_k)

        if os.getenv("NOVA_ENABLE_GITHUB_SEARCH") == "1":
            return self._search_github(query, top_k)

        return []

    def _search_tavily(self, query: str, api_key: str, top_k: int) -> list[SearchResult]:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": top_k},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        out = []
        for i, item in enumerate(resp.json().get("results", [])[:top_k], start=1):
            url = item.get("url", "")
            out.append(
                SearchResult(
                    id=i,
                    title=item.get("title", "Web Result"),
                    url=url,
                    snippet=item.get("content", "")[:400],
                    date=item.get("published_date", "live") or "live",
                    domain=urlparse(url).netloc or "web",
                    score=float(item.get("score", 1.0)),
                )
            )
        return out

    def _search_serper(self, query: str, api_key: str, top_k: int) -> list[SearchResult]:
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": top_k},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        out = []
        for i, item in enumerate(resp.json().get("organic", [])[:top_k], start=1):
            url = item.get("link", "")
            out.append(
                SearchResult(
                    id=i,
                    title=item.get("title", "Web Result"),
                    url=url,
                    snippet=item.get("snippet", "")[:400],
                    date=item.get("date", "live") or "live",
                    domain=urlparse(url).netloc or "web",
                    score=1.0 / i,
                )
            )
        return out

    def _search_brave(self, query: str, api_key: str, top_k: int) -> list[SearchResult]:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            params={"q": query, "count": top_k},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        items = resp.json().get("web", {}).get("results", [])[:top_k]
        out = []
        for i, item in enumerate(items, start=1):
            url = item.get("url", "")
            out.append(
                SearchResult(
                    id=i,
                    title=item.get("title", "Web Result"),
                    url=url,
                    snippet=item.get("description", "")[:400],
                    date=item.get("age", "live") or "live",
                    domain=urlparse(url).netloc or "web",
                    score=1.0 / i,
                )
            )
        return out

    def _search_github(self, query: str, top_k: int) -> list[SearchResult]:
        resp = httpx.get(
            "https://api.github.com/search/repositories",
            headers={"Accept": "application/vnd.github+json"},
            params={"q": query, "per_page": top_k},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        out = []
        for i, item in enumerate(resp.json().get("items", [])[:top_k], start=1):
            url = item.get("html_url", "")
            desc = item.get("description") or "No description provided."
            stars = item.get("stargazers_count", 0)
            out.append(
                SearchResult(
                    id=i,
                    title=f"{item.get('full_name', 'repo')} ({stars} stars)",
                    url=url,
                    snippet=desc[:350],
                    date=(item.get("updated_at") or "live")[:10],
                    domain="github.com",
                    score=1.0 / i,
                )
            )
        return out


class HybridSearchBackend:
    """Tries live HTTP search provider when configured, seamlessly falling back to BM25 index."""

    def __init__(
        self,
        local_index: BM25LocalWebIndex | None = None,
        live_backend: LiveHTTPWebBackend | None = None,
    ) -> None:
        self.local = local_index or BM25LocalWebIndex()
        self.live = live_backend or LiveHTTPWebBackend()

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        try:
            live_results = self.live.search(query, top_k=top_k)
            if live_results:
                return live_results
        except Exception:
            pass
        return self.local.search(query, top_k=top_k)


def format_search_results(results: list[SearchResult]) -> str:
    """Format ranked SearchResults into the compact citation-indexed `<|tool_result|>` payload."""
    if not results:
        return "No relevant web search results found."
    lines = []
    for r in results:
        lines.append(f"[{r.id}] {r.title} ({r.domain}, {r.date}): {r.snippet}")
    return "\n".join(lines)
