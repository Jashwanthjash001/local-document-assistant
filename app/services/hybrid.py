"""Hybrid retrieval — dense (vector) + BM25 (keyword), fused with Reciprocal
Rank Fusion. Dense is great at paraphrase/meaning; BM25 is great at exact terms,
codes, and identifiers. RRF combines their rankings without tuning weights.

Kept simple on purpose: for the modest corpora this app handles we rank the whole
set in memory, so there's nothing to maintain beyond the vectors already in Chroma.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app import config
from app.services.embedding import get_embedding_service
from app.services.vector_store import get_vector_store

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokens that keep identifiers intact (e-4096, x-api-key, /v3/ingest)."""
    return _TOKEN.findall(text.lower())


@dataclass
class Ranked:
    doc_id: str
    page_number: int
    chunk_index: int
    text: str
    distance: float | None = None   # cosine distance (dense); lower = closer
    score: float | None = None      # fused/BM25 score; higher = better


def _cosine_distance(a: list[float], b: list[float]) -> float:
    # vectors are L2-normalized in the store, so cosine distance = 1 - dot.
    return 1.0 - sum(x * y for x, y in zip(a, b))


class HybridRetriever:
    def __init__(self, embedder=None, store=None, rrf_k: int | None = None) -> None:
        self._embedder = embedder or get_embedding_service()
        self._store = store or get_vector_store()
        self._rrf_k = config.RRF_K if rrf_k is None else rrf_k
        self._cache: dict[str, tuple[list[dict], object]] = {}

    def _load(self, doc_id: str | None):
        key = doc_id or "__all__"
        if key not in self._cache:
            chunks = self._store.all_chunks_with_vectors(doc_id)
            bm25 = None
            if chunks:
                from rank_bm25 import BM25Okapi
                bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
            self._cache[key] = (chunks, bm25)
        return self._cache[key]

    def retrieve(self, query: str, top_k: int = 5, doc_id: str | None = None,
                 mode: str = "hybrid") -> list[Ranked]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        chunks, bm25 = self._load(doc_id)
        if not chunks:
            return []

        qvec = self._embedder.embed([query])[0]
        dense = sorted(chunks, key=lambda c: _cosine_distance(qvec, c["embedding"]))
        dist = {c["id"]: _cosine_distance(qvec, c["embedding"]) for c in chunks}

        if mode == "dense":
            return [self._row(c, distance=dist[c["id"]]) for c in dense[:top_k]]

        bm_scores = bm25.get_scores(_tokenize(query))
        bm_order = sorted(range(len(chunks)), key=lambda i: -bm_scores[i])
        if mode == "bm25":
            return [self._row(chunks[i], distance=dist[chunks[i]["id"]], score=float(bm_scores[i]))
                    for i in bm_order[:top_k]]

        # hybrid: Reciprocal Rank Fusion of the two rankings, keyed by chunk id.
        rank_d = {c["id"]: r for r, c in enumerate(dense, start=1)}
        rank_b = {chunks[i]["id"]: r for r, i in enumerate(bm_order, start=1)}
        meta = {c["id"]: c for c in chunks}
        fused: dict[str, float] = {}
        for cid in meta:
            s = 0.0
            if cid in rank_d:
                s += 1.0 / (self._rrf_k + rank_d[cid])
            if cid in rank_b:
                s += 1.0 / (self._rrf_k + rank_b[cid])
            fused[cid] = s
        order = sorted(meta, key=lambda c: -fused[c])[:top_k]
        return [self._row(meta[c], distance=dist.get(c), score=fused[c]) for c in order]

    @staticmethod
    def _row(c: dict, distance: float | None = None, score: float | None = None) -> Ranked:
        return Ranked(
            doc_id=c["doc_id"], page_number=c["page_number"],
            chunk_index=c["chunk_index"], text=c["text"],
            distance=distance, score=score,
        )


def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever()
