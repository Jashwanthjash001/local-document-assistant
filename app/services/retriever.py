"""
Retriever — Stage 4.

Turns a natural-language query into ranked document chunks:

    query -> query embedding -> ChromaDB similarity search -> ranked chunks

Deliberately free of FastAPI so it can be unit-tested directly. It does NOT call
the LLM and does NOT generate an answer — that is Stage 5.

Scoring note: the vector store is configured for **cosine distance**. Results carry
the raw `distance` (lower = more similar); we do not fabricate a similarity %.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.embedding import get_embedding_service
from app.services.vector_store import get_vector_store


@dataclass
class RetrievedChunk:
    doc_id: str
    page_number: int
    chunk_index: int
    text: str
    distance: float   # raw ChromaDB distance (cosine here); lower = closer


class Retriever:
    def __init__(self, embedder, store) -> None:
        # Dependencies are injected so tests can pass fakes / isolated stores.
        self._embedder = embedder
        self._store = store

    @property
    def distance_metric(self) -> str:
        return self._store.distance_metric

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        # Guard first: an empty/whitespace query must never reach the model.
        if query is None or not query.strip():
            raise ValueError("Query must not be empty or whitespace.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_vector = self._embedder.embed([query])[0]
        rows = self._store.query(query_vector, top_k=top_k, doc_id=doc_id)
        return [
            RetrievedChunk(
                doc_id=row["doc_id"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                distance=row["distance"],
            )
            for row in rows
        ]


def get_retriever() -> Retriever:
    """Wire the retriever to the shared embedding service + persistent store."""
    return Retriever(get_embedding_service(), get_vector_store())
