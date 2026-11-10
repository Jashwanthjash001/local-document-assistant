"""
Vector store — Stage 3.

Thin wrapper over a *persistent* ChromaDB collection. We pass our own embeddings
in explicitly (computed by embedding.py), so Chroma never needs to load its own
embedding model. Cosine distance matches our L2-normalized vectors.

Stored per chunk: the original text (`documents`), the vector (`embeddings`), and
`metadata` (doc_id, filename, page_number, chunk_index) — enough to cite the
source page later.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

from app import config
from app.services.chunker import Chunk


class VectorStore:
    def __init__(self, persist_dir: Path | str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        # embedding_function=None: we always supply embeddings ourselves.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks + their vectors. Stable ids make this idempotent."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    @property
    def distance_metric(self) -> str:
        """The configured distance space, e.g. 'cosine'. Read from the collection."""
        return (self._collection.metadata or {}).get("hnsw:space", "l2")

    def query(
        self,
        embedding: list[float],
        top_k: int,
        doc_id: str | None = None,
    ) -> list[dict]:
        """Nearest-neighbour search. Returns rows ordered best-first (smallest
        distance). `distance` is the collection's raw distance (cosine here);
        lower = more similar. Returns [] on an empty collection / no matches."""
        if self._collection.count() == 0:
            return []
        where = {"doc_id": doc_id} if doc_id else None
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma nests results one level per query embedding; we sent one.
        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        rows: list[dict] = []
        for text, meta, distance in zip(documents, metadatas, distances):
            rows.append(
                {
                    "doc_id": meta.get("doc_id"),
                    "page_number": meta.get("page_number"),
                    "chunk_index": meta.get("chunk_index"),
                    "text": text,
                    "distance": distance,
                }
            )
        return rows

    def count_for_document(self, doc_id: str) -> int:
        res = self._collection.get(where={"doc_id": doc_id}, include=[])
        return len(res["ids"])

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def total_count(self) -> int:
        return self._collection.count()


# --- module singleton ------------------------------------------------------
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(config.CHROMA_DIR, config.CHROMA_COLLECTION)
    return _store


def reset_vector_store() -> None:
    """Drop the cached singleton (used by tests to re-point at a temp dir)."""
    global _store
    _store = None
