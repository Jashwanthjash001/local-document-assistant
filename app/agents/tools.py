"""Read-only tools the document agent calls based on the request.

Same idea as the @tool registry in a production multi-agent service, scoped to
the local document stack: each tool is a named, described capability, the
orchestrator picks which one to run, and every call is recorded in the response
trace so you can see the agent working in the background. All read-only — nothing
here changes state.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Passage:
    """Duck-types the retriever's result so prompt_builder/the gate accept it."""
    doc_id: str
    page_number: int
    chunk_index: int
    text: str
    distance: float | None = None
    score: float | None = None


# Tool catalogue (name -> what it does) — surfaced so the agent's tool use is legible.
TOOLS = {
    "search_documents": "Hybrid retrieval (BM25 + dense + RRF) — find the most relevant passages.",
    "read_document": "Read a whole document in order — used for extraction/schema-fill so nothing is missed.",
    "list_documents": "List the indexed documents with page and chunk counts.",
}


class Toolbox:
    """Wraps the retriever/vector store as the tools the agent invokes."""

    def __init__(self, retriever) -> None:
        self._r = retriever

    def search_documents(self, query: str, top_k: int = 5, doc_id: str | None = None, mode: str = "hybrid"):
        return self._r.retrieve(query, top_k=top_k, doc_id=doc_id, mode=mode)

    def read_document(self, doc_id: str) -> list[Passage]:
        store = getattr(self._r, "_store", None)
        if store is None:
            return []
        chunks = store.all_chunks_with_vectors(doc_id)
        chunks.sort(key=lambda c: (c["page_number"], c["chunk_index"]))
        return [Passage(c["doc_id"], c["page_number"], c["chunk_index"], c["text"]) for c in chunks]

    def list_documents(self) -> list[dict]:
        store = getattr(self._r, "_store", None)
        if store is None:
            return []
        docs: dict = defaultdict(lambda: {"pages": set(), "chunks": 0})
        for c in store.all_chunks_with_vectors(None):
            docs[c["doc_id"]]["pages"].add(c["page_number"])
            docs[c["doc_id"]]["chunks"] += 1
        return [{"doc_id": d, "pages": len(v["pages"]), "chunks": v["chunks"]} for d, v in docs.items()]
