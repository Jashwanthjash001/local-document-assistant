"""
Stage 4 tests — retrieval pipeline.

Deterministic unit tests use fixed fake embeddings against a real (temp) ChromaDB
collection, so we exercise genuine ranking without the model. One slow test uses
real CPU embeddings to confirm semantic ranking.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config
from app.services.chunker import Chunk
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever

client = TestClient(app)


class FixedEmbedder:
    """Returns a fixed query vector regardless of input text."""
    def __init__(self, vector):
        self._vector = vector

    def embed(self, texts):
        return [list(self._vector) for _ in texts]


def _chunk(doc_id, page, idx, text):
    return Chunk(doc_id=doc_id, filename="f.pdf", page_number=page, chunk_index=idx, text=text)


def _store_with(tmp_path, rows):
    """rows: list of (chunk, vector). Build a real temp store populated with them."""
    vs = VectorStore(tmp_path / "db", "documents")
    chunks = [c for c, _ in rows]
    vecs = [v for _, v in rows]
    vs.add_chunks(chunks, vecs)
    return vs


# --- Ranking & provenance --------------------------------------------------

def test_relevant_chunk_ranks_ahead(tmp_path):
    store = _store_with(tmp_path, [
        (_chunk("docA", 2, 0, "notice before termination"), [1.0, 0.0, 0.0]),  # close
        (_chunk("docA", 5, 0, "solar plant output in August"), [0.0, 1.0, 0.0]),  # far
    ])
    retriever = Retriever(FixedEmbedder([1.0, 0.0, 0.0]), store)
    results = retriever.retrieve("how much notice to cancel?", top_k=2)
    assert results[0].text == "notice before termination"
    assert results[0].distance <= results[1].distance


def test_provenance_fields_present(tmp_path):
    store = _store_with(tmp_path, [
        (_chunk("abc", 2, 0, "the termination notice period is 30 days"), [1.0, 0.0, 0.0]),
    ])
    r = Retriever(FixedEmbedder([1.0, 0.0, 0.0]), store).retrieve("notice?", top_k=1)[0]
    assert r.doc_id == "abc"
    assert r.page_number == 2
    assert r.chunk_index == 0
    assert isinstance(r.distance, float)


def test_top_k_limits_results(tmp_path):
    store = _store_with(tmp_path, [
        (_chunk("d", 1, i, f"chunk {i}"), [1.0, 0.0, 0.0]) for i in range(5)
    ])
    results = Retriever(FixedEmbedder([1.0, 0.0, 0.0]), store).retrieve("q", top_k=2)
    assert len(results) == 2


def test_filter_by_doc_id(tmp_path):
    store = _store_with(tmp_path, [
        (_chunk("docA", 1, 0, "alpha"), [1.0, 0.0, 0.0]),
        (_chunk("docB", 1, 0, "beta"), [1.0, 0.0, 0.0]),
    ])
    results = Retriever(FixedEmbedder([1.0, 0.0, 0.0]), store).retrieve("q", top_k=5, doc_id="docA")
    assert len(results) == 1
    assert results[0].doc_id == "docA"


# --- Validation & edge cases ----------------------------------------------

def test_empty_query_raises(tmp_path):
    store = _store_with(tmp_path, [(_chunk("d", 1, 0, "x"), [1.0, 0.0, 0.0])])
    r = Retriever(FixedEmbedder([1.0, 0.0, 0.0]), store)
    with pytest.raises(ValueError):
        r.retrieve("", top_k=5)
    with pytest.raises(ValueError):
        r.retrieve("   ", top_k=5)


def test_invalid_top_k_raises(tmp_path):
    store = _store_with(tmp_path, [(_chunk("d", 1, 0, "x"), [1.0, 0.0, 0.0])])
    r = Retriever(FixedEmbedder([1.0, 0.0, 0.0]), store)
    with pytest.raises(ValueError):
        r.retrieve("q", top_k=0)


def test_empty_collection_returns_empty(tmp_path):
    store = VectorStore(tmp_path / "db", "documents")  # nothing added
    results = Retriever(FixedEmbedder([1.0, 0.0, 0.0]), store).retrieve("q", top_k=5)
    assert results == []


def test_retrieval_from_persisted_collection(tmp_path):
    _store_with(tmp_path, [(_chunk("d", 3, 0, "persisted chunk"), [1.0, 0.0, 0.0])])
    # New store object, same dir -> data persisted.
    store2 = VectorStore(tmp_path / "db", "documents")
    results = Retriever(FixedEmbedder([1.0, 0.0, 0.0]), store2).retrieve("q", top_k=5)
    assert len(results) == 1
    assert results[0].text == "persisted chunk"


# --- API layer -------------------------------------------------------------

def test_search_endpoint_returns_structure(tmp_path, monkeypatch):
    import app.api.search as search_mod
    store = _store_with(tmp_path, [
        (_chunk("docA", 2, 0, "the termination notice period is 30 days"), [1.0, 0.0, 0.0]),
    ])
    monkeypatch.setattr(search_mod, "get_retriever",
                        lambda: Retriever(FixedEmbedder([1.0, 0.0, 0.0]), store))
    r = client.post("/search", json={"query": "notice to cancel?", "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query"] == "notice to cancel?"
    assert body["distance_metric"] == "cosine"
    assert body["total_results"] == 1
    assert body["results"][0]["page_number"] == 2
    assert "distance" in body["results"][0]


def test_search_empty_query_returns_422():
    r = client.post("/search", json={"query": "   ", "top_k": 5})
    assert r.status_code == 422


def test_search_invalid_top_k_returns_422():
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 999}).status_code == 422


# --- Real CPU embedding smoke ---------------------------------------------

@pytest.mark.slow
def test_real_retrieval_ranks_semantically(tmp_path):
    from app.services.embedding import EmbeddingService

    emb = EmbeddingService(config.EMBEDDING_MODEL, "cpu")
    relevant = "The agreement requires 30 days' notice before termination."
    irrelevant = "The solar plant generated 12 MW during August."
    chunks = [_chunk("doc", 1, 0, relevant), _chunk("doc", 2, 0, irrelevant)]
    store = VectorStore(tmp_path / "db", "documents")
    store.add_chunks(chunks, emb.embed([c.text for c in chunks]))

    results = Retriever(emb, store).retrieve(
        "How much notice is needed to cancel the agreement?", top_k=2
    )
    assert results[0].text == relevant          # semantics win over keywords
    assert results[0].distance < results[1].distance
