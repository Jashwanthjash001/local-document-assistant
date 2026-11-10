"""
Stage 3 tests — chunking, embeddings, ChromaDB persistence, and the index endpoint.

Fast tests use fakes (word-based token counter, deterministic embeddings). Real
ChromaDB runs in temp dirs. One real CPU embedding smoke test verifies the model
actually produces 384-dim vectors.
"""
import shutil

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config
from app.services.pdf_processor import ExtractionResult, PageText
from app.services.chunker import chunk_pages, Chunk
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore

client = TestClient(app)

# A word == a token, for fast deterministic chunking tests.
word_counter = lambda s: len(s.split())


def _extraction(pages: list[tuple[int, str]], doc_id="doc1", filename="contract.pdf"):
    return ExtractionResult(
        doc_id=doc_id,
        filename=filename,
        total_pages=len(pages),
        pages=[PageText(page_number=n, text=t) for n, t in pages],
    )


# --- Chunking --------------------------------------------------------------

def test_chunking_multiple_pages_preserves_page_numbers():
    ext = _extraction([(1, "alpha beta gamma"), (2, "delta epsilon zeta")])
    chunks = chunk_pages(ext, word_counter, target_tokens=10, overlap_tokens=2)
    pages = {c.page_number for c in chunks}
    assert pages == {1, 2}
    # chunk_index resets per page
    assert all(c.chunk_index == 0 for c in chunks)


def test_chunking_skips_empty_pages():
    ext = _extraction([(1, "real content here"), (2, "   "), (3, "")])
    chunks = chunk_pages(ext, word_counter, target_tokens=10, overlap_tokens=2)
    assert {c.page_number for c in chunks} == {1}


def test_chunk_overlap_shares_words():
    words = " ".join(f"w{i}" for i in range(12))
    ext = _extraction([(1, words)])
    chunks = chunk_pages(ext, word_counter, target_tokens=5, overlap_tokens=2)
    assert len(chunks) >= 2
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    # last 2 words of chunk 0 reappear at the start of chunk 1
    assert first_words[-2:] == second_words[:2]


def test_chunk_page_provenance_and_ids_are_stable():
    ext = _extraction([(2, "the termination notice period is thirty days")], doc_id="abc")
    chunks = chunk_pages(ext, word_counter, target_tokens=100, overlap_tokens=10)
    assert chunks[0].page_number == 2
    assert chunks[0].id == "abc_p2_c0"


# --- Embedding service -----------------------------------------------------

def test_embedding_empty_input_returns_empty_without_loading_model():
    svc = EmbeddingService(config.EMBEDDING_MODEL, "cpu")
    assert svc.embed([]) == []
    assert svc._model is None   # model was never loaded


@pytest.mark.slow
def test_real_cpu_embedding_has_384_dims():
    svc = EmbeddingService(config.EMBEDDING_MODEL, "cpu")
    vecs = svc.embed(["The termination notice period is 30 days."])
    assert len(vecs) == 1
    assert len(vecs[0]) == config.EMBEDDING_DIM == 384


# --- Vector store (real ChromaDB in temp dir) ------------------------------

def _chunks(doc_id, n, page=1):
    return [
        Chunk(doc_id=doc_id, filename="f.pdf", page_number=page, chunk_index=i, text=f"chunk {i}")
        for i in range(n)
    ]


def _embeds(n, dim=8):
    return [[float(i + 1)] * dim for i in range(n)]


def test_vector_store_add_count_and_metadata(tmp_path):
    vs = VectorStore(tmp_path / "db", "documents")
    chunks = _chunks("docA", 3)
    vs.add_chunks(chunks, _embeds(3))
    assert vs.count_for_document("docA") == 3
    # metadata round-trips correctly
    res = vs._collection.get(where={"doc_id": "docA"}, include=["metadatas"])
    metas = res["metadatas"]
    assert all(m["doc_id"] == "docA" for m in metas)
    assert {m["chunk_index"] for m in metas} == {0, 1, 2}
    assert all(m["page_number"] == 1 for m in metas)


def test_vector_store_repeated_add_no_duplicates(tmp_path):
    vs = VectorStore(tmp_path / "db", "documents")
    chunks = _chunks("docA", 3)
    vs.add_chunks(chunks, _embeds(3))
    vs.add_chunks(chunks, _embeds(3))   # same ids -> upsert, not duplicate
    assert vs.count_for_document("docA") == 3


def test_vector_store_document_specific_deletion(tmp_path):
    vs = VectorStore(tmp_path / "db", "documents")
    vs.add_chunks(_chunks("docA", 2), _embeds(2))
    vs.add_chunks(_chunks("docB", 3), _embeds(3))
    vs.delete_document("docA")
    assert vs.count_for_document("docA") == 0
    assert vs.count_for_document("docB") == 3


def test_vector_store_persists_across_restart(tmp_path):
    db = tmp_path / "db"
    vs1 = VectorStore(db, "documents")
    vs1.add_chunks(_chunks("docA", 4), _embeds(4))
    del vs1
    # A brand-new client pointed at the same dir sees the data.
    vs2 = VectorStore(db, "documents")
    assert vs2.count_for_document("docA") == 4


# --- Full upload -> index workflow (fake embedder, real Chroma) ------------

class _FakeEmbedder:
    """Deterministic, fast stand-in so the workflow test needs no model."""
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 2.0, 3.0] for t in texts]


def _make_pdf_bytes(make_pdf, pages):
    return make_pdf(pages).read_bytes()


def test_upload_then_index_workflow(make_pdf, monkeypatch):
    import app.api.upload as upload_mod
    monkeypatch.setattr(upload_mod, "get_embedding_service", lambda: _FakeEmbedder())

    pdf_bytes = _make_pdf_bytes(make_pdf, ["Clause one content", "Termination notice is 30 days"])
    up = client.post("/documents/upload", files={"file": ("c.pdf", pdf_bytes, "application/pdf")})
    doc_id = up.json()["doc_id"]

    r = client.post(f"/documents/{doc_id}/index")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "indexed"
    assert body["total_pages"] == 2
    assert body["total_chunks"] >= 2
    assert body["embedding_dimensions"] == 384

    # Re-indexing must not duplicate chunks.
    r2 = client.post(f"/documents/{doc_id}/index")
    assert r2.json()["total_chunks"] == body["total_chunks"]
