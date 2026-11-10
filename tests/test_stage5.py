"""
Stage 5 tests — grounded RAG answering.

The LLM is mocked for all fast tests (deterministic, no GPU). One integration
test hits the real local llama.cpp server and SKIPS automatically if it isn't up.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config
from app.services.retriever import RetrievedChunk
from app.services import prompt_builder, llm_client
from app.services.llm_client import (
    LLMClient, LLMConnectionError, LLMTimeoutError, LLMResponseError,
)

client = TestClient(app)


def _rc(doc_id, page, idx, text, distance):
    return RetrievedChunk(doc_id=doc_id, page_number=page, chunk_index=idx, text=text, distance=distance)


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_doc_id = "unset"

    def retrieve(self, query, top_k=5, doc_id=None):
        self.last_doc_id = doc_id
        rows = [c for c in self._chunks if doc_id is None or c.doc_id == doc_id]
        return rows[:top_k]


class FakeLLM:
    def __init__(self, answer="", raises=None):
        self.answer = answer
        self.raises = raises
        self.called = False
        self.last_messages = None

    def complete(self, messages, temperature=None, max_tokens=None):
        self.called = True
        self.last_messages = messages
        if self.raises:
            raise self.raises
        return self.answer


def _patch(monkeypatch, retriever, llm):
    import app.api.chat as chat_mod
    monkeypatch.setattr(chat_mod, "get_retriever", lambda: retriever)
    monkeypatch.setattr(chat_mod, "get_llm_client", lambda: llm)


# --- Prompt construction ---------------------------------------------------

def test_prompt_includes_question_and_passages():
    chunks = [_rc("d", 2, 0, "The termination notice period is 30 days.", 0.2)]
    sources = prompt_builder.build_sources(chunks)
    msgs = prompt_builder.build_messages("How much notice?", chunks, sources)
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "How much notice?" in user
    assert "termination notice period is 30 days" in user
    assert "[Source 1]" in user


def test_system_prompt_has_injection_and_grounding_rules():
    assert "untrusted" in prompt_builder.SYSTEM_PROMPT.lower()
    assert "ignore" in prompt_builder.SYSTEM_PROMPT.lower()
    assert "insufficient" in prompt_builder.SYSTEM_PROMPT.lower()


def test_source_identifiers_preserve_provenance():
    chunks = [_rc("abc", 2, 0, "x", 0.1), _rc("abc", 5, 1, "y", 0.2)]
    sources = prompt_builder.build_sources(chunks)
    assert sources[0].source_id == "Source 1"
    assert sources[1].page_number == 5
    assert sources[1].chunk_index == 1


# --- Citation validation ---------------------------------------------------

def test_citation_validation_drops_fabricated_ids():
    sources = prompt_builder.build_sources([_rc("d", 1, 0, "x", 0.1)])  # only Source 1
    # Model cites a real one and a fabricated one.
    kept = prompt_builder.validate_citations("Answer [Source 1] and [Source 9].", sources)
    ids = {s.source_id for s in kept}
    assert ids == {"Source 1"}


def test_citation_validation_falls_back_when_none_cited():
    sources = prompt_builder.build_sources([_rc("d", 1, 0, "x", 0.1)])
    kept = prompt_builder.validate_citations("A plain answer with no markers.", sources)
    assert len(kept) == 1  # evidence still reported


# --- LLM client (httpx mocked) --------------------------------------------

class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _make_client():
    return LLMClient("http://localhost:8080/v1", "m", "k", 5)


def test_llm_success(monkeypatch):
    payload = {"choices": [{"message": {"content": "hello"}}]}
    monkeypatch.setattr(llm_client.httpx, "post", lambda *a, **k: _Resp(200, payload))
    assert _make_client().complete([{"role": "user", "content": "hi"}]) == "hello"


def test_llm_connection_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(llm_client.httpx, "post", boom)
    with pytest.raises(LLMConnectionError):
        _make_client().complete([])


def test_llm_timeout(monkeypatch):
    def boom(*a, **k):
        raise httpx.TimeoutException("slow")
    monkeypatch.setattr(llm_client.httpx, "post", boom)
    with pytest.raises(LLMTimeoutError):
        _make_client().complete([])


def test_llm_http_error(monkeypatch):
    monkeypatch.setattr(llm_client.httpx, "post", lambda *a, **k: _Resp(500, text="boom"))
    with pytest.raises(LLMResponseError):
        _make_client().complete([])


def test_llm_invalid_shape(monkeypatch):
    monkeypatch.setattr(llm_client.httpx, "post", lambda *a, **k: _Resp(200, {"nope": 1}))
    with pytest.raises(LLMResponseError):
        _make_client().complete([])


# --- /chat endpoint --------------------------------------------------------

def test_chat_answers_with_sources(monkeypatch):
    retriever = FakeRetriever([_rc("d", 2, 0, "The notice period is 30 days.", 0.2)])
    llm = FakeLLM(answer="30 days' notice is required. [Source 1]")
    _patch(monkeypatch, retriever, llm)

    r = client.post("/chat", json={"query": "How much notice?", "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered"
    assert "[Source 1]" in body["answer"]
    assert body["sources"][0]["page_number"] == 2
    assert llm.called is True


def test_chat_abstains_when_no_relevant_chunk(monkeypatch):
    # distance 0.95 is above the default threshold (0.6) -> abstain, no LLM call.
    retriever = FakeRetriever([_rc("d", 1, 0, "unrelated text", 0.95)])
    llm = FakeLLM(answer="SHOULD NOT BE CALLED")
    _patch(monkeypatch, retriever, llm)

    r = client.post("/chat", json={"query": "What is the budget?"})
    body = r.json()
    assert body["status"] == "insufficient_evidence"
    assert body["answer"] == config.ABSTAIN_MESSAGE
    assert body["sources"] == []
    assert llm.called is False   # the LLM was never called


def test_chat_abstains_on_empty_retrieval(monkeypatch):
    _patch(monkeypatch, FakeRetriever([]), FakeLLM(answer="x"))
    r = client.post("/chat", json={"query": "anything"})
    assert r.json()["status"] == "insufficient_evidence"


def test_chat_relevance_threshold_boundary(monkeypatch):
    # Just inside the threshold -> answered.
    retriever = FakeRetriever([_rc("d", 1, 0, "relevant", config.RELEVANCE_MAX_DISTANCE - 0.01)])
    llm = FakeLLM(answer="answer [Source 1]")
    _patch(monkeypatch, retriever, llm)
    assert client.post("/chat", json={"query": "q"}).json()["status"] == "answered"


def test_chat_filters_by_doc_id(monkeypatch):
    retriever = FakeRetriever([
        _rc("docA", 1, 0, "A content", 0.2),
        _rc("docB", 1, 0, "B content", 0.2),
    ])
    llm = FakeLLM(answer="from A [Source 1]")
    _patch(monkeypatch, retriever, llm)
    r = client.post("/chat", json={"query": "q", "doc_id": "docA"})
    assert retriever.last_doc_id == "docA"
    assert r.json()["sources"][0]["doc_id"] == "docA"


def test_chat_llm_connection_failure_returns_503(monkeypatch):
    retriever = FakeRetriever([_rc("d", 1, 0, "relevant", 0.2)])
    llm = FakeLLM(raises=LLMConnectionError("down"))
    _patch(monkeypatch, retriever, llm)
    r = client.post("/chat", json={"query": "q"})
    assert r.status_code == 503


def test_chat_llm_timeout_returns_503(monkeypatch):
    retriever = FakeRetriever([_rc("d", 1, 0, "relevant", 0.2)])
    llm = FakeLLM(raises=LLMTimeoutError("slow"))
    _patch(monkeypatch, retriever, llm)
    assert client.post("/chat", json={"query": "q"}).status_code == 503


def test_chat_empty_query_returns_422():
    assert client.post("/chat", json={"query": "   "}).status_code == 422


# --- Real local-model integration (auto-skips if server is down) ----------

@pytest.mark.integration
def test_real_local_model_grounded_answer(tmp_path, monkeypatch):
    # Skip unless the llama.cpp server is actually serving.
    try:
        httpx.get(f"{config.LLM_BASE_URL}/models", timeout=2)
    except Exception:
        pytest.skip("local LLM server not running at " + config.LLM_BASE_URL)

    from app.services.embedding import EmbeddingService
    from app.services.vector_store import VectorStore
    from app.services.retriever import Retriever
    from app.services.chunker import Chunk

    emb = EmbeddingService(config.EMBEDDING_MODEL, "cpu")
    store = VectorStore(tmp_path / "db", "documents")
    chunk = Chunk("doc", "f.pdf", 2, 0, "The agreement requires 30 days' notice before termination.")
    store.add_chunks([chunk], emb.embed([chunk.text]))
    monkeypatch.setattr("app.api.chat.get_retriever", lambda: Retriever(emb, store))

    r = client.post("/chat", json={"query": "How much notice before termination?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered"
    assert "30" in body["answer"]   # grounded in the passage
