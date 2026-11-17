"""Agent-layer tests — triage + orchestrator, fully offline (fake LLM, fake
retriever). No GPU, no model, no network."""
from __future__ import annotations

from dataclasses import dataclass

from app import config
from app.agents.triage import triage, _rule_based
from app.agents.orchestrator import run_turn


# --- fakes ----------------------------------------------------------------- #
@dataclass
class FakeHit:
    doc_id: str
    page_number: int
    chunk_index: int
    text: str
    distance: float
    score: float | None = None


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def retrieve(self, query, top_k=5, doc_id=None, mode="hybrid"):
        self.calls.append({"query": query, "mode": mode, "doc_id": doc_id})
        return self._hits[:top_k]


class FakeLLM:
    """Returns a scripted answer; records the messages it was called with."""
    def __init__(self, answer="The notice period is 30 days [Source 1]."):
        self.answer = answer
        self.calls = []

    def complete(self, messages, temperature=None, max_tokens=None):
        self.calls.append(messages)
        return self.answer


def _near(n=2):
    return [FakeHit("service_agreement", 2, 0, "The termination notice period is 30 days.", 0.12)]


# --- triage ---------------------------------------------------------------- #
def test_rule_based_detects_smalltalk():
    out = _rule_based("what can you do?", [])
    assert out["intent"] == "smalltalk" and out["needs_retrieval"] is False


def test_rule_based_detects_summarize():
    out = _rule_based("give me a summary of the contract", [])
    assert out["intent"] == "summarize" and out["needs_retrieval"] is True


def test_rule_based_condenses_followup_with_history():
    history = [{"role": "user", "content": "What is the termination notice period?"},
               {"role": "assistant", "content": "30 days."}]
    out = _rule_based("what about payment?", history)
    # the short follow-up is expanded with the previous user question
    assert "termination notice period" in out["standalone_query"].lower()


def test_triage_without_llm_uses_rules():
    out = triage("hello", [], llm=None)
    assert out["intent"] == "smalltalk"


def test_triage_falls_back_on_bad_llm_json():
    class BadLLM:
        def complete(self, *a, **k):
            return "not json at all"
    out = triage("summarize this", [], llm=BadLLM())
    assert out["intent"] == "summarize"   # fell back to rules


# --- orchestrator ---------------------------------------------------------- #
def test_smalltalk_short_circuits_without_retrieval():
    llm, retr = FakeLLM(), FakeRetriever(_near())
    res = run_turn([{"role": "user", "content": "hi, what can you do?"}], llm=llm, retriever=retr)
    assert res["status"] == "smalltalk"
    assert res["sources"] == [] and retr.calls == []   # never retrieved, never answered


def test_answered_grounds_and_cites():
    llm, retr = FakeLLM(), FakeRetriever(_near())
    res = run_turn([{"role": "user", "content": "What is the notice period?"}],
                   llm=llm, retriever=retr, mode="hybrid")
    assert res["status"] == "answered"
    assert res["sources"] and res["sources"][0]["page_number"] == 2
    assert res["mode"] == "hybrid"
    assert any(s["step"] == "retrieve" for s in res["trace"])


def test_abstains_when_nothing_relevant():
    far = [FakeHit("d", 1, 0, "unrelated", 0.95)]   # beyond RELEVANCE_MAX_DISTANCE
    llm, retr = FakeLLM(), FakeRetriever(far)
    res = run_turn([{"role": "user", "content": "What is the refund policy?"}],
                   llm=llm, retriever=retr)
    assert res["status"] == "insufficient_evidence"
    assert res["answer"] == config.ABSTAIN_MESSAGE
    # triage may consult the LLM, but no ANSWER was generated (abstained first).
    assert len(llm.calls) == 1   # only the triage call, never a grounded-answer call


def test_mode_is_passed_through_to_retriever():
    llm, retr = FakeLLM(), FakeRetriever(_near())
    run_turn([{"role": "user", "content": "notice period?"}], llm=llm, retriever=retr, mode="bm25")
    assert retr.calls[0]["mode"] == "bm25"


# --- image extraction intent ----------------------------------------------- #
def test_rule_based_detects_image_and_page():
    out = _rule_based("can you extract the images on page 3?", [])
    assert out["intent"] == "extract_images"
    assert out["page"] == 3 and out["needs_retrieval"] is False


def test_image_intent_routes_to_vision_and_degrades_cleanly():
    # VISION_MODEL is unset by default -> honest "not configured" message, no crash.
    llm, retr = FakeLLM(), FakeRetriever(_near())
    res = run_turn([{"role": "user", "content": "extract the images on page 2"}],
                   llm=llm, retriever=retr, doc_id="doc1")
    assert res["intent"] == "extract_images" and res["status"] == "images"
    assert "configured" in res["answer"].lower()
    assert any(s["step"] == "extract_images" and s["page"] == 2 for s in res["trace"])
    assert retr.calls == []   # image path never hits document retrieval


def test_image_intent_without_a_document_asks_to_pick_one():
    llm, retr = FakeLLM(), FakeRetriever(_near())
    res = run_turn([{"role": "user", "content": "show me the figures"}],
                   llm=llm, retriever=retr, doc_id=None)
    assert res["status"] == "images" and "document" in res["answer"].lower()


# --- structured / schema-fill extraction ----------------------------------- #
def test_format_request_routes_to_schema_extract():
    llm = FakeLLM(answer='{"name":"Alice Ng","phone":""}')
    retr = FakeRetriever(_near())
    res = run_turn([{"role": "user", "content": 'give me {name:"", phone:""} in this format'}],
                   llm=llm, retriever=retr)
    assert res["intent"] == "extract" and res["status"] == "answered"
    assert "Alice Ng" in res["answer"]


def test_schema_extract_does_not_decline_on_blanks():
    # a schema answer with empty fields must NOT be flipped to insufficient_evidence
    llm = FakeLLM(answer='[{"name":"Bob","phone":""}]  (no information for phone)')
    retr = FakeRetriever(_near())
    res = run_turn([{"role": "user", "content": "list everyone as a table"}],
                   llm=llm, retriever=retr)
    assert res["status"] == "answered" and res["intent"] == "extract"
