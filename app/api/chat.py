"""
Chat API — Stage 5: grounded RAG answering.

Pipeline: question -> retrieve -> relevance gate -> prompt -> local LLM -> answer
with citations. Abstains (without calling the LLM) when no retrieved passage is
relevant enough. Returns a real service error if the local LLM is unreachable.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app import config
from app.services.retriever import get_retriever
from app.services.llm_client import get_llm_client, LLMError, LLMTimeoutError, LLMConnectionError
from app.services import prompt_builder

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    doc_id: str | None = None

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty or whitespace")
        return v


def _sources_payload(sources) -> list[dict]:
    return [asdict(s) for s in sources]


@router.post("")
async def chat(request: ChatRequest):
    retriever = get_retriever()
    retrieved = retriever.retrieve(request.query, top_k=request.top_k, doc_id=request.doc_id)

    # --- Relevance gate: only passages within the distance threshold count. ---
    relevant = [c for c in retrieved if c.distance <= config.RELEVANCE_MAX_DISTANCE]

    if not relevant:
        # Abstain WITHOUT calling the LLM — no evidence to ground an answer.
        return {
            "query": request.query,
            "answer": config.ABSTAIN_MESSAGE,
            "status": "insufficient_evidence",
            "sources": [],
            "relevance_threshold": config.RELEVANCE_MAX_DISTANCE,
        }

    # --- Build grounded prompt and call the local LLM. ---
    sources = prompt_builder.build_sources(relevant)
    messages = prompt_builder.build_messages(request.query, relevant, sources)

    try:
        answer = get_llm_client().complete(messages)
    except (LLMTimeoutError, LLMConnectionError) as exc:
        # Local model unreachable/slow: a real service error, not a fake answer.
        raise HTTPException(status_code=503, detail=f"Local LLM unavailable: {exc}") from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"Local LLM error: {exc}") from exc

    # --- Keep only citations that refer to passages actually supplied. ---
    cited_sources = prompt_builder.validate_citations(answer, sources)

    return {
        "query": request.query,
        "answer": answer,
        "status": "answered",
        "sources": _sources_payload(cited_sources),
        "relevance_threshold": config.RELEVANCE_MAX_DISTANCE,
    }
