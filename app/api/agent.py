"""Agent API — the multi-agent, multi-turn endpoint behind the chat UI.

POST /agent with the running conversation; the orchestrator triages the request,
retrieves (dense / bm25 / hybrid — switchable), answers or summarizes grounded in
the documents, or abstains. Returns the answer, citations, and a step trace.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app import config
from app.agents.orchestrator import run_turn
from app.services.llm_client import LLMError, LLMTimeoutError, LLMConnectionError

router = APIRouter(prefix="/agent", tags=["agent"])

_MODES = ("dense", "bm25", "hybrid")


class Message(BaseModel):
    role: str
    content: str


class AgentRequest(BaseModel):
    messages: list[Message]
    mode: str | None = None
    doc_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("messages")
    @classmethod
    def _has_user(cls, v: list[Message]) -> list[Message]:
        if not v or not any(m.role == "user" and m.content.strip() for m in v):
            raise ValueError("messages must include a non-empty user message")
        return v

    @field_validator("mode")
    @classmethod
    def _mode_ok(cls, v: str | None) -> str | None:
        if v is not None and v not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}")
        return v


@router.post("")
async def agent(request: AgentRequest):
    try:
        return run_turn(
            [m.model_dump() for m in request.messages],
            mode=request.mode or config.RETRIEVAL_MODE,
            doc_id=request.doc_id,
            top_k=request.top_k,
        )
    except (LLMTimeoutError, LLMConnectionError) as exc:
        raise HTTPException(status_code=503, detail=f"Local LLM unavailable: {exc}") from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"Local LLM error: {exc}") from exc
