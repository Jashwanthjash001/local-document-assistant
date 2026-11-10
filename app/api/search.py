"""
Search API — Stage 4.

POST /search — retrieve the most relevant indexed chunks for a query. Retrieval
only: no LLM, no generated answer (that is Stage 5).
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app.services.retriever import get_retriever

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)   # invalid top_k -> 422 automatically
    doc_id: str | None = None                    # optional: restrict to one document

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        # Reject empty/whitespace at validation time, so it never hits the model.
        if not v or not v.strip():
            raise ValueError("query must not be empty or whitespace")
        return v


@router.post("/search")
async def search(request: SearchRequest):
    retriever = get_retriever()
    results = retriever.retrieve(
        request.query, top_k=request.top_k, doc_id=request.doc_id
    )
    return {
        "query": request.query,
        "distance_metric": retriever.distance_metric,  # 'cosine' — lower distance = more similar
        "results": [asdict(r) for r in results],
        "total_results": len(results),
    }
