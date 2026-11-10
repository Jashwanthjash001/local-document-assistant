"""
Upload API — Stage 1.

Receives a PDF via multipart/form-data, validates it, stores it on disk under a
unique document id, and returns metadata. Text extraction / chunking / embedding
come in later stages — here we only prove the file arrives and is persisted.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app import config
from app.services.pdf_processor import extract_pdf, PDFProcessingError
from app.services.chunker import chunk_pages
from app.services.embedding import get_embedding_service
from app.services.vector_store import get_vector_store

# An APIRouter is a mini-app: we define routes here and "include" it in main.py.
# prefix="/documents" means every path below starts with /documents.
router = APIRouter(prefix="/documents", tags=["documents"])

# doc_ids we mint are uuid4 hex (32 lowercase hex chars). Validating the path
# parameter against this prevents path-traversal (e.g. "../../etc/passwd").
_DOC_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _resolve_stored_pdf(doc_id: str) -> Path:
    """Validate a doc_id and return the path to its stored PDF (must exist)."""
    if not _DOC_ID_RE.match(doc_id):
        raise HTTPException(status_code=400, detail="Invalid document id format.")
    stored_path = config.UPLOAD_DIR / f"{doc_id}.pdf"
    if not stored_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return stored_path


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accept a single PDF and store it.

    `UploadFile = File(...)` tells FastAPI to read a file field named `file` from
    the multipart form. The `...` (Ellipsis) marks it as required.
    """
    # 1) Validate the extension.
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}",
        )

    # 2) Read the bytes and enforce a size cap (read once, reuse).
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)} bytes). Max is {config.MAX_UPLOAD_BYTES}.",
        )

    # 3) Sanity-check it really looks like a PDF (magic bytes).
    if not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not look like a valid PDF.")

    # 4) Store under a generated document id so filenames can't collide or escape.
    doc_id = uuid.uuid4().hex
    stored_path = config.UPLOAD_DIR / f"{doc_id}.pdf"
    stored_path.write_bytes(contents)

    # 5) Return metadata (this is what the frontend will track per document).
    return {
        "doc_id": doc_id,
        "original_filename": file.filename,
        "size_bytes": len(contents),
        "stored_path": config.rel_to_base(stored_path),
        "status": "stored",
        "note": "Uploaded. Call POST /documents/{doc_id}/process to extract text.",
    }


@router.post("/{doc_id}/process")
async def process_document(doc_id: str):
    """Extract text from a previously uploaded PDF, page by page.

    This is a separate step from upload on purpose: storing a file and
    *understanding* its contents are different responsibilities.
    """
    stored_path = _resolve_stored_pdf(doc_id)

    try:
        result = extract_pdf(stored_path, doc_id=doc_id)
    except PDFProcessingError as exc:
        # 422: the request was well-formed but the document couldn't be processed.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # FastAPI serializes the ExtractionResult dataclass (and its nested PageText
    # list) to JSON automatically.
    return result


@router.post("/{doc_id}/index")
async def index_document(doc_id: str):
    """Extract -> chunk -> embed -> persist a document's chunks in ChromaDB.

    Idempotent: re-indexing the same document replaces its chunks instead of
    duplicating them (we delete the doc's existing chunks first, and chunk ids
    are stable anyway).
    """
    stored_path = _resolve_stored_pdf(doc_id)

    # 1) Extract text (reuses the Stage 2 service).
    try:
        extraction = extract_pdf(stored_path, doc_id=doc_id)
    except PDFProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    embedder = get_embedding_service()
    store = get_vector_store()

    # 2) Chunk, using the model's own tokenizer for accurate token counts.
    chunks = chunk_pages(
        extraction,
        count_tokens=embedder.count_tokens,
        target_tokens=config.CHUNK_TARGET_TOKENS,
        overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        max_tokens=config.CHUNK_MAX_TOKENS,
    )

    # 3) Replace any previous index for this document (idempotency).
    store.delete_document(doc_id)

    # 4) Embed + persist.
    if chunks:
        vectors = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, vectors)

    return {
        "doc_id": doc_id,
        "status": "indexed",
        "total_pages": extraction.total_pages,
        "total_chunks": len(chunks),
        "embedding_dimensions": config.EMBEDDING_DIM,
    }
