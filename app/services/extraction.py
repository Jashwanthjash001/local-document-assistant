"""Pluggable PDF extraction.

- "basic"   : the original PyMuPDF text extractor (fast, no extra deps).
- "docling" : layout-aware extraction with table structure + OCR for scanned
              pages (heavier, optional dependency).

Both return the same ExtractionResult (per-page text with page numbers), so the
chunk -> embed -> retrieve pipeline downstream is unchanged. If docling isn't
installed or a conversion fails, we fall back to basic rather than erroring — the
app keeps working, just without the richer extraction.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from app import config
from app.services.pdf_processor import extract_pdf, ExtractionResult, PageText


def extract(path, doc_id: str, filename: str | None = None, mode: str | None = None) -> ExtractionResult:
    mode = (mode or config.EXTRACTION_MODE or "basic").lower()
    if mode == "docling":
        try:
            return _docling_extract(path, doc_id, filename)
        except Exception as exc:  # optional dep missing / conversion failed
            print(f"[extraction] docling unavailable ({type(exc).__name__}: {exc}); using basic")
    return extract_pdf(path, doc_id=doc_id, filename=filename)


def _docling_extract(path, doc_id: str, filename: str | None) -> ExtractionResult:
    """Best-effort Docling extraction with per-page text + tables as markdown.
    Guarded so any API difference across docling versions falls back to basic."""
    from docling.document_converter import DocumentConverter  # type: ignore

    result = DocumentConverter().convert(str(path))
    doc = result.document

    pages: dict[int, list[str]] = defaultdict(list)

    def _page_of(item) -> int:
        prov = getattr(item, "prov", None)
        if prov:
            return getattr(prov[0], "page_no", 1)
        return 1

    for item in getattr(doc, "texts", []) or []:
        text = getattr(item, "text", "") or ""
        if text.strip():
            pages[_page_of(item)].append(text)

    for table in getattr(doc, "tables", []) or []:
        try:
            pages[_page_of(table)].append(table.export_to_markdown())
        except Exception:
            pass

    if not pages:
        # Nothing page-attributed — put the whole markdown on page 1 so we still
        # index *something* rather than losing the document.
        pages[1].append(doc.export_to_markdown())

    page_texts = [PageText(page_number=p, text="\n\n".join(pages[p])) for p in sorted(pages)]
    return ExtractionResult(
        doc_id=doc_id,
        filename=filename or Path(path).name,
        total_pages=len(page_texts),
        pages=page_texts,
    )
