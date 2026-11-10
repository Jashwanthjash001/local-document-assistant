"""
PDF processor — Stage 2.

Pure extraction logic, deliberately free of any FastAPI / HTTP concerns so it can
be unit-tested on its own and reused by other callers later (e.g. the chunker in
Stage 2.5 / embedding pipeline in Stage 3).

We use PyMuPDF (imported as `pymupdf`) to open a PDF and pull text out one page at
a time, **preserving the page number** — that per-page structure is what makes
source citations possible later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF (the old `import fitz` alias is deprecated)


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be opened or read."""


@dataclass
class PageText:
    """Text extracted from a single page."""
    page_number: int   # 1-based, matching what a human sees in a PDF viewer
    text: str


@dataclass
class ExtractionResult:
    """Structured result of extracting an entire document."""
    doc_id: str
    filename: str
    total_pages: int
    pages: list[PageText]


def extract_pdf(path: Path | str, doc_id: str, filename: str | None = None) -> ExtractionResult:
    """Open a stored PDF and extract its text page by page.

    Args:
        path: filesystem path to the stored PDF.
        doc_id: the document's id (echoed into the result).
        filename: display name; defaults to the file's own name on disk.

    Returns:
        ExtractionResult with one PageText per page, page numbers starting at 1.

    Raises:
        PDFProcessingError: if the file is missing, not a PDF, or unreadable.
    """
    path = Path(path)
    if not path.exists():
        raise PDFProcessingError(f"File not found: {path}")

    # Opening can fail on a corrupt / non-PDF file — wrap it so callers get a
    # single, predictable exception type instead of a raw PyMuPDF error.
    try:
        document = pymupdf.open(str(path))
    except Exception as exc:  # noqa: BLE001 - normalize any backend error
        raise PDFProcessingError(f"Could not open PDF '{path.name}': {exc}") from exc

    try:
        pages: list[PageText] = []
        # `enumerate(document, start=1)` walks every page and gives us a 1-based
        # counter at the same time — this little loop is the heart of ingestion.
        for page_number, page in enumerate(document, start=1):
            # An empty/scanned page simply returns "" here — no crash.
            text = page.get_text().strip()
            pages.append(PageText(page_number=page_number, text=text))
        total_pages = document.page_count
    except Exception as exc:  # noqa: BLE001
        raise PDFProcessingError(f"Failed while reading '{path.name}': {exc}") from exc
    finally:
        # Always release the file handle, even if extraction raised.
        document.close()

    return ExtractionResult(
        doc_id=doc_id,
        filename=filename or path.name,
        total_pages=total_pages,
        pages=pages,
    )
