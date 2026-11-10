"""
Shared pytest fixtures.

The autouse fixture below points the app's UPLOAD_DIR at a fresh temp directory
for *every* test, so tests never read or write real user documents under
`data/uploads/`. Because the API code reads `config.UPLOAD_DIR` dynamically at
request time, patching the attribute is enough to redirect it.
"""
from pathlib import Path

import pytest
import pymupdf

from app import config


@pytest.fixture(autouse=True)
def isolate_upload_dir(tmp_path, monkeypatch):
    """Redirect UPLOAD_DIR and CHROMA_DIR to per-test temp dirs, so tests never
    touch real user uploads or the real vector store."""
    from app.services import vector_store

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    monkeypatch.setattr(config, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(config, "CHROMA_DIR", chroma)
    vector_store.reset_vector_store()   # drop any cached client from a prior test
    yield uploads
    vector_store.reset_vector_store()


@pytest.fixture
def make_pdf(tmp_path):
    """Factory that writes a real multi-page PDF from a list of page strings.

    Returns the path to the created file. An empty string in the list produces a
    genuinely empty page (no text), which is exactly the edge case we want to test.
    """
    def _make(pages: list[str], name: str = "sample.pdf") -> Path:
        out = tmp_path / name
        doc = pymupdf.open()  # new empty PDF
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text, fontsize=12)
        doc.save(out)
        doc.close()
        return out

    return _make
