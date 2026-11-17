"""Extraction abstraction — basic path, and docling-mode graceful fallback."""
from __future__ import annotations

import pymupdf

from app.services.extraction import extract


def _make_pdf(path, text):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 90), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def test_basic_extraction_reads_text(tmp_path):
    p = tmp_path / "d.pdf"
    _make_pdf(p, "HELLO\nworld")
    r = extract(p, doc_id="x", mode="basic")
    assert r.total_pages == 1
    assert "hello" in r.pages[0].text.lower()
    assert r.pages[0].page_number == 1


def test_docling_mode_falls_back_when_unavailable(tmp_path):
    # docling isn't installed in this venv, so docling mode must fall back to basic
    # (not crash) and still return the page text.
    p = tmp_path / "d.pdf"
    _make_pdf(p, "FALLBACK OK")
    r = extract(p, doc_id="x", mode="docling")
    assert r.total_pages == 1
    assert "fallback ok" in r.pages[0].text.lower()
