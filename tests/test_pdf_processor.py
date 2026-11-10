"""
Stage 2 tests — PDF text extraction (service layer + API endpoint).

All PDFs are generated in temp dirs via the `make_pdf` fixture; nothing touches
real uploads (the autouse `isolate_upload_dir` fixture also redirects UPLOAD_DIR).
"""
import shutil

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config
from app.services.pdf_processor import extract_pdf, PDFProcessingError

client = TestClient(app)


# --- Service-layer tests (extract_pdf directly) ----------------------------

def test_extract_multi_page_pdf(make_pdf):
    path = make_pdf(["Page one text", "Page two text", "Page three text"])
    result = extract_pdf(path, doc_id="doc1")

    assert result.doc_id == "doc1"
    assert result.total_pages == 3
    assert len(result.pages) == 3


def test_page_numbering_starts_at_one_and_is_sequential(make_pdf):
    path = make_pdf(["a", "b", "c", "d"])
    result = extract_pdf(path, doc_id="doc1")
    assert [p.page_number for p in result.pages] == [1, 2, 3, 4]


def test_each_page_has_its_own_text(make_pdf):
    path = make_pdf(["Alpha content", "Bravo content"])
    result = extract_pdf(path, doc_id="doc1")
    assert "Alpha" in result.pages[0].text
    assert "Bravo" in result.pages[1].text
    # Pages are kept distinct, not merged into one blob.
    assert result.pages[0].text != result.pages[1].text


def test_empty_page_does_not_crash(make_pdf):
    path = make_pdf(["Has text", ""])  # second page is intentionally empty
    result = extract_pdf(path, doc_id="doc1")
    assert result.total_pages == 2
    assert result.pages[1].text == ""


def test_invalid_pdf_raises(tmp_path):
    bad = tmp_path / "not.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    with pytest.raises(PDFProcessingError):
        extract_pdf(bad, doc_id="doc1")


def test_missing_file_raises(tmp_path):
    with pytest.raises(PDFProcessingError):
        extract_pdf(tmp_path / "nope.pdf", doc_id="doc1")


# --- API-layer tests (POST /documents/{doc_id}/process) --------------------

def _store_pdf_as(doc_id: str, src_path) -> None:
    """Copy a generated PDF into the (temp) UPLOAD_DIR under a given doc_id."""
    shutil.copy(src_path, config.UPLOAD_DIR / f"{doc_id}.pdf")


def test_process_endpoint_returns_structured_pages(make_pdf):
    doc_id = "a" * 32  # valid 32-hex-char id
    _store_pdf_as(doc_id, make_pdf(["First page", "Second page"]))

    r = client.post(f"/documents/{doc_id}/process")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["doc_id"] == doc_id
    assert body["total_pages"] == 2
    assert [p["page_number"] for p in body["pages"]] == [1, 2]
    assert "First" in body["pages"][0]["text"]
    assert "Second" in body["pages"][1]["text"]


def test_process_missing_document_returns_404():
    r = client.post(f"/documents/{'b' * 32}/process")
    assert r.status_code == 404


def test_process_invalid_doc_id_returns_400():
    r = client.post("/documents/not-a-valid-id/process")
    assert r.status_code == 400


def test_process_corrupt_pdf_returns_422():
    doc_id = "c" * 32
    (config.UPLOAD_DIR / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4 but broken contents")
    r = client.post(f"/documents/{doc_id}/process")
    assert r.status_code == 422


def test_upload_then_process_roundtrip(make_pdf):
    # Upload through the real endpoint, then process the returned doc_id.
    pdf_bytes = make_pdf(["Clause 1", "Termination notice is 30 days"]).read_bytes()
    up = client.post(
        "/documents/upload",
        files={"file": ("contract.pdf", pdf_bytes, "application/pdf")},
    )
    assert up.status_code == 200, up.text
    doc_id = up.json()["doc_id"]

    proc = client.post(f"/documents/{doc_id}/process")
    assert proc.status_code == 200, proc.text
    text_all = " ".join(p["text"] for p in proc.json()["pages"])
    assert "Termination notice" in text_all
