"""
Stage 1 tests — exercise the API skeleton with FastAPI's TestClient.

TestClient runs the app in-process (no server, no network, no GPU), so these are
fast and safe to run anytime.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# Minimal valid-looking PDF bytes: the %PDF magic header is all Stage 1 checks for.
FAKE_PDF = b"%PDF-1.4\n%fake minimal pdf for tests\n"


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_pdf_succeeds():
    r = client.post(
        "/documents/upload",
        files={"file": ("contract.pdf", FAKE_PDF, "application/pdf")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "stored"
    assert body["original_filename"] == "contract.pdf"
    assert body["size_bytes"] == len(FAKE_PDF)
    assert len(body["doc_id"]) == 32  # uuid4 hex


def test_upload_rejects_non_pdf():
    r = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400


def test_upload_rejects_fake_pdf_content():
    # Right extension, wrong magic bytes -> rejected.
    r = client.post(
        "/documents/upload",
        files={"file": ("bad.pdf", b"not really a pdf", "application/pdf")},
    )
    assert r.status_code == 400
