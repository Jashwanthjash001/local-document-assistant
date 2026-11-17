"""
Central configuration for the Local AI Document Assistant.

Everything that might change between machines or environments lives here, so the
rest of the code never hard-codes a path or a URL. Values can be overridden with
environment variables (useful later for Docker / different machines).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# BASE_DIR = the project root (…/local-document-assistant), computed relative to
# this file so it works no matter where the app is launched from.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"       # raw uploaded PDFs land here
CHROMA_DIR = DATA_DIR / "chroma_db"     # vector store (used from Stage 3)

# Make sure the data directories exist at import time.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- Upload rules ----------------------------------------------------------
ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024     # 50 MB cap for now

# --- Local LLM (OpenAI-compatible: llama.cpp or Ollama) --------------------
# Default points at the llama.cpp server (serving/serve.sh). To use Ollama instead:
#   LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:7b
# This is ALWAYS a local address — the app must never fall back to a cloud model
# or send document content off the machine.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-local")   # llama.cpp ignores it; sent for OpenAI-compat
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))  # deterministic, factual
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# --- Relevance gate (Stage 5) ----------------------------------------------
# Max cosine distance for a retrieved chunk to count as "relevant enough" to
# answer from. Lower distance = more similar. Calibrated against real documents:
# with all-MiniLM-L6-v2, genuinely relevant passages for natural questions land
# ~0.70–0.85 while off-topic queries sit ~0.93+, so 0.88 separates them. The LLM's
# "say if it's not stated" rule is the second backstop. Still a heuristic — tune
# per corpus. If nothing clears it, the assistant abstains w/o calling the LLM.
RELEVANCE_MAX_DISTANCE = float(os.getenv("RELEVANCE_MAX_DISTANCE", "0.88"))
ABSTAIN_MESSAGE = (
    "I couldn't find this information in the uploaded documents."
)

# --- Embeddings & vector store (Stage 3) -----------------------------------
# The embedding model is separate from the LLM: it turns text into vectors for
# semantic search. Runs on CPU so it doesn't compete with LLM inference for the
# Mac's unified memory.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_DIM = 384                      # all-MiniLM-L6-v2 output dimension

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

# Chunking: stay comfortably under the model's 256-token limit so text is never
# silently truncated. ~200 tokens/chunk with 30-token overlap, within each page.
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "200"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "30"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "256"))

# Retrieval mode: "dense" (vector only, the original), "bm25" (keyword only),
# or "hybrid" (dense + BM25 fused with Reciprocal Rank Fusion). Switchable per
# request from the UI; this is the default.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense")
RRF_K = int(os.getenv("RRF_K", "60"))   # RRF damping constant

# Extraction: "basic" (PyMuPDF text, the default) or "docling" (layout + tables +
# OCR for scanned pages; heavier, optional dependency, falls back to basic).
EXTRACTION_MODE = os.getenv("EXTRACTION_MODE", "basic")

# Local vision model for extracting images/figures from a page (a VLM served by
# Ollama, e.g. a qwen2-vl or granite-docling). Empty = image extraction disabled.
VISION_MODEL = os.getenv("VISION_MODEL", "")
VISION_BASE_URL = os.getenv("VISION_BASE_URL", LLM_BASE_URL)

# --- App metadata ----------------------------------------------------------
APP_NAME = "Local AI Document Assistant"
APP_VERSION = "0.3.0"   # Stage 3


# --- Helpers ---------------------------------------------------------------
def rel_to_base(path: Path) -> str:
    """Return `path` relative to the project root for tidy display.

    Falls back to the absolute path if `path` lives outside the project (e.g.
    a temp directory during tests), so this never raises.
    """
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)
