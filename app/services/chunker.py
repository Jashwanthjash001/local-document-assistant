"""
Chunker — Stage 3.

Splits extracted pages into overlapping, token-sized chunks suitable for
embedding. Kept free of heavy ML imports: the token count is *injected* as a
callable, so this module is fast to import and trivial to unit-test. In
production the counter is the embedding model's own tokenizer (see embedding.py);
in tests it can be a simple word counter.

Design rules (from the spec):
- process each page independently; chunks never span page boundaries
- skip empty / whitespace-only pages
- ~target tokens per chunk with ~overlap tokens shared between neighbours
- never exceed the model's max tokens (no silent truncation)
- preserve page numbers and a per-page chunk index
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.services.pdf_processor import ExtractionResult

TokenCounter = Callable[[str], int]


@dataclass
class Chunk:
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int   # per-page, starts at 0
    text: str

    @property
    def id(self) -> str:
        """Stable, unique id: same input -> same id (so re-indexing can't dupe)."""
        return f"{self.doc_id}_p{self.page_number}_c{self.chunk_index}"


def _pack_words(
    words: list[str],
    count_tokens: TokenCounter,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
) -> list[str]:
    """Greedily pack words into ~target-token chunks with ~overlap-token overlap."""
    chunks: list[str] = []
    n = len(words)
    i = 0
    while i < n:
        # Grow the window [i, j) until one more word would exceed the target
        # (but always keep at least one word, and never exceed the hard max).
        j = i
        while j < n:
            candidate = " ".join(words[i : j + 1])
            tokens = count_tokens(candidate)
            if j > i and tokens > target_tokens:
                break
            if tokens > max_tokens and j == i:
                # A single "word" longer than the hard limit: emit it alone
                # rather than silently dropping/truncating it.
                j += 1
                break
            j += 1

        chunks.append(" ".join(words[i:j]))
        if j >= n:
            break

        # Back up the start so the next chunk shares ~overlap_tokens with this one.
        k = j
        while k > i + 1 and count_tokens(" ".join(words[k - 1 : j])) <= overlap_tokens:
            k -= 1
        i = max(k, i + 1)   # guarantee forward progress

    return chunks


def chunk_pages(
    result: ExtractionResult,
    count_tokens: TokenCounter,
    target_tokens: int = 200,
    overlap_tokens: int = 30,
    max_tokens: int = 256,
) -> list[Chunk]:
    """Turn an ExtractionResult into a flat list of Chunks (page order preserved)."""
    chunks: list[Chunk] = []
    for page in result.pages:
        text = page.text.strip()
        if not text:
            continue   # skip empty / whitespace-only pages
        words = text.split()
        for idx, chunk_text in enumerate(
            _pack_words(words, count_tokens, target_tokens, overlap_tokens, max_tokens)
        ):
            chunks.append(
                Chunk(
                    doc_id=result.doc_id,
                    filename=result.filename,
                    page_number=page.page_number,
                    chunk_index=idx,
                    text=chunk_text,
                )
            )
    return chunks
