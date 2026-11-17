"""Quantization x RAG quality — does the LLM's quantization level change the
ANSWER, not just its speed?

Runs the real RAG answer path (retrieve -> grounded prompt -> local LLM -> cited
answer, or abstain) over a curated question set, and scores:

  - grounded    : the answer contains the expected fact
  - cited_page  : it cited the correct page
  - abstain_ok  : unanswerable questions are correctly refused
  - latency     : generation time per answer

Point LLM_BASE_URL/LLM_MODEL at one quant, give it a label, run; repeat for the
other quant; the combined table is printed when both result files exist.

    QUANT_LABEL=q4_k_m LLM_BASE_URL=http://localhost:8080/v1 \
      PYTHONPATH=. .venv/bin/python evals/run_rag_quality.py
"""
from __future__ import annotations

import json
import os
import time
import shutil
from pathlib import Path

from app import config
from app.services.pdf_processor import extract_pdf
from app.services.chunker import chunk_pages
from app.services.embedding import get_embedding_service
from app.services.vector_store import VectorStore
from app.services.hybrid import HybridRetriever
from app.services import prompt_builder
from app.services.llm_client import LLMClient

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
RESULTS = ROOT / "results"
EVAL_CHROMA = ROOT / ".chroma_rag"

# Curated answerable questions with acceptable answer substrings (case-insensitive).
EXPECT: dict[str, list[str]] = {
    "sa05": ["90", "ninety"], "sa07": ["30", "thirty"], "sa08": ["1.5"],
    "sa09": ["3 year", "three year"], "sa12": ["delaware"],
    "eh01": ["three", "3 day"], "eh03": ["20", "twenty"], "eh05": ["10", "ten"],
    "eh09": ["12", "twelve"], "eh12": ["two week", "2 week"],
    "pm02": ["40"], "pm04": ["99.97"], "pm07": ["2 year", "two year"],
    "pp03": ["90", "ninety"],
    "h01": ["x-api-key"], "h02": ["25"], "h03": ["60"], "h04": ["e-4096", "4096"],
    "h09": ["27001", "soc 2", "soc2"], "h10": ["aes-256", "aes"],
    "h14": ["3.2.0"], "h15": ["1287"],
}
_DECLINE = ("do not contain", "does not contain", "couldn't find", "could not find",
            "not stated", "no information", "not mentioned", "not provided", "cannot find")


def load_questions() -> tuple[list[dict], list[dict]]:
    gold = [json.loads(l) for l in open(ROOT / "gold.jsonl", encoding="utf-8") if l.strip()]
    answerable = [g for g in gold if g["answerable"] and g["id"] in EXPECT]
    abstain = [g for g in gold if not g["answerable"]]
    return answerable, abstain


def ingest(store, embedder) -> int:
    n = 0
    for pdf in sorted(CORPUS.glob("*.pdf")):
        res = extract_pdf(pdf, doc_id=pdf.stem, filename=pdf.name)
        chunks = chunk_pages(res, embedder.count_tokens, config.CHUNK_TARGET_TOKENS,
                             config.CHUNK_OVERLAP_TOKENS, config.CHUNK_MAX_TOKENS)
        store.add_chunks(chunks, embedder.embed([c.text for c in chunks]))
        n += len(chunks)
    return n


def answer(retriever, llm, question: str, top_k=5):
    """Mirror the /chat path: retrieve -> gate -> grounded answer (or abstain)."""
    hits = retriever.retrieve(question, top_k=top_k, mode="dense")
    relevant = [h for h in hits if h.distance is not None and h.distance <= config.RELEVANCE_MAX_DISTANCE]
    if not relevant:
        return {"status": "abstain", "text": config.ABSTAIN_MESSAGE, "pages": [], "latency": 0.0}
    sources = prompt_builder.build_sources(relevant)
    msgs = prompt_builder.build_messages(question, relevant, sources)
    t0 = time.monotonic()
    text = llm.complete(msgs)
    latency = time.monotonic() - t0
    cited = prompt_builder.validate_citations(text, sources)
    return {"status": "answered", "text": text, "pages": [s.page_number for s in cited], "latency": latency}


def main() -> None:
    label = os.getenv("QUANT_LABEL", "model")
    if EVAL_CHROMA.exists():
        shutil.rmtree(EVAL_CHROMA)
    answerable, abstain = load_questions()
    embedder = get_embedding_service()
    store = VectorStore(EVAL_CHROMA, "rag")
    ingest(store, embedder)
    retriever = HybridRetriever(embedder, store)
    llm = LLMClient(config.LLM_BASE_URL, config.LLM_MODEL, config.LLM_API_KEY, config.LLM_TIMEOUT)

    grounded = cited = 0
    latencies = []
    for g in answerable:
        r = answer(retriever, llm, g["question"])
        txt = r["text"].lower()
        if any(sub in txt for sub in EXPECT[g["id"]]):
            grounded += 1
        if g["page"] in r["pages"]:
            cited += 1
        if r["latency"]:
            latencies.append(r["latency"])

    abstain_ok = 0
    for g in abstain:
        r = answer(retriever, llm, g["question"])
        if r["status"] == "abstain" or any(p in r["text"].lower() for p in _DECLINE):
            abstain_ok += 1

    na, nab = len(answerable), len(abstain)
    latencies.sort()
    out = {
        "label": label, "model": config.LLM_MODEL, "base_url": config.LLM_BASE_URL,
        "n_answerable": na, "n_abstain": nab,
        "grounded_accuracy": round(grounded / na, 3),
        "cited_page_accuracy": round(cited / na, 3),
        "abstain_accuracy": round(abstain_ok / nab, 3),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p50_latency_s": round(latencies[len(latencies) // 2], 2) if latencies else None,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"quant_rag_{label}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{label}] grounded {out['grounded_accuracy']:.0%}  "
          f"cited-page {out['cited_page_accuracy']:.0%}  abstain {out['abstain_accuracy']:.0%}  "
          f"avg-latency {out['avg_latency_s']}s")

    # combined table when both quants are present
    files = sorted(RESULTS.glob("quant_rag_*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    if len(rows) >= 2:
        print("\n============ QUANTIZATION x RAG QUALITY ============")
        print(f"  {'quant':<10} {'grounded':>9} {'cite-page':>10} {'abstain':>8} {'avg lat':>8} {'p50':>6}")
        for r in rows:
            print(f"  {r['label']:<10} {r['grounded_accuracy']:>9.0%} {r['cited_page_accuracy']:>10.0%} "
                  f"{r['abstain_accuracy']:>8.0%} {str(r['avg_latency_s'])+'s':>8} {str(r['p50_latency_s'])+'s':>6}")
        (RESULTS / "quant_rag.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
