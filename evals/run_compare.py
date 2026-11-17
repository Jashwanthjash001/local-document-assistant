"""Compare retrieval modes — dense vs BM25 vs hybrid — on the gold set, overall
and on the HARD subset (exact-term / identifier / confusable questions, where
keyword matching is expected to help). Pure ranking metrics (recall@k, MRR), so
it's an apples-to-apples comparison independent of the answer gate.

    .venv/bin/python evals/make_corpus.py
    PYTHONPATH=. .venv/bin/python evals/run_compare.py    # writes results/retrieval_compare.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from app import config
from app.services.pdf_processor import extract_pdf
from app.services.chunker import chunk_pages
from app.services.embedding import get_embedding_service
from app.services.vector_store import VectorStore
from app.services.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
RESULTS = ROOT / "results"
EVAL_CHROMA = ROOT / ".chroma_eval"
MODES = ("dense", "bm25", "hybrid")
RECALL_KS = (1, 3, 5)
TOP_K = 10


def load_gold() -> list[dict]:
    with open(ROOT / "gold.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ingest(store: VectorStore, embedder) -> int:
    pdfs = sorted(CORPUS.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No corpus. Run: .venv/bin/python evals/make_corpus.py")
    n = 0
    for pdf in pdfs:
        result = extract_pdf(pdf, doc_id=pdf.stem, filename=pdf.name)
        chunks = chunk_pages(result, embedder.count_tokens,
                             config.CHUNK_TARGET_TOKENS, config.CHUNK_OVERLAP_TOKENS,
                             config.CHUNK_MAX_TOKENS)
        store.add_chunks(chunks, embedder.embed([c.text for c in chunks]))
        n += len(chunks)
    return n


def first_rank(hits, doc, page) -> int | None:
    for i, h in enumerate(hits, start=1):
        if h.doc_id == doc and h.page_number == page:
            return i
    return None


def evaluate(retriever: HybridRetriever, questions: list[dict], mode: str) -> dict:
    recall = {k: 0 for k in RECALL_KS}
    rr = 0.0
    for q in questions:
        hits = retriever.retrieve(q["question"], top_k=TOP_K, mode=mode)
        rank = first_rank(hits, q["doc"], q["page"])
        for k in RECALL_KS:
            if rank is not None and rank <= k:
                recall[k] += 1
        rr += (1.0 / rank) if rank else 0.0
    n = len(questions)
    return {
        "recall": {f"@{k}": round(recall[k] / n, 3) for k in RECALL_KS},
        "mrr": round(rr / n, 3),
        "n": n,
    }


def main() -> None:
    if EVAL_CHROMA.exists():
        shutil.rmtree(EVAL_CHROMA)
    gold = load_gold()
    answerable = [g for g in gold if g["answerable"]]
    hard = [g for g in answerable if g.get("hard")]

    embedder = get_embedding_service()
    store = VectorStore(EVAL_CHROMA, "eval")
    n_chunks = ingest(store, embedder)
    retriever = HybridRetriever(embedder, store)

    out = {"corpus_docs": len(list(CORPUS.glob("*.pdf"))), "chunks": n_chunks,
           "top_k": TOP_K, "all": {}, "hard": {}}
    for mode in MODES:
        out["all"][mode] = evaluate(retriever, answerable, mode)
        out["hard"][mode] = evaluate(retriever, hard, mode)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval_compare.json").write_text(json.dumps(out, indent=2))

    def table(title, block, n):
        print(f"\n{title}  (n={n})")
        print(f"  {'mode':<8} {'recall@1':>9} {'recall@3':>9} {'recall@5':>9} {'MRR':>7}")
        for mode in MODES:
            r = block[mode]
            print(f"  {mode:<8} {r['recall']['@1']:>9.3f} {r['recall']['@3']:>9.3f} "
                  f"{r['recall']['@5']:>9.3f} {r['mrr']:>7.3f}")

    print(f"\n============ RETRIEVAL COMPARISON ============")
    print(f"corpus: {out['corpus_docs']} docs, {n_chunks} chunks")
    table("ALL answerable questions", out["all"], len(answerable))
    table("HARD subset (exact-term / identifier / confusable)", out["hard"], len(hard))
    print("\nwrote evals/results/retrieval_compare.json")


if __name__ == "__main__":
    main()
