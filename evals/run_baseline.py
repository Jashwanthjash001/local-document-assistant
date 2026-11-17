"""Retrieval eval baseline — measure the CURRENT dense-only retriever before we
change anything. Reports recall@k, MRR, and abstain accuracy on the gold set.

Ingests the synthetic corpus into an isolated Chroma collection (never touches the
app's own db), then runs retrieval for every gold question.

    .venv/bin/python evals/make_corpus.py      # once, builds evals/corpus/*.pdf
    .venv/bin/python evals/run_baseline.py     # writes evals/results/baseline.json

A retrieved chunk is "relevant" to an answerable question when its doc_id and page
both match the gold label. Abstain is judged with the app's own RELEVANCE_MAX_DISTANCE
gate: a question is "abstained" when no retrieved chunk is within that distance.
"""
from __future__ import annotations

import json
from pathlib import Path

from app import config
from app.services.pdf_processor import extract_pdf
from app.services.chunker import chunk_pages
from app.services.embedding import get_embedding_service
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
RESULTS = ROOT / "results"
EVAL_CHROMA = ROOT / ".chroma_eval"
TOP_K = 10
RECALL_KS = (1, 3, 5, 10)


def load_gold() -> list[dict]:
    with open(ROOT / "gold.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ingest(store: VectorStore, embedder) -> int:
    pdfs = sorted(CORPUS.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No corpus found. Run: .venv/bin/python evals/make_corpus.py")
    n_chunks = 0
    for pdf in pdfs:
        doc_id = pdf.stem
        result = extract_pdf(pdf, doc_id=doc_id, filename=pdf.name)
        chunks = chunk_pages(
            result, embedder.count_tokens,
            config.CHUNK_TARGET_TOKENS, config.CHUNK_OVERLAP_TOKENS, config.CHUNK_MAX_TOKENS,
        )
        vectors = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, vectors)
        n_chunks += len(chunks)
    return n_chunks


def first_relevant_rank(hits: list, doc: str, page: int) -> int | None:
    """1-based rank of the first retrieved chunk matching doc+page, else None."""
    for i, h in enumerate(hits, start=1):
        if h.doc_id == doc and h.page_number == page:
            return i
    return None


def main() -> None:
    # fresh isolated store each run
    import shutil
    if EVAL_CHROMA.exists():
        shutil.rmtree(EVAL_CHROMA)

    gold = load_gold()
    embedder = get_embedding_service()
    store = VectorStore(EVAL_CHROMA, "eval")
    n_chunks = ingest(store, embedder)
    retriever = Retriever(embedder, store)

    thr = config.RELEVANCE_MAX_DISTANCE
    answerable = [g for g in gold if g["answerable"]]
    abstain = [g for g in gold if not g["answerable"]]

    recall = {k: 0 for k in RECALL_KS}
    rr_sum = 0.0
    false_abstain = 0          # answerable, but the gate would (wrongly) abstain
    per_q = []

    for g in answerable:
        hits = retriever.retrieve(g["question"], top_k=TOP_K)
        rank = first_relevant_rank(hits, g["doc"], g["page"])
        for k in RECALL_KS:
            if rank is not None and rank <= k:
                recall[k] += 1
        rr_sum += (1.0 / rank) if rank else 0.0
        top_dist = round(hits[0].distance, 3) if hits else None
        gated = bool(hits) and hits[0].distance <= thr
        if not gated:
            false_abstain += 1
        per_q.append({"id": g["id"], "rank": rank, "top_distance": top_dist, "answered": gated})

    correct_abstain = 0
    for g in abstain:
        hits = retriever.retrieve(g["question"], top_k=TOP_K)
        within = bool(hits) and hits[0].distance <= thr
        if not within:
            correct_abstain += 1
        per_q.append({"id": g["id"], "rank": None,
                      "top_distance": round(hits[0].distance, 3) if hits else None,
                      "answered": within})

    n_ans = len(answerable)
    out = {
        "mode": "dense-only (baseline)",
        "corpus_docs": len(list(CORPUS.glob("*.pdf"))),
        "chunks_indexed": n_chunks,
        "n_answerable": n_ans,
        "n_abstain": len(abstain),
        "relevance_threshold": thr,
        "top_k": TOP_K,
        "recall": {f"@{k}": round(recall[k] / n_ans, 3) for k in RECALL_KS},
        "mrr": round(rr_sum / n_ans, 3),
        "abstain_accuracy": round(correct_abstain / len(abstain), 3),
        "false_abstain_rate": round(false_abstain / n_ans, 3),
        "per_question": per_q,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline.json").write_text(json.dumps(out, indent=2))

    print("\n==================== RETRIEVAL BASELINE (dense-only) ====================")
    print(f"corpus: {out['corpus_docs']} docs, {n_chunks} chunks   "
          f"questions: {n_ans} answerable + {len(abstain)} abstain")
    print(f"threshold (RELEVANCE_MAX_DISTANCE): {thr}\n")
    for k in RECALL_KS:
        print(f"  recall@{k:<2} {out['recall'][f'@{k}']:.3f}")
    print(f"  MRR       {out['mrr']:.3f}")
    print(f"  abstain accuracy     {out['abstain_accuracy']:.3f}  "
          f"({correct_abstain}/{len(abstain)} correctly refused)")
    print(f"  false-abstain rate   {out['false_abstain_rate']:.3f}  "
          f"({false_abstain}/{n_ans} answerable wrongly gated out)")
    print("\nwrote evals/results/baseline.json")


if __name__ == "__main__":
    main()
