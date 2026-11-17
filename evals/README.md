# Retrieval & RAG evals

A reproducible eval set for the retriever and the abstention gate. Used to measure
changes (dense vs hybrid, quantization levels) with numbers instead of vibes.

## Corpus
Self-authored, fictional sample documents (no real company, person, or copyrighted
text) — a service agreement, an employee handbook, a product manual, and a privacy
policy. One topic per page, so each gold question maps to an unambiguous page.

```
.venv/bin/python evals/make_corpus.py     # builds evals/corpus/*.pdf (git-ignored)
```

## Gold set
`gold.jsonl` — questions labelled with the document and page that answer them, plus
unanswerable questions (`answerable: false`) that the system *should* refuse.

## Run
```
PYTHONPATH=. .venv/bin/python evals/run_baseline.py   # dense-only baseline
```
Writes `results/baseline.json`.

## Metrics
- **recall@k** — fraction of answerable questions whose correct doc+page appears in the top k.
- **MRR** — mean reciprocal rank of the first correct chunk.
- **abstain accuracy** — fraction of unanswerable questions correctly refused (no chunk within the relevance threshold).
- **false-abstain rate** — answerable questions wrongly gated out by the threshold.

## dense vs BM25 vs hybrid
```
PYTHONPATH=. .venv/bin/python evals/run_compare.py   # writes results/retrieval_compare.json
```
7 docs (47 chunks). The gold set includes a **hard subset** (18 exact-term /
identifier / confusable questions — error codes, endpoints, bug IDs) where keyword
matching is expected to help.

| | recall@1 | recall@3 | recall@5 | MRR |
| --- | --- | --- | --- | --- |
| **All (58)** dense | 0.897 | 0.966 | 0.983 | 0.938 |
| All — bm25 | 0.897 | 0.931 | 0.966 | 0.924 |
| All — **hybrid** | **0.914** | 0.948 | 0.966 | **0.940** |
| **Hard (18)** dense | 0.889 | 0.944 | 0.944 | 0.926 |
| Hard — **bm25** | **0.944** | **1.000** | **1.000** | **0.972** |
| Hard — hybrid | 0.889 | 0.944 | 1.000 | 0.931 |

**Read:** on exact-term queries **BM25 clearly wins** (recall@3 1.00 vs dense 0.94).
Across all queries **hybrid has the best recall@1 and MRR** — it's the robust default
that doesn't collapse on either query type. No single mode wins everything; that's
the honest result, and why hybrid (RRF) is a sensible default with the mode left
switchable.
