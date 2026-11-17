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

## Note on difficulty
The first corpus is clean and well-separated, so dense retrieval already saturates
(recall@5 = 1.0). A larger, more confusable corpus with keyword/identifier-style
queries is used to actually discriminate dense vs BM25 vs hybrid — see the hybrid
eval results.
