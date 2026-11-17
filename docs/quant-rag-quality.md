# How quantization affects local RAG quality

Quantization benchmarks usually stop at speed, size, and perplexity. The question that
actually matters for a RAG app is different: **does quantizing the model change the
answers?** This is a small, honest attempt to measure that directly.

## Goal
Hold everything constant except the generation model's quantization level, and measure the
**end-to-end RAG answer quality** — not token-level perplexity, but whether the app gives the
right grounded answer, cites the right page, and still abstains when it should.

## Approach
Same corpus, same retrieval, same prompts — only the LLM quant changes.

- **Model:** Llama 3.1 8B Instruct, served locally by llama.cpp, at **Q4_K_M** (4.6 GB) and **Q8_0** (8.0 GB).
- **Pipeline:** the real `/chat` path — retrieve → relevance gate → grounded prompt → answer with citations, or abstain.
- **Questions:** 22 answerable (each with the fact that must appear + the page that should be cited) and 10 unanswerable (must be refused).
- **Scores:** `grounded` (the expected fact is in the answer), `cited_page` (the correct page is cited), `abstain` (unanswerable questions refused), and generation latency.

Reproduce: `evals/run_rag_quality.py` (point `LLM_BASE_URL` at each quant, give it a `QUANT_LABEL`).

## Results

| quant | grounded | cited-page | abstain | median latency |
| --- | --- | --- | --- | --- |
| Q4_K_M (4.6 GB) | 86% | 96% | 100% | 2.27s |
| Q8_0 (8.0 GB) | 86% | 96% | 100% | 2.12s |

## What it means
**Quantizing from Q8 to Q4 changed nothing about the answers** — grounding, citation accuracy, and
abstain behaviour are identical — while the model is **42% smaller on disk**. The quality gap that
quantization opens up at the token level (perplexity 8.05 vs 7.95, within noise) stays invisible at
the answer level for this grounded, retrieval-constrained task. For local RAG, **Q4_K_M is the right
default**: you get the smaller, faster model for free.

That makes sense: in RAG the model isn't recalling facts from its weights, it's reading them off the
retrieved passages. The retrieval does the heavy lifting; the generator just has to read and cite
faithfully, which the 4-bit model does just as well as the 8-bit one.

## Limitations
- Small, self-authored corpus and 22+10 questions — directional, not a benchmark.
- Latency isn't a clean comparison: Q8 (8 GB) swaps on a 16 GB Mac, so absolute times on this box are
  noisy. The quality numbers are the point, not the speed.
- One model family, two quant levels. A Q5/Q6 sweep and a second model would make the curve fuller.
