# Case Study — Quantization tradeoffs for on-device Llama 3.1 8B

**Goal:** determine whether a 16 GB MacBook Pro can run Llama 3.1 8B Instruct usefully and
locally, and quantify the cost of the quantization needed to make it fit.

## Method

- **Engine:** llama.cpp (build 12119), all layers offloaded to the Apple M1 Pro GPU via Metal (`-ngl 99`).
- **Throughput:** `llama-bench`, 5 repetitions, 512-token prompt (prefill, `pp512`) and 128-token generation (decode, `tg128`). Reported as mean ± stddev.
- **Backend:** reported by llama-bench as `MTL,BLAS` (Metal), 6 threads.
- **Output format:** the original benchmark script ran `llama-bench` **twice** per model — once to emit a Markdown table, once to emit CSV. This has since been consolidated to a single pass (CSV as source of truth, Markdown derived from it) to reduce load.
- **Precisions:** Q4_K_M (4-bit k-quant, practical 16 GB target) vs Q8_0 (8-bit, near-lossless reference).
- **Quality:** `llama-perplexity` over a ~40 KB slice of the WikiText-2 test set (18 chunks, `n_ctx 512`), reported as perplexity ± stderr. Lower is better; both models run on the identical text.

## Measured results

| Precision | Size on disk | Prefill (pp512)    | Decode (tg128)    | Perplexity (WikiText-2) |
| --------- | ------------ | ------------------ | ----------------- | ----------------------- |
| Q4_K_M    | 4.58 GiB     | 204.19 ± 20.00 t/s | 24.43 ± 0.36 t/s  | 8.05 ± 0.29             |
| Q8_0      | 7.95 GiB     | 103.37 ± 19.67 t/s | 14.90 ± 0.76 t/s  | 7.95 ± 0.29 (reference) |

(Raw data: throughput [`results/q4_k_m.csv`](../results/q4_k_m.csv) · [`results/q8_0.csv`](../results/q8_0.csv); perplexity [`results/perplexity.csv`](../results/perplexity.csv). Lower perplexity = better.)

## Findings

- **Speed:** Q4_K_M runs ~2.0× faster at prefill (204 vs 103 t/s) and ~1.6× faster at decode (24.4 vs 14.9 t/s). Decode speed is what a user feels as "typing speed": ~24 t/s is comfortably faster than reading speed; ~15 t/s is usable but noticeably slower.
- **Size:** Q4_K_M is 42% smaller **on disk** (4.58 vs 7.95 GiB). Note this is file size, not runtime memory — actual RAM use also depends on context length, KV cache, and other allocations, which were not separately measured here.
- **Quality:** Q4_K_M scores **8.05** perplexity vs Q8_0's **7.95** — a **+0.10 (~1.3%)** increase that sits *within* the ±0.29 confidence interval on this slice. So on this benchmark 4-bit quantization is effectively near-lossless: the 2× speed and 42% size win comes at no measurable quality cost.

## Thermal behavior & decision to pause inference

Running `llama-bench` drives the GPU flat-out — by design it is the most thermally intense
workload in this project. Back-to-back Q4_K_M and Q8_0 benchmark passes (each run twice by
the original script to emit both Markdown and CSV) plus smoke-test generations drove the M1
Pro fans to full speed, and the user reported the chassis getting hot quickly.

Two **non-inference** background loads were also observed in `top` samples around the same
time and are *suspected* contributors to the heat, though not confirmed as the cause:

- macOS **Spotlight** (`mds_stores` ~63% CPU, `mdworker` ~30% CPU) — consistent with indexing the ~13 GB of newly downloaded GGUF files.
- A **VS Code git** process (~81% CPU, parent process was VS Code) — consistent with its Git integration scanning the same new files. It exited on its own.

**Decision:** inference was paused to let the machine cool, at the user's direction. The
benchmark numbers above were already captured and are preserved, so no measurement was lost.
The output-quality side-by-side was later run as a short, low-load perplexity pass over a small
WikiText-2 slice (one model at a time) — see the results above. The benchmark harness was also
made lighter — a single pass instead of two, and fewer default repetitions — to reduce load.

## Takeaway for a client

- A capable 8B assistant runs locally on existing 16 GB laptop hardware, with no cloud dependency, at ~24 tokens/sec (Q4_K_M).
- **Q4_K_M is the performance and memory default on 16 GB** — ~2× the throughput, 42% smaller on disk, and a perplexity gap vs Q8_0 that's within measurement noise (8.05 vs 7.95). Q8_0 remains the reference if a workload later proves quality-sensitive.
- The same model is served behind an OpenAI-compatible API (`serve.sh`), so existing application code switches from cloud to local with a one-line base-URL change.
- Sustained benchmarking is thermally heavy; day-to-day interactive use is far lighter (the GPU engages only while generating).

## Status

Performance-focused quantization case study: **complete**. Output-quality comparison:
**complete** — Q4_K_M vs Q8_0 perplexity measured on a WikiText-2 slice (above). The gap is
within measurement noise, confirming 4-bit is near-lossless here. A larger WikiText-2 run would
tighten the confidence interval but isn't expected to change the conclusion.
