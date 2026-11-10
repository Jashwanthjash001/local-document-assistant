#!/usr/bin/env bash
#
# bench.sh — benchmark a GGUF model's throughput (prefill + decode) with llama-bench.
#   ./serving/bench.sh <model.gguf> [label]
# Output: serving/results/<label>.csv (+ a derived .md).
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?Usage: bench.sh <model.gguf> [label]}"
LABEL="${2:-$(basename "$MODEL" .gguf)}"
PROMPT_TOKENS="${PROMPT_TOKENS:-256}"; GEN_TOKENS="${GEN_TOKENS:-64}"
NGL="${NGL:-99}"; REPS="${REPS:-3}"
mkdir -p "$ROOT/results"
echo ">> $MODEL  (pp=$PROMPT_TOKENS tg=$GEN_TOKENS ngl=$NGL reps=$REPS)"
"$ROOT/bin/llama-bench" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -ngl "$NGL" -r "$REPS" -o csv \
  | tee "$ROOT/results/${LABEL}.csv"
echo ">> saved serving/results/${LABEL}.csv"
