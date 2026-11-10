#!/usr/bin/env bash
#
# serve.sh — serve the local model over an OpenAI-compatible HTTP API on :8080,
# which is exactly what the RAG backend (app/config.py LLM_BASE_URL) calls.
#
#   ./serving/serve.sh [model.gguf] [port]
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-$ROOT/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
PORT="${2:-8080}"
[[ -f "$MODEL" ]] || { echo "Model not found: $MODEL  (run ./serving/setup.sh)" >&2; exit 1; }
echo ">> Serving $(basename "$MODEL") on http://localhost:$PORT  (OpenAI-compatible /v1)"
exec "$ROOT/bin/llama-server" -m "$MODEL" -ngl 99 -c 4096 --host 127.0.0.1 --port "$PORT"
