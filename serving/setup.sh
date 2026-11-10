#!/usr/bin/env bash
#
# setup.sh — download the llama.cpp binaries + the GGUF model into serving/
# (both are git-ignored because they're large). macOS Apple Silicon by default;
# see the README table for other platforms' llama.cpp builds.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/vendor" "$ROOT/models"

BUILD="b12119"
echo ">> downloading llama.cpp $BUILD (macOS arm64 / Metal)…"
curl -L -o "$ROOT/vendor/llama.tar.gz" \
  "https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/llama-$BUILD-bin-macos-arm64.tar.gz"
mkdir -p "$ROOT/vendor/llama-$BUILD"
tar -xzf "$ROOT/vendor/llama.tar.gz" -C "$ROOT/vendor/llama-$BUILD"
# macOS only: clear Gatekeeper quarantine on the downloaded binaries
xattr -dr com.apple.quarantine "$ROOT/vendor/llama-$BUILD" 2>/dev/null || true

MODEL="$ROOT/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
if [[ ! -f "$MODEL" ]]; then
  echo ">> downloading Llama 3.1 8B Q4_K_M (~4.9 GB)…"
  curl -L -o "$MODEL" \
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
fi
echo ">> done. Start the server with:  ./serving/serve.sh"
