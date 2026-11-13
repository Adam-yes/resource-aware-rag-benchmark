#!/bin/bash
# (Re)starts the project-local Ollama server with N parallel slots and waits until it answers.
# N=1 for latency-clean generation, N=8 for judging, N=4 for the concurrency test.
set -euo pipefail
N="${1:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RT="$HERE/../runtime"
pkill -f "$RT/bin/ollama serve" 2>/dev/null || true
sleep 1
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$RT/models" OLLAMA_NUM_PARALLEL="$N" \
  OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m \
  nohup "$RT/bin/ollama" serve >> "$RT/ollama.log" 2>&1 &
for _ in $(seq 60); do
  curl -sf --max-time 1 http://127.0.0.1:11434/api/version >/dev/null && exit 0
  sleep 1
done
echo "Ollama did not start, see $RT/ollama.log" >&2
exit 1
