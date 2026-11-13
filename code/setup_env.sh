#!/bin/bash
# Installs Ollama and the Python env inside the repository (no sudo, no system service).
# Usage: ./setup_env.sh            -> runtime + Python env + smoke-test models
#        ./setup_env.sh --all      -> also pull every model named in protocol.yaml
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RT="$HERE/../runtime"
V=v0.40.14

if [ ! -x "$RT/bin/ollama" ]; then
  curl -fL --retry 3 -o "$RT/ollama.tar.zst" \
    "https://github.com/ollama/ollama/releases/download/$V/ollama-linux-amd64.tar.zst"
  tar --zstd -xf "$RT/ollama.tar.zst" -C "$RT"
  rm "$RT/ollama.tar.zst"
fi
[ -x "$HERE/.venv/bin/python" ] || uv venv -q --python 3.12 "$HERE/.venv"
uv pip install -q --python "$HERE/.venv/bin/python" -r "$HERE/requirements.txt"

"$HERE/serve.sh" 1
pull() { OLLAMA_HOST=127.0.0.1:11434 "$RT/bin/ollama" pull "$1"; }
if [ "${1:-}" = "--all" ]; then
  for m in $("$HERE/.venv/bin/python" -c "
import yaml;p=yaml.safe_load(open('$HERE/protocol.yaml'))
print(*{p['embedding_model'],p['judge_primary'],p['judge_secondary'],*p['grid']['model']})"); do pull "$m"; done
else
  pull embeddinggemma:300m
  pull gemma3:1b
  pull gpt-oss:20b
fi
