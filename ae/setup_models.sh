#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="${AE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODELS_DIR="${MODELS_DIR:-$AE_ROOT/models}"
MODEL_VENV="${MODEL_VENV:-$SCRIPT_DIR/.venv-models}"

QWEN_7B_REPO="${QWEN_7B_REPO:-Qwen/Qwen2.5-7B-Instruct}"
QWEN_30B_REPO="${QWEN_30B_REPO:-Qwen/Qwen3-30B-A3B}"

QWEN_7B_DIR="${QWEN_7B_DIR:-$MODELS_DIR/$(basename "$QWEN_7B_REPO")}"
QWEN_30B_DIR="${QWEN_30B_DIR:-$MODELS_DIR/$(basename "$QWEN_30B_REPO")}"

mkdir -p "$MODELS_DIR"

need_download=false
for dir in "$QWEN_7B_DIR" "$QWEN_30B_DIR"; do
  if [ ! -f "$dir/config.json" ]; then
    need_download=true
  fi
done

if [ "$need_download" = true ] && [ ! -x "$MODEL_VENV/bin/huggingface-cli" ]; then
  python3 -m venv "$MODEL_VENV"
  "$MODEL_VENV/bin/python" -m pip install --upgrade pip
  "$MODEL_VENV/bin/python" -m pip install "huggingface_hub[cli]"
fi

download_model() {
  local repo="$1"
  local dir="$2"
  if [ -f "$dir/config.json" ]; then
    echo "[ae] model already exists at $dir; skipping download for $repo"
    return 0
  fi

  echo "[ae] downloading $repo to $dir"
  "$MODEL_VENV/bin/huggingface-cli" download "$repo" \
    --local-dir "$dir"
}

download_model "$QWEN_7B_REPO" "$QWEN_7B_DIR"
download_model "$QWEN_30B_REPO" "$QWEN_30B_DIR"

cat <<EOF
[ae] Model preparation complete.

7B model:
  repo: $QWEN_7B_REPO
  path: $QWEN_7B_DIR

30B model:
  repo: $QWEN_30B_REPO
  path: $QWEN_30B_DIR

Use these paths as MODEL_PATH_7B and MODEL_PATH_30B in
MetricsTestRunner/ae.env, or override the download script with:
  MODELS_DIR=/path/to/models
  QWEN_7B_REPO=<hf-org/model>
  QWEN_30B_REPO=<hf-org/model>
  QWEN_7B_DIR=/existing/path
  QWEN_30B_DIR=/existing/path
EOF
