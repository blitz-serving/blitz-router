#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="${AE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
YAULLM_DIR="${YAULLM_DIR:-$AE_ROOT/yaullm}"
VLLM_VENV="${VLLM_VENV:-$YAULLM_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_EXTRA_PACKAGES="${VLLM_EXTRA_PACKAGES:-flashinfer-python==0.6.9}"
VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://wheels.vllm.ai/d31a64712489d7c079fe48515c7ddd8a60bc0e71/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl}"
export VLLM_USE_PRECOMPILED
export VLLM_PRECOMPILED_WHEEL_LOCATION

if [ ! -f "$YAULLM_DIR/pyproject.toml" ]; then
  echo "[ae] yaullm checkout not found at $YAULLM_DIR" >&2
  echo "[ae] run ae/setup_repos.sh first, or set YAULLM_DIR to the patched yaullm checkout" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ae] Python not found: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$(dirname "$VLLM_VENV")"
"$PYTHON_BIN" -m venv "$VLLM_VENV"

PY="$VLLM_VENV/bin/python"
PIP=("$PY" -m pip)

echo "[ae] VLLM_USE_PRECOMPILED=$VLLM_USE_PRECOMPILED"
if [ "$VLLM_USE_PRECOMPILED" = "1" ]; then
  echo "[ae] VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_PRECOMPILED_WHEEL_LOCATION"
fi

"${PIP[@]}" install --upgrade pip
"${PIP[@]}" install -r "$YAULLM_DIR/requirements/build.txt" -r "$YAULLM_DIR/requirements/cuda.txt"

if [ -n "$VLLM_EXTRA_PACKAGES" ]; then
  # shellcheck disable=SC2086
  "${PIP[@]}" install $VLLM_EXTRA_PACKAGES
fi

(cd "$YAULLM_DIR" && "${PIP[@]}" install --no-build-isolation -e .)

if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
  echo "[ae] vLLM executable was not created at $VLLM_VENV/bin/vllm" >&2
  exit 1
fi

cat <<EOF
[ae] Patched vLLM environment is ready:
  $VLLM_VENV

[ae] Put this in MetricsTestRunner/ae.env:
  export VENV_PATH=$VLLM_VENV

[ae] If your two-node setup does not share this filesystem, run this script on
[ae] the remote node as well, with YAULLM_DIR and VLLM_VENV set to paths that
[ae] exist on that node. Then put the remote venv path in REMOTE_VENV_PATH.
EOF
