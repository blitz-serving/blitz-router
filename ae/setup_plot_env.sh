#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${AE_PLOT_VENV:-$SCRIPT_DIR/.venv}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-plot.txt"

echo "[ae] Plotting environment ready: $VENV_DIR"
echo "[ae] Activate with: source $VENV_DIR/bin/activate"
