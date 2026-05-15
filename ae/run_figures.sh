#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="${AE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PLOTS_DIR="${PLOTS_DIR:-${PAPER_DIR:-$AE_ROOT/xmetric-plots}}"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"

GROUP="${1:-evaluation}"

if [ ! -x "$PYTHON" ]; then
  echo "[ae] Plot venv not found at $PYTHON; run ae/setup_plot_env.sh first" >&2
  exit 1
fi

if [ ! -d "$PLOTS_DIR/figs" ]; then
  echo "[ae] xmetric-plots not found at $PLOTS_DIR" >&2
  exit 1
fi

export MPLBACKEND=Agg

run_one() {
  local rel="$1"
  echo "[ae] plotting $rel"
  "$PYTHON" "$SCRIPT_DIR/run_python_figure.py" "$PLOTS_DIR/$rel"
}

case "$GROUP" in
  evaluation)
    run_one "figs/e2e-cdf-v1/qwen3-to-c.py"
    run_one "figs/e2e-cdf-v1/qwen2-to-b.py"
    run_one "figs/e2e-cdf-v1/qwen3-coder.py"
    run_one "figs/e2e-cdf-v1/qwen2-mooncake-tool.py"
    run_one "figs/scaling-test/scaling-fix-v1.py"
    ;;
  all)
    find "$PLOTS_DIR/figs" -type f -name '*.py' \
      ! -name '__init__.py' \
      ! -name 'common.py' \
      ! -name 'a_plus_b.py' \
      | sort \
      | while IFS= read -r script; do
          run_one "${script#$PLOTS_DIR/}"
        done
    ;;
  *)
    echo "Usage: $0 [evaluation|all]" >&2
    exit 1
    ;;
esac

echo "[ae] Figure generation complete. Outputs are under $PLOTS_DIR/figs/."
