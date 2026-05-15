#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="${AE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MTR_DIR="${MTR_DIR:-$AE_ROOT/MetricsTestRunner}"
PLOTS_DIR="${PLOTS_DIR:-$AE_ROOT/xmetric-plots}"
MAP_FILE="${MAP_FILE:-$SCRIPT_DIR/slow_path_map.tsv}"
GROUP="${1:-all}"
SLOW_DATA_DIR="${SLOW_DATA_DIR:-$PLOTS_DIR/slow-data}"
AE_ENV_FILE="${AE_ENV_FILE:-$MTR_DIR/ae.env}"
PLOT_VENV="${AE_PLOT_VENV:-$SCRIPT_DIR/.venv}"
PLOT_PYTHON="${PLOT_PYTHON:-$PLOT_VENV/bin/python}"

if [ ! -f "$MAP_FILE" ]; then
  echo "[ae] slow path map not found at $MAP_FILE" >&2
  exit 1
fi

if [ ! -f "$MTR_DIR/sweep_test_robust.sh" ]; then
  echo "[ae] sweep_test_robust.sh not found in $MTR_DIR" >&2
  exit 1
fi

if [ -f "$AE_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$AE_ENV_FILE"
  set +a
fi

if [ "$GROUP" != "all" ] && [ "$GROUP" != "fig21" ] && [ "$GROUP" != "fig21a" ] \
  && [ "$GROUP" != "fig21b" ] && [ "$GROUP" != "fig21c" ] && [ "$GROUP" != "fig21d" ] \
  && [ "$GROUP" != "fig22" ]; then
  echo "Usage: $0 [all|fig21|fig21a|fig21b|fig21c|fig21d|fig22]" >&2
  exit 1
fi

detect_output_base() {
  if [ -n "${OUTPUT_BASE:-}" ]; then
    printf '%s\n' "$OUTPUT_BASE"
    return
  fi
  printf '%s\n' "$AE_ROOT/slow-path-logs"
}

OUTPUT_BASE_DETECTED="$(detect_output_base)"
if [ -z "$OUTPUT_BASE_DETECTED" ]; then
  echo "[ae] unable to determine OUTPUT_BASE; export OUTPUT_BASE to the sweep output directory" >&2
  exit 1
fi

mkdir -p "$OUTPUT_BASE_DETECTED" "$SLOW_DATA_DIR"

trace_short() {
  printf '%s\n' "$1" \
    | sed 's/\.jsonl$//' \
    | sed 's/_blksz_16//' \
    | sed 's/anony-//' \
    | sed 's/qwen_//' \
    | sed 's/-/./g'
}

latest_client_dir() {
  local run_dir="$1"
  find "$run_dir" -mindepth 3 -maxdepth 3 -type f -path '*/attempt*/node1/client.jsonl' \
    | sort \
    | tail -1 \
    | xargs -r dirname
}

import_for_index() {
  local idx="$1"
  local run_dir="$2"
  local client_dir
  client_dir="$(latest_client_dir "$run_dir")"
  if [ -z "$client_dir" ]; then
    echo "[ae] no attempt*/node1/client.jsonl found under $run_dir" >&2
    exit 1
  fi

  awk -F'\t' -v idx="$idx" -v group="$GROUP" '
    NR == 1 { next }
    $3 == idx && (group == "all" || index($1, group) > 0) { print $2 }
  ' "$MAP_FILE" | while IFS= read -r canonical; do
    rm -rf "$SLOW_DATA_DIR/$canonical"
    ln -s "$client_dir" "$SLOW_DATA_DIR/$canonical"
    echo "[ae] linked slow data $canonical -> $client_dir"
  done
}

run_one_index() {
  local idx="$1"
  local before after new_dirs run_dir
  before="$(mktemp)"
  after="$(mktemp)"
  find "$OUTPUT_BASE_DETECTED" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$before"

  echo "[ae] running sweep index $idx"
  (cd "$MTR_DIR" && SWEEP_TEST_LIMIT=1 bash sweep_test_robust.sh "$idx")

  find "$OUTPUT_BASE_DETECTED" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$after"
  new_dirs="$(comm -13 "$before" "$after" || true)"
  rm -f "$before" "$after"

  if [ -n "$new_dirs" ]; then
    run_dir="$OUTPUT_BASE_DETECTED/$(printf '%s\n' "$new_dirs" | tail -1)"
  else
    run_dir="$(find "$OUTPUT_BASE_DETECTED" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  fi

  if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    echo "[ae] unable to locate output directory for sweep index $idx" >&2
    exit 1
  fi

  import_for_index "$idx" "$run_dir"
}

echo "[ae] slow path group: $GROUP"
echo "[ae] MetricsTestRunner: $MTR_DIR"
echo "[ae] AE env file: $AE_ENV_FILE"
echo "[ae] OUTPUT_BASE: $OUTPUT_BASE_DETECTED"
echo "[ae] slow data directory: $SLOW_DATA_DIR"

awk -F'\t' -v group="$GROUP" '
  NR == 1 { next }
  group == "all" || index($1, group) > 0 { print $3 }
' "$MAP_FILE" | sort -n -u | while IFS= read -r idx; do
  run_one_index "$idx"
done

echo "[ae] regenerating figures from slow-path data"
XMETRIC_DATA_DIR="$SLOW_DATA_DIR" "$SCRIPT_DIR/setup_plot_env.sh"

run_plot() {
  local rel="$1"
  XMETRIC_DATA_DIR="$SLOW_DATA_DIR" "$PLOT_PYTHON" "$SCRIPT_DIR/run_python_figure.py" "$PLOTS_DIR/$rel"
}

case "$GROUP" in
  all)
    XMETRIC_DATA_DIR="$SLOW_DATA_DIR" "$SCRIPT_DIR/run_figures.sh" evaluation
    ;;
  fig21)
    run_plot "figs/e2e-cdf-v1/qwen3-to-c.py"
    run_plot "figs/e2e-cdf-v1/qwen2-to-b.py"
    run_plot "figs/e2e-cdf-v1/qwen3-coder.py"
    run_plot "figs/e2e-cdf-v1/qwen2-mooncake-tool.py"
    ;;
  fig21a)
    run_plot "figs/e2e-cdf-v1/qwen3-to-c.py"
    ;;
  fig21b)
    run_plot "figs/e2e-cdf-v1/qwen2-to-b.py"
    ;;
  fig21c)
    run_plot "figs/e2e-cdf-v1/qwen3-coder.py"
    ;;
  fig21d)
    run_plot "figs/e2e-cdf-v1/qwen2-mooncake-tool.py"
    ;;
  fig22)
    run_plot "figs/scaling-test/scaling-fix-v1.py"
    ;;
esac
