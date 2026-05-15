#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-figures}"

case "$MODE" in
  figures|archived)
    "$SCRIPT_DIR/setup_plot_env.sh"
    "$SCRIPT_DIR/run_figures.sh" evaluation
    ;;
  full)
    "$SCRIPT_DIR/setup_repos.sh"
    "$SCRIPT_DIR/setup_traces.sh"
    echo "[ae] skipping model download in run_all.sh full; run ae/setup_models.sh if models are not already available"
    "$SCRIPT_DIR/run_slow_path.sh" all
    ;;
  print-experiments)
    "$SCRIPT_DIR/run_experiments.sh" print
    ;;
  *)
    echo "Usage: $0 [figures|archived|full|print-experiments]" >&2
    exit 1
    ;;
esac
