#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="${AE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MTR_DIR="${MTR_DIR:-$AE_ROOT/MetricsTestRunner}"

MODE="${1:-print}"
GROUP="${2:-all}"

if [ "$MODE" != "print" ] && [ "$MODE" != "execute" ]; then
  echo "Usage: $0 [print|execute] [all|fig21|fig21a|fig21b|fig21c|fig21d|fig22]" >&2
  exit 1
fi

if [ ! -d "$MTR_DIR" ]; then
  echo "[ae] MetricsTestRunner not found at $MTR_DIR; run ae/setup_repos.sh first" >&2
  exit 1
fi

if [ "$MODE" = "execute" ] && [ ! -f "$MTR_DIR/sweep_test_robust.sh" ]; then
  echo "[ae] sweep_test_robust.sh not found in $MTR_DIR" >&2
  echo "[ae] switch MetricsTestRunner to branch ae before running GPU experiments" >&2
  exit 1
fi

print_header() {
  cat <<'EOF'
[ae] Full GPU reproduction uses MetricsTestRunner branch ae.
[ae] Entry point: sweep_test_robust.sh

[ae] Full experiment reproduction budget:
  - Hardware: 16 NVIDIA H20 96GB GPUs, launched as 16 vLLM instances.
  - Typical layout: 2 machines x 8 GPUs, with SSH access between them.
  - Each sweep point: TIME_IN_SEC=1200 seconds by default plus startup,
    router wait, cleanup, and log merge.
  - Practical estimate: 25-35 minutes per policy/trace/scale point on one
    16-GPU testbed.

[ae] Before executing:
  - cd "${MTR_DIR:-${AE_ROOT}/MetricsTestRunner}" && git checkout ae
  - cp ae.env.example ae.env
  - Edit ae.env for MODEL_PATH_30B, MODEL_PATH_7B, VENV_PATH, WORK_DIR,
    DATASET_DIR, OUTPUT_BASE, REMOTE_HOST, REMOTE_USER, and REMOTE_SSH_PORT.
  - source ae.env
  - The sweep builds router with the policy feature for every run, starts vLLM,
    starts the router, runs request-sim, waits, cleans up, and merges client.jsonl.

EOF
}

print_cmd() {
  local estimate="$1"
  local scope="$2"
  local cmd="$3"
  printf '  # estimated time: %s\n' "$estimate"
  printf '  # scope: %s\n' "$scope"
  printf '  cd "${MTR_DIR:-${AE_ROOT}/MetricsTestRunner}" && %s\n\n' "$cmd"
}

run_cmd() {
  local cmd="$1"
  (cd "$MTR_DIR" && eval "$cmd")
}

emit_or_run() {
  local estimate="$1"
  local scope="$2"
  local cmd="$3"
  if [ "$MODE" = "execute" ]; then
    echo "[ae] scope: $scope"
    echo "[ae] estimated time: $estimate"
    echo "[ae] executing: $cmd"
    run_cmd "$cmd"
  else
    print_cmd "$estimate" "$scope" "$cmd"
  fi
}

emit_index_set() {
  local estimate="$1"
  local scope="$2"
  shift 2
  local idx
  for idx in "$@"; do
    emit_or_run "$estimate" "$scope; sweep index $idx" "SWEEP_TEST_LIMIT=1 bash sweep_test_robust.sh $idx"
  done
}

if [ "${AE_RUN_EXPERIMENTS_NO_HEADER:-0}" != "1" ]; then
  print_header
fi

case "$GROUP" in
  all)
    AE_RUN_EXPERIMENTS_NO_HEADER=1 "$0" "$MODE" fig21
    AE_RUN_EXPERIMENTS_NO_HEADER=1 "$0" "$MODE" fig22
    ;;
  fig21)
    echo "[ae] Figure 21: end-to-end CDF points from sweep_test_robust.sh"
    AE_RUN_EXPERIMENTS_NO_HEADER=1 "$0" "$MODE" fig21a
    AE_RUN_EXPERIMENTS_NO_HEADER=1 "$0" "$MODE" fig21b
    AE_RUN_EXPERIMENTS_NO_HEADER=1 "$0" "$MODE" fig21c
    AE_RUN_EXPERIMENTS_NO_HEADER=1 "$0" "$MODE" fig21d
    ;;
  fig21a)
    echo "[ae] Figure 21(a): ChatBot/Qwen end-to-end CDF"
    emit_index_set "2-3 hours total" "Figure 21(a), ChatBot/Qwen, scale 5.6, 5 plotted policies" 0 10 15 25 30
    ;;
  fig21b)
    echo "[ae] Figure 21(b): Agent/API Qwen end-to-end CDF"
    emit_index_set "2-3 hours total" "Figure 21(b), Agent/API Qwen, Qwen2.5-7B-Instruct concrete checkpoint, scale 5.5, 5 plotted policies" 35 45 50 60 65
    ;;
  fig21c)
    echo "[ae] Figure 21(c): Coder end-to-end CDF"
    emit_index_set "2-3 hours total" "Figure 21(c), Coder, scale 2.4, 5 plotted policies" 72 82 87 97 102
    ;;
  fig21d)
    echo "[ae] Figure 21(d): ToolAgent/Kimi end-to-end CDF"
    emit_index_set "2-3 hours total" "Figure 21(d), ToolAgent/Kimi, scale 1.6, 5 plotted policies" 107 115 119 127 131
    ;;
  fig22)
    echo "[ae] Figure 22: scaling curves for the 5 plotted policies"
    emit_index_set "10-15 hours total" "Figure 22 ChatBot/Qwen scaling, selected indices from 0-34" 0 1 2 3 4 10 11 12 13 14 15 16 17 18 19 25 26 27 28 29 30 31 32 33 34
    emit_index_set "10-15 hours total" "Figure 22 Agent/API Qwen scaling, selected indices from 35-69" 35 36 37 38 39 45 46 47 48 49 50 51 52 53 54 60 61 62 63 64 65 66 67 68 69
    emit_index_set "10-15 hours total" "Figure 22 Coder scaling, selected indices from 70-104" 70 71 72 73 74 80 81 82 83 84 85 86 87 88 89 95 96 97 98 99 100 101 102 103 104
    emit_index_set "8-12 hours total" "Figure 22 ToolAgent/Kimi scaling, selected indices from 105-132" 105 106 107 108 113 114 115 116 117 118 119 120 125 126 127 128 129 130 131 132
    ;;
  *)
    echo "Usage: $0 [print|execute] [all|fig21|fig21a|fig21b|fig21c|fig21d|fig22]" >&2
    exit 1
    ;;
esac
