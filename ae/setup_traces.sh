#!/usr/bin/env bash
set -euo pipefail

AE_ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TRACE_DIR="${TRACE_DIR:-$AE_ROOT/qwen-bailian-usagetraces-anon}"

if [ -d "$TRACE_DIR/.git" ]; then
  echo "[ae] Alibaba/Qwen trace repo already exists at $TRACE_DIR; leaving checkout untouched"
else
  git clone https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon "$TRACE_DIR"
fi

cat <<EOF
[ae] Trace preparation complete.

Alibaba/Qwen traces:
  $TRACE_DIR

MetricsTestRunner expects this directory through DATASET_DIR in:
  MetricsTestRunner/ae.env

The paper also uses the Mooncake ToolAgent trace:
  https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/traces/toolagent_trace.jsonl

The converted AE copy is released in:
  xmetric-plots/traces/mooncake_toolagent_trace_poissoned.jsonl

Place or symlink that file under DATASET_DIR with the same filename before
running the ToolAgent/Kimi experiments.
EOF
