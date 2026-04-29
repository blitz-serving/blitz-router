#!/bin/bash
set -euo pipefail

MODEL_PATH="/models/Qwen3-30B-A3B"
MODEL_NAME="Qwen3-30B-A3B"
TOKENIZER_PATH="/nvme/models/Qwen3-30B-A3B/tokenizer.json"
TOKENIZER_CONFIG="/nvme/models/Qwen3-30B-A3B/tokenizer_config.json"
TRACES_DIR="/workspace/tmp/bailian-traces"
ROUTER_DIR="/workspace/blitz-router"
CLIENT_BIN="/workspace/request-sim/target/release/request-sim"
ROUTER_PORT=58009
CONTEXT_LENGTH=8192
TIME_IN_SECS=180
SCALE_FACTOR=0.5
OUTPUT_BASE="/workspace/tmp/lmetric/e2e-results/$(date +%Y%m%d-%H%M%S)"

declare -A POLICY_BINS=(
  [rr]="router_round-robin-q"
  [bailian]="router_bailian-impl-q"
  [aibrix]="router_aibrix"
  [dynamo]="router_dynamo"
  [lmetric]="router_bounded-most-hit-q"
)

TRACES=(
  "qwen_traceA_blksz_16.jsonl"
)

mkdir -p "$OUTPUT_BASE"

free_port() {
    local pids
    pids=$(ss -tlnp | grep ":$1 " | grep -oP 'pid=\K[0-9]+' || true)
    for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
    sleep 1
}

run_one() {
    local policy="$1"
    local trace_file="$2"
    local trace_name="${trace_file%.jsonl}"
    local router_bin="${ROUTER_DIR}/target/release/${POLICY_BINS[$policy]}"
    local out_dir="${OUTPUT_BASE}/${policy}/${trace_name}"
    mkdir -p "$out_dir"

    echo ""
    echo "======== Policy: $policy | Trace: $trace_name ========"
    free_port "$ROUTER_PORT"

    "$router_bin" \
        --port "$ROUTER_PORT" \
        --client-config "${ROUTER_DIR}/exps/blitz-run/configs/config-stubs.json" \
        --model-name "$MODEL_NAME" \
        --tokenizer-name "/nvme/models/${MODEL_NAME}" \
        --use-tokenizer \
        --max-input-length "$((CONTEXT_LENGTH - 1))" \
        --max-total-tokens "$CONTEXT_LENGTH" \
        --max-batch-prefill-tokens 19999 \
        --max-concurrent-requests 4096 \
        --kvcache-block-size 16 \
        --log-path "${out_dir}/router.log" \
        > "${out_dir}/router_stdout.log" 2>&1 &
    local router_pid=$!
    sleep 8

    if ! kill -0 "$router_pid" 2>/dev/null; then
        echo "  ERROR: Router failed to start!"
        cat "${out_dir}/router_stdout.log" | tail -5
        return 1
    fi
    echo "  Router OK (PID $router_pid)"

    timeout $((TIME_IN_SECS + 120)) "$CLIENT_BIN" \
        --tokenizer "$TOKENIZER_PATH" \
        --tokenizer-config "$TOKENIZER_CONFIG" \
        --endpoint "http://localhost:${ROUTER_PORT}/generate" \
        --api tgi \
        --dataset bailian \
        --dataset-path "${TRACES_DIR}/${trace_file}" \
        --scale-factor "$SCALE_FACTOR" \
        --time-in-secs "$TIME_IN_SECS" \
        --output-path "${out_dir}/results.jsonl" \
        --context-length "$CONTEXT_LENGTH" \
        --track-output \
        > "${out_dir}/client_stdout.log" 2>&1 || true

    kill "$router_pid" 2>/dev/null || true
    wait "$router_pid" 2>/dev/null || true
    sleep 3
    free_port "$ROUTER_PORT"

    local count
    count=$(wc -l < "${out_dir}/results.jsonl" 2>/dev/null || echo 0)
    echo "  Done: $count results -> ${out_dir}/results.jsonl"
}

echo "============================================"
echo " lmetric Experiment: $MODEL_NAME, 8GPU"
echo " Scale=$SCALE_FACTOR, Time=${TIME_IN_SECS}s"
echo " Output: $OUTPUT_BASE"
echo "============================================"

for trace in "${TRACES[@]}"; do
    for policy in rr bailian aibrix dynamo lmetric; do
        run_one "$policy" "$trace" || echo "WARNING: $policy/$trace failed"
    done
done

echo ""
echo "Experiment complete! Results: $OUTPUT_BASE"
