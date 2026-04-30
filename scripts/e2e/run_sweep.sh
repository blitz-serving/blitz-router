#!/bin/bash
set -euo pipefail

MODEL_NAME="/models/Qwen3-30B-A3B"
TOKENIZER_DIR="/nvme/models/Qwen3-30B-A3B"
TOKENIZER_PATH="${TOKENIZER_DIR}/tokenizer.json"
TOKENIZER_CONFIG="${TOKENIZER_DIR}/tokenizer_config.json"
TRACES_DIR="/workspace/tmp/bailian-traces"
ROUTER_DIR="/workspace/blitz-router"
CLIENT_BIN="/workspace/request-sim/target/release/request-sim"
ROUTER_PORT=58009
CONTEXT_LENGTH=8192
TIME_IN_SECS=480
CONFIG_STUBS="${ROUTER_DIR}/exps/blitz-run/configs/config-stubs-5gpu.json"
OUTPUT_BASE="/workspace/tmp/lmetric/sweep-5gpu/$(date +%Y%m%d-%H%M%S)"

declare -A POLICY_BINS=(
  [vllm]="router_join-shortest-q-weight"
  [bailian]="router_bailian-impl-q"
  [aibrix]="router_aibrix-q"
  [dynamo-t1]="router_dynamo-q"
  [dynamo-t2]="router_dynamo-po-q"
  [lmetric]="router_lmetric-q"
  [preble]="router_preble-q"
)

TRACE_FILE="${1:-qwen_traceA_blksz_16.jsonl}"
START_SF="${2:-2.5}"
DELTA_SF="${3:-0.5}"
MAX_SF="${4:-8.0}"
TRACE_NAME="${TRACE_FILE%.jsonl}"

mkdir -p "$OUTPUT_BASE"

free_port() {
    local pids
    pids=$(ss -tlnp | grep ":$1 " | grep -oP 'pid=\K[0-9]+' || true)
    for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
    sleep 1
}

run_one() {
    local policy="$1" sf="$2"
    local router_bin="${ROUTER_DIR}/target/release/${POLICY_BINS[$policy]}"
    local out_dir="${OUTPUT_BASE}/${policy}/sf${sf}_${TRACE_NAME}"
    mkdir -p "$out_dir"

    echo ""
    echo "======== Policy: $policy | SF: $sf | Trace: $TRACE_NAME ========"
    free_port "$ROUTER_PORT"

    "$router_bin" \
        --port "$ROUTER_PORT" \
        --client-config "$CONFIG_STUBS" \
        --model-name "$MODEL_NAME" \
        --tokenizer-name "$TOKENIZER_DIR" \
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
        tail -10 "${out_dir}/router_stdout.log"
        return 1
    fi
    echo "  Router OK (PID $router_pid)"

    timeout $((TIME_IN_SECS + 120)) "$CLIENT_BIN" \
        --tokenizer "$TOKENIZER_PATH" \
        --tokenizer-config "$TOKENIZER_CONFIG" \
        --endpoint "http://localhost:${ROUTER_PORT}/generate" \
        --api tgi \
        --dataset bailian \
        --dataset-path "${TRACES_DIR}/${TRACE_FILE}" \
        --scale-factor "$sf" \
        --time-in-secs "$TIME_IN_SECS" \
        --output-path "${out_dir}/results.jsonl" \
        --context-length "$CONTEXT_LENGTH" \
        > "${out_dir}/client_stdout.log" 2>&1 || true

    kill "$router_pid" 2>/dev/null || true
    wait "$router_pid" 2>/dev/null || true
    sleep 15
    free_port "$ROUTER_PORT"

    local total ok
    total=$(wc -l < "${out_dir}/results.jsonl" 2>/dev/null || echo 0)
    ok=$(grep -c '"status":"200"' "${out_dir}/results.jsonl" 2>/dev/null || echo 0)
    echo "  Done: $ok/$total (ok/total) -> ${out_dir}/results.jsonl"
}

echo "============================================"
echo " Sweep Test: $MODEL_NAME, 5GPU (host 3-7)"
echo " Trace=$TRACE_FILE, SF=${START_SF}..${MAX_SF} (delta=${DELTA_SF})"
echo " Time=${TIME_IN_SECS}s + 120s drain"
echo " Output: $OUTPUT_BASE"
echo "============================================"

SF="$START_SF"
while (( $(awk "BEGIN{print ($SF <= $MAX_SF) ? 1 : 0}") )); do
    echo ""
    echo "==================== SF=$SF ===================="
    for policy in vllm bailian aibrix dynamo-t1 dynamo-t2 lmetric preble; do
        run_one "$policy" "$SF" || echo "WARNING: $policy@SF=$SF failed"
    done
    SF=$(awk "BEGIN{printf \"%.1f\", $SF + $DELTA_SF}")
done

echo ""
echo "Sweep complete! Results: $OUTPUT_BASE"
