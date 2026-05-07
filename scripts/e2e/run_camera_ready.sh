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
TIME_IN_SECS=180
CONFIG_STUBS="${ROUTER_DIR}/exps/blitz-run/configs/config-stubs-7gpu.json"
OUTPUT_BASE="/workspace/tmp/lmetric/e2e-camera-ready/$(date +%Y%m%d-%H%M%S)"

declare -A POLICY_BINS=(
  [vllm]="router_join-shortest-weight-q"
  [bailian]="router_bailian-impl-q"
  [aibrix]="router_aibrix-q"
  [dynamo-t1]="router_dynamo-q"
  [dynamo-t2]="router_dynamo-po-q"
  [lmetric]="router_lmetric-q"
  [preble]="router_preble-q"
)

TRACE_FILE="${1:-qwen_traceA_blksz_16.jsonl}"
SCALE_FACTOR="${2:-0.3}"
TRACE_NAME="${TRACE_FILE%.jsonl}"

mkdir -p "$OUTPUT_BASE"

free_port() {
    local pids
    pids=$(ss -tlnp | grep ":$1 " | grep -oP 'pid=\K[0-9]+' || true)
    for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
    sleep 1
}

run_one() {
    local policy="$1"
    local router_bin="${ROUTER_DIR}/target/release/${POLICY_BINS[$policy]}"
    local out_dir="${OUTPUT_BASE}/${policy}/sf${SCALE_FACTOR}_${TRACE_NAME}"
    mkdir -p "$out_dir"

    echo ""
    echo "======== Policy: $policy | Trace: $TRACE_NAME | SF: $SCALE_FACTOR ========"
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
        --scale-factor "$SCALE_FACTOR" \
        --time-in-secs "$TIME_IN_SECS" \
        --output-path "${out_dir}/results.jsonl" \
        --context-length "$CONTEXT_LENGTH" \
        > "${out_dir}/client_stdout.log" 2>&1 || true

    kill "$router_pid" 2>/dev/null || true
    wait "$router_pid" 2>/dev/null || true
    sleep 15
    free_port "$ROUTER_PORT"

    local count
    count=$(wc -l < "${out_dir}/results.jsonl" 2>/dev/null || echo 0)
    echo "  Done: $count results -> ${out_dir}/results.jsonl"
}

echo "============================================"
echo " Camera-Ready Experiment: $MODEL_NAME, 7GPU"
echo " Trace=$TRACE_FILE, Scale=$SCALE_FACTOR, Time=${TIME_IN_SECS}s"
echo " Output: $OUTPUT_BASE"
echo "============================================"

for policy in vllm bailian aibrix dynamo-t1 dynamo-t2 lmetric preble; do
    run_one "$policy" || echo "WARNING: $policy failed"
done

echo ""
echo "Experiment complete! Results: $OUTPUT_BASE"
echo "Scale factor: $SCALE_FACTOR"
