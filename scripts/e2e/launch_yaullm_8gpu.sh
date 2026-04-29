#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/nvme/models/Qwen3-30B-A3B}"
BASE_PORT="${BASE_PORT:-8100}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_WAIT="${MAX_WAIT:-600}"
LOG_DIR="${1:?Usage: $0 <log-dir>}"
PID_FILE="${2:-${LOG_DIR}/yaullm.pids}"

mkdir -p "$LOG_DIR"

# Free ports from any previous run
for i in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + i))
    fuser -k ${port}/tcp 2>/dev/null || true
    pids=$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' || true)
    for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
done
sleep 2

echo "Launching $NUM_GPUS yaullm instances..."
echo "  Model: $MODEL_PATH"
echo "  Ports: ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
echo "  GPU mem util: $GPU_MEM_UTIL"
echo "  Logs: $LOG_DIR"

> "$PID_FILE"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + i))
    echo "  Starting gpu${i} on port $port..."

    CUDA_VISIBLE_DEVICES=$i \
    VLLM_REPORT_METRICS=1 \
    VLLM_SSE_MODE=full \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$port" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enable-prefix-caching \
        --max-num-seqs 256 \
        --block-size 16 \
        --trust-remote-code \
        > "$LOG_DIR/gpu${i}.log" 2>&1 &
    echo "$!" >> "$PID_FILE"
done

echo ""
echo "Waiting for all engines to become healthy (max ${MAX_WAIT}s)..."

HEALTHY=0
ELAPSED=0
while [ "$HEALTHY" -lt "$NUM_GPUS" ] && [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
    HEALTHY=0
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        port=$((BASE_PORT + i))
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            HEALTHY=$((HEALTHY + 1))
        fi
    done
    if [ "$HEALTHY" -lt "$NUM_GPUS" ]; then
        printf "\r  %d/%d healthy (elapsed: %ds)..." "$HEALTHY" "$NUM_GPUS" "$ELAPSED"
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    fi
done
echo ""

if [ "$HEALTHY" -eq "$NUM_GPUS" ]; then
    echo "All $NUM_GPUS engines healthy."
    echo "PIDs: $(cat "$PID_FILE" | tr '\n' ' ')"
else
    echo "ERROR: Only $HEALTHY/$NUM_GPUS engines healthy after ${MAX_WAIT}s."
    echo "Check logs in $LOG_DIR"
    # Kill any that did start
    while read -r pid; do
        kill "$pid" 2>/dev/null || true
    done < "$PID_FILE"
    exit 1
fi
