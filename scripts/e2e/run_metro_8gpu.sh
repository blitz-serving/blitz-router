#!/bin/bash
set -euo pipefail

#############################################
# Metro 8-GPU DP8 Validation — Main Orchestrator
#
# Usage:
#   ./run_metro_8gpu.sh [TRACE_FILE]
#
# Runs lmetric policy at SF=3.0 and SF=3.5 with full
# yaullm restart between runs (clean state).
#
# Unattended execution:
#   nohup ./run_metro_8gpu.sh > sweep.log 2>&1 &
#############################################

# ==================== Configuration ====================
MODEL_NAME="/nvme/models/Qwen3-30B-A3B"
MODEL_PATH="/nvme/models/Qwen3-30B-A3B"
TOKENIZER_PATH="${MODEL_PATH}/tokenizer.json"
TOKENIZER_CONFIG="${MODEL_PATH}/tokenizer_config.json"
TRACES_DIR="/workspace/tmp/bailian-traces"
ROUTER_DIR="/workspace/blitz-router"
CLIENT_BIN="/workspace/request-sim/target/release/request-sim"
SCRIPT_DIR="${ROUTER_DIR}/scripts/e2e"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify_staleness.py"
ROUTER_PORT=58009
CONTEXT_LENGTH=40960
ENGINE_PORT_SETS=(8100 8200)  # alternating port ranges between runs
NUM_GPUS=8
POLICY="lmetric-q"
ROUTER_BIN="${ROUTER_DIR}/target/release/router_${POLICY}"

TRACE_FILE="${1:-qwen_traceA_blksz_16.jsonl}"
TRACE_NAME="${TRACE_FILE%.jsonl}"

SCALE_FACTORS=(3.0 3.5)

NO_BUILD="${NO_BUILD:-false}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_BASE="/workspace/tmp/lmetric/metro-8gpu-validation/${RUN_ID}"

declare -A RESULTS_PASS
declare -A RESULTS_COUNT
declare -A RESULTS_TIME
TOTAL_PASS=0
TOTAL_FAIL=0

mkdir -p "$OUTPUT_BASE"

# ==================== Utility Functions ====================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

free_port() {
    local pids
    pids=$(ss -tlnp 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' || true)
    for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
    sleep 1
}

kill_yaullm() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        while read -r pid; do
            kill "$pid" 2>/dev/null || true
        done < "$pid_file"
        sleep 3
        while read -r pid; do
            kill -9 "$pid" 2>/dev/null || true
        done < "$pid_file"
        sleep 2
    fi
    for base_port in "${ENGINE_PORT_SETS[@]}"; do
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            free_port $((base_port + i))
        done
    done
}

# ==================== Phase 0: Pre-flight ====================

phase0_preflight() {
    log "========== Phase 0: Pre-flight Checks =========="
    local errors=0

    log "  Checking GPUs..."
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    if [ "$gpu_count" -lt "$NUM_GPUS" ]; then
        log "  ERROR: Need $NUM_GPUS GPUs, found $gpu_count"
        errors=$((errors + 1))
    else
        log "  OK: $gpu_count GPUs available"
    fi

    log "  Checking model..."
    if [ -f "$TOKENIZER_PATH" ]; then
        log "  OK: Tokenizer found at $TOKENIZER_PATH"
    else
        log "  ERROR: Tokenizer not found at $TOKENIZER_PATH"
        errors=$((errors + 1))
    fi

    log "  Checking trace..."
    if [ -f "${TRACES_DIR}/${TRACE_FILE}" ]; then
        local lines
        lines=$(wc -l < "${TRACES_DIR}/${TRACE_FILE}")
        log "  OK: Trace has $lines entries"
    else
        log "  ERROR: Trace not found at ${TRACES_DIR}/${TRACE_FILE}"
        errors=$((errors + 1))
    fi

    log "  Checking verify_staleness.py..."
    if [ -f "$VERIFY_SCRIPT" ]; then
        log "  OK"
    else
        log "  ERROR: $VERIFY_SCRIPT not found"
        errors=$((errors + 1))
    fi

    if [ $errors -gt 0 ]; then
        log "  ABORT: $errors pre-flight check(s) failed"
        exit 1
    fi
    log "  All pre-flight checks passed."
}

# ==================== Phase 1: Build ====================

phase1_build() {
    log "========== Phase 1: Build =========="
    local build_log="${OUTPUT_BASE}/build.log"

    if [ "$NO_BUILD" = "true" ]; then
        log "  Skipping build (NO_BUILD=true)"
        if [ ! -x "$ROUTER_BIN" ]; then
            log "  ERROR: Missing binary: $ROUTER_BIN"
            exit 1
        fi
        if [ ! -x "$CLIENT_BIN" ]; then
            log "  ERROR: Missing binary: $CLIENT_BIN"
            exit 1
        fi
        log "  Binaries verified."
        return
    fi

    log "  Building lmetric router + request-sim..."
    log "  Build log: $build_log"

    "${SCRIPT_DIR}/build_all_policies.sh" "$POLICY" "$build_log"

    log "  Build phase complete."
}

# ==================== Phase 3: Per-Run Execution ====================

run_one() {
    local sf="$1"
    local run_idx="$2"
    local run_key="sf${sf}"
    local out_dir="${OUTPUT_BASE}/lmetric/${run_key}_${TRACE_NAME}"
    local vllm_log_dir="${out_dir}/vllm"
    local pid_file="${vllm_log_dir}/yaullm.pids"
    mkdir -p "$out_dir" "$vllm_log_dir"

    # Alternate port set between runs to avoid TIME_WAIT conflicts
    local base_port="${ENGINE_PORT_SETS[$((run_idx % ${#ENGINE_PORT_SETS[@]}))]}"

    # Generate config-stubs on the fly for this port range
    local config_file="${out_dir}/config-stubs.json"
    printf '[\n' > "$config_file"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        local port=$((base_port + i))
        [ $i -gt 0 ] && printf ',\n' >> "$config_file"
        printf '  "http://localhost:%d"' "$port" >> "$config_file"
    done
    printf '\n]\n' >> "$config_file"

    log ""
    log "================================================================"
    log "  Run: lmetric | SF=$sf | Trace=$TRACE_NAME | Ports=$base_port-$((base_port + NUM_GPUS - 1))"
    log "  Output: $out_dir"
    log "================================================================"

    # --- Launch yaullm ---
    log "  Launching $NUM_GPUS yaullm instances..."
    BASE_PORT="$base_port" "${SCRIPT_DIR}/launch_yaullm_8gpu.sh" "$vllm_log_dir" "$pid_file"

    # --- Launch router ---
    free_port "$ROUTER_PORT"

    local start_ts
    start_ts=$(date +%s)

    LOG_LEVEL="info,cache_tracking=info" \
    "$ROUTER_BIN" \
        --port "$ROUTER_PORT" \
        --client-config "$config_file" \
        --model-name "$MODEL_NAME" \
        --tokenizer-name "$MODEL_PATH" \
        --use-tokenizer \
        --max-input-length "$((CONTEXT_LENGTH - 1))" \
        --max-total-tokens "$CONTEXT_LENGTH" \
        --max-batch-prefill-tokens 81920 \
        --max-concurrent-requests 4096 \
        --kvcache-block-size 16 \
        --log-path "${out_dir}/router.log" \
        > "${out_dir}/router_stdout.log" 2>&1 &
    local router_pid=$!
    sleep 8

    if ! kill -0 "$router_pid" 2>/dev/null; then
        log "  ERROR: Router failed to start!"
        tail -5 "${out_dir}/router_stdout.log"
        RESULTS_PASS[$run_key]="FAIL(launch)"
        RESULTS_COUNT[$run_key]="0/0"
        RESULTS_TIME[$run_key]="0s"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        kill_yaullm "$pid_file"
        return 1
    fi
    log "  Router PID=$router_pid"

    # --- Run request-sim ---
    log "  Running request-sim (SF=$sf, trace-replay)..."
    "$CLIENT_BIN" \
        --tokenizer "$TOKENIZER_PATH" \
        --tokenizer-config "$TOKENIZER_CONFIG" \
        --endpoint "http://localhost:${ROUTER_PORT}/v1/chat/completions" \
        --api openai \
        --model-name "$MODEL_NAME" \
        --dataset bailian \
        --dataset-path "${TRACES_DIR}/${TRACE_FILE}" \
        --mode trace-replay \
        --scale-factor "$sf" \
        --output-path "${out_dir}/results.jsonl" \
        --context-length "$CONTEXT_LENGTH" \
        > "${out_dir}/client_stdout.log" 2>&1 || true

    # --- Teardown ---
    log "  Stopping router..."
    kill "$router_pid" 2>/dev/null || true
    wait "$router_pid" 2>/dev/null || true
    free_port "$ROUTER_PORT"

    log "  Stopping yaullm instances..."
    kill_yaullm "$pid_file"

    local end_ts
    end_ts=$(date +%s)
    local elapsed=$((end_ts - start_ts))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))

    # --- Count results ---
    local total ok
    total=$(wc -l < "${out_dir}/results.jsonl" 2>/dev/null || echo 0)
    ok=$(grep -c '"status":"200"' "${out_dir}/results.jsonl" 2>/dev/null || echo 0)
    log "  Requests: $ok/$total OK (${elapsed_min}m${elapsed_sec}s elapsed)"

    RESULTS_COUNT[$run_key]="$ok/$total"
    RESULTS_TIME[$run_key]="${elapsed_min}m${elapsed_sec}s"

    # --- Staleness validation ---
    if [ -f "${out_dir}/router.log" ] && [ -s "${out_dir}/router.log" ]; then
        log "  Running staleness validation..."
        if python3 "$VERIFY_SCRIPT" "${out_dir}/router.log" \
             > "${out_dir}/validation.txt" 2>&1; then
            log "  STALENESS: ALL PASS"
            RESULTS_PASS[$run_key]="PASS"
            TOTAL_PASS=$((TOTAL_PASS + 1))
        else
            log "  STALENESS: FAILED — see ${out_dir}/validation.txt"
            RESULTS_PASS[$run_key]="FAIL(staleness)"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    else
        log "  WARNING: No router.log found, skipping validation"
        RESULTS_PASS[$run_key]="FAIL(no-log)"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# ==================== Phase 4: Report ====================

phase4_report() {
    log ""
    log "========== Phase 4: Report =========="

    local summary="${OUTPUT_BASE}/summary.txt"
    {
        echo "============================================"
        echo " Metro 8-GPU DP8 Validation — Summary"
        echo " Run ID: $RUN_ID"
        echo " Date: $(date)"
        echo " Model: $MODEL_NAME"
        echo " GPUs: $NUM_GPUS"
        echo " Policy: lmetric"
        echo " Trace: $TRACE_FILE"
        echo "============================================"
        echo ""
        printf "%-10s %-6s %-18s %-12s %-10s\n" "Policy" "SF" "Staleness" "Requests" "Time"
        printf "%-10s %-6s %-18s %-12s %-10s\n" "------" "--" "---------" "--------" "----"
        for sf in "${SCALE_FACTORS[@]}"; do
            local key="sf${sf}"
            printf "%-10s %-6s %-18s %-12s %-10s\n" \
                "lmetric" \
                "$sf" \
                "${RESULTS_PASS[$key]:-N/A}" \
                "${RESULTS_COUNT[$key]:-0/0}" \
                "${RESULTS_TIME[$key]:-0s}"
        done
        echo ""
        echo "Result: $TOTAL_PASS PASS, $TOTAL_FAIL FAIL out of ${#SCALE_FACTORS[@]} runs"
        echo ""
        echo "Output directory: $OUTPUT_BASE"
    } | tee "$summary"
}

# ==================== Main ====================

main() {
    log "============================================"
    log " Metro 8-GPU DP8 Validation"
    log " Output: $OUTPUT_BASE"
    log "============================================"
    echo ""

    phase0_preflight
    echo ""
    phase1_build
    echo ""

    local run_idx=0
    for sf in "${SCALE_FACTORS[@]}"; do
        run_one "$sf" "$run_idx" || true
        run_idx=$((run_idx + 1))
    done

    phase4_report

    log ""
    if [ $TOTAL_FAIL -eq 0 ]; then
        log "ALL RUNS PASSED STALENESS VALIDATION"
    else
        log "$TOTAL_FAIL RUN(S) FAILED"
    fi

    [ $TOTAL_FAIL -eq 0 ]
}

main "$@" 2>&1 | tee "${OUTPUT_BASE}/full.log"
