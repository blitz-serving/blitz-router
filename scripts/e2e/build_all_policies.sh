#!/bin/bash
set -euo pipefail

ROUTER_DIR="${ROUTER_DIR:-/workspace/blitz-router}"
REQUEST_SIM_DIR="${REQUEST_SIM_DIR:-/workspace/request-sim}"

declare -a POLICIES=(
  "join-shortest-q-weight"
  "bailian-impl-q"
  "aibrix-q"
  "dynamo-q"
  "dynamo-po-q"
  "lmetric-q"
  "preble-q"
)

ONLY="${1:-}"
LOG_FILE="${2:-/dev/stderr}"

cd "$ROUTER_DIR"

if [ -n "$ONLY" ]; then
    POLICIES=("$ONLY")
    echo "Building single policy: $ONLY"
else
    echo "Building ${#POLICIES[@]} router policies..."
fi
echo "Log: $LOG_FILE"
echo ""

FAILED=0
for policy in "${POLICIES[@]}"; do
    echo -n "  Building router_${policy}..."
    if cargo build -p router --release --features "$policy" \
         >> "$LOG_FILE" 2>&1; then
        cp target/release/router "target/release/router_${policy}"
        SIZE=$(stat -c%s "target/release/router_${policy}" 2>/dev/null || \
               stat -f%z "target/release/router_${policy}" 2>/dev/null || echo '?')
        echo " OK (${SIZE} bytes)"
    else
        echo " FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo -n "  Building request-sim..."
if (cd "$REQUEST_SIM_DIR" && cargo build --release) >> "$LOG_FILE" 2>&1; then
    echo " OK"
else
    echo " FAILED"
    FAILED=$((FAILED + 1))
fi

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All builds succeeded."
else
    echo "WARNING: $FAILED build(s) failed. Check $LOG_FILE"
    exit 1
fi
