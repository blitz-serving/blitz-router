#!/bin/bash

# Manual config
MODEL_PATH='/nvme/models/Meta-Llama-3-8B-Instruct'
# venv path with installed vllm
# VENV_PATH='/nvme/zdy/lib/uv/venv/24.06'
VENV_PATH='/nvme/zkx/modified-vllm/myenv'
# blitz-infer-pack project path
WORK_DIR='/nvme/zdy/workspace/blitz-infer-pack'
# skip launch backend
NO_BACKEND=false
# skip terminate backend
KEEP_BACKEND=false
# kip terminate session
KEEP_TMUX=false
# output base path
OUTPUT_BASE="/nvme/lmetric/logs"
OUTPUT_DIR="${OUTPUT_BASE}/$(date +%Y%m%d%H%M%S)"
# evaluation time in second
TIME_IN_SEC=120

# CLI config
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-backend)
            NO_BACKEND=true
            shift
        ;;
        --keep-backend)
            KEEP_BACKEND=true
            shift
        ;;
        --keep-tmux)
            KEEP_TMUX=true
            shift
        ;;
        --work-dir)
            WORK_DIR="$2"
            shift
            shift
        ;;
        --venv-path)
            VENV_PATH="$2"
            shift
            shift
        ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
        ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
        ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

# make sure configs are sufficient
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 [--no-backend] [--keep-backend] [--work-dir DIR] [--venv-path PATH] <backend-cfg> <router-cfg> <client-cfg>"
    exit 1
fi

CONFIG1="$1"
CONFIG2="$2"
CONFIG3="$3"

echo "Create output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR
cp $CONFIG1 $OUTPUT_DIR/backend.toml
cp $CONFIG2 $OUTPUT_DIR/router.toml
cp $CONFIG3 $OUTPUT_DIR/client.toml

# config file must exist
for config in "$CONFIG1" "$CONFIG2" "$CONFIG3"; do
    if [ ! -f "$config" ]; then
        echo "Error: Configuration file '$config' not found."
        exit 1
    fi
done

# kill previous tmux session of the same evaluation
if tmux has-session -t azure 2>/dev/null; then
    echo "Killing existing 'azure' tmux session..."
    
    # 向 azure 会话的每个窗口发送 Ctrl+C (中断信号)
    for pane in $(tmux list-panes -t azure -F '#{pane_id}'); do
        tmux send-keys -t "$pane" C-c
    done
    
    # waiting process to terminate
    sleep 2
    
    tmux kill-session -t azure
else
    echo "No existing 'azure' tmux session found."
fi

tmux new-session -d -s azure

# activate python venv
TMUX_CMD="source $VENV_PATH/bin/activate"

# skip launching backend
if [ "$NO_BACKEND" = false ]; then
    echo "launch vllm and waiting 120s..."
    tmux new-session -d -s vllm
    tmux send-keys -t vllm:0 "$TMUX_CMD && python $WORK_DIR/scripts/batchv3/smart_runner.py --toml $CONFIG1 --log-dir=$OUTPUT_BASE --output-dir=$OUTPUT_DIR --model-path=$MODEL_PATH --venv-path=$VENV_PATH --work-dir=$WORK_DIR" C-m
    sleep 120
fi

echo "launch router and waiting 10s..."
tmux new-window -t azure -n window2
tmux send-keys -t azure:window2 "$TMUX_CMD && python $WORK_DIR/scripts/batchv3/smart_runner.py --toml $CONFIG2 --log-dir=$OUTPUT_BASE --output-dir=$OUTPUT_DIR --model-path=$MODEL_PATH --venv-path=$VENV_PATH --work-dir=$WORK_DIR" C-m
sleep 10

echo "launch client..."
tmux new-window -t azure -n window3
tmux send-keys -t azure:window3 "$TMUX_CMD && python $WORK_DIR/scripts/batchv3/smart_runner.py --toml $CONFIG3 --log-dir=$OUTPUT_BASE --output-dir=$OUTPUT_DIR --model-path=$MODEL_PATH --venv-path=$VENV_PATH --work-dir=$WORK_DIR" C-m

echo "sleep ${TIME_IN_SEC}s..."
# sleep longer for pending requests to be processed
sleep $(($TIME_IN_SEC + 30))

echo "Merge client logs..."
# merge multiple jsonl files
cat $OUTPUT_DIR/client*.jsonl > $OUTPUT_DIR/client.jsonl

# echo "Plot Figures..."
${VENV_PATH}/bin/python $WORK_DIR/scripts/figures/final_figure.py --output-dir=$OUTPUT_DIR

# leverage tmux to propagate kill signal
if [ "$NO_BACKEND" = false -a "$KEEP_BACKEND" = false ]; then
    tmux send-keys -t vllm C-c
    tmux kill-session -t vllm
fi
if [ "$KEEP_TMUX" = false ]; then
    tmux kill-window -t azure:window2
    tmux kill-window -t azure:window3
else
    echo "You can attach using: tmux attach-session -t azure"
fi