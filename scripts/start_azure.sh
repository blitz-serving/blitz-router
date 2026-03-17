#!/bin/bash


# ./start_azure.sh /mnt/debugger/hjb/blitz-infer-pack/config/dense_vllm_dp4.toml /mnt/debugger/hjb/blitz-infer-pack/config/dense_router.toml /mnt/debugger/hjb/blitz-infer-pack/config/dense_clients.toml


# 检查是否提供了三个参数
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <config1> <config2> <config3>"
    exit 1
fi

CONFIG1="$1"
CONFIG2="$2"
CONFIG3="$3"

OUTPUT_BASE="/nvme/lmetric/logs"
OUTPUT_DIR="${OUTPUT_BASE}/$(date +%Y%m%d%H%M%S)"

# 创建输出目录
echo "Create output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR
cp $CONFIG1 $OUTPUT_DIR/config1.toml
cp $CONFIG2 $OUTPUT_DIR/config2.toml
cp $CONFIG3 $OUTPUT_DIR/config3.toml


# 检查配置文件是否存在
for config in "$CONFIG1" "$CONFIG2" "$CONFIG3"; do
    if [ ! -f "$config" ]; then
        echo "Error: Configuration file '$config' not found."
        exit 1
    fi
done

# 检查是否存在名为 azure 的 tmux 会话
if tmux has-session -t azure 2>/dev/null; then
    echo "Killing existing 'azure' tmux session..."

    # 向 azure 会话的每个窗口发送 Ctrl+C (中断信号)
    for pane in $(tmux list-panes -t azure -F '#{pane_id}'); do
        tmux send-keys -t "$pane" C-c
    done

    # 等待一段时间确保进程终止
    sleep 2

    # 关闭整个 session
    tmux kill-session -t azure
else
    echo "No existing 'azure' tmux session found."
fi

# 创建新的 tmux 会话 'azure'，不附着
tmux new-session -d -s azure
tmux new-window -t azure -n window1

# 设置源环境
TMUX_CMD="source /nvme/zkx/modified-vllm/myenv/bin/activate"

# echo "create vllm and waiting 120s(python /nvme/zkx/blitz-infer-pack/scripts/batchv3/smart_runner.py --toml $CONFIG1 --output_dir=$OUTPUT_DIR)..."
# # 第一个窗口：运行 config1 并 sleep 120s
# tmux send-keys -t azure:window1 "$TMUX_CMD && python /nvme/zkx/blitz-infer-pack/scripts/batchv3/smart_runner.py --toml $CONFIG1 --output_dir=$OUTPUT_DIR" C-m

# sleep 120

echo "create router and waiting 10s(python /nvme/zkx/blitz-infer-pack/scripts/batchv3/smart_runner.py --toml $CONFIG2 --output_dir=$OUTPUT_DIR)..."
# 第二个窗口：运行 config2 并 sleep 30s
tmux new-window -t azure -n window2
tmux send-keys -t azure:window2 "$TMUX_CMD && python /nvme/zkx/blitz-infer-pack/scripts/batchv3/smart_runner.py --toml $CONFIG2 --output_dir=$OUTPUT_DIR" C-m

sleep 10

echo "create client now!(python /nvme/zkx/blitz-infer-pack/scripts/batchv3/smart_runner.py --toml $CONFIG3 --output_dir=$OUTPUT_DIR)..."
# 第三个窗口：运行 config3
tmux new-window -t azure -n window3
tmux send-keys -t azure:window3 "$TMUX_CMD && python /nvme/zkx/blitz-infer-pack/scripts/batchv3/smart_runner.py --toml $CONFIG3 --output_dir=$OUTPUT_DIR" C-m

echo "sleep 120s..."
sleep 150
echo "Merge client logs..."
# 将$OUTPUT_DIR下所有client*.jsonl合并成为一个文件
cat $OUTPUT_DIR/client*.jsonl > $OUTPUT_DIR/client.jsonl


echo "Paint Figures..."
/nvme/zkx/modified-vllm/myenv/bin/python /nvme/zkx/blitz-infer-pack/scripts/figures/final_figure.py --output_dir=$OUTPUT_DIR/

echo "Successfully started 'azure' tmux session with 3 windows." echo "You can attach using: tmux attach-session -t azure"
tmux kill-session -t azure