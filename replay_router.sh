#!/bin/bash

for i in {1..6}
do
  echo "Running router_index=$i"
  python ./scripts/batchv3/smart_runner.py \
    --toml ./config/dense_6simulator_replay.toml \
    --router_index=$i & 
done

# mkdir -p /nvme/lmetric/logs/modified_rr_20min_3code_1.5conv_newclient_v4/replay
# mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_rr_20min_3code_1.5conv_newclient_v4/replay