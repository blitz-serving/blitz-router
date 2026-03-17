# ======================
# Round Robin
# ======================
# cargo build -p router_v2 --features round-robin-q,vllm-backend,prefill_tokens_alert,request_in_queue_alert,batch_size_alert,request_num_alert,ngrok --no-default-features

# pkill -f vllm
# ./scripts/start_azure.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_rr_20min_3code_1.5conv_oldclient_v8
# mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_rr_20min_3code_1.5conv_oldclient_v8/vllm

# pkill -f vllm
# ./scripts/start_azure.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2_new_client.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_rr_20min_3code_1.5conv_newclient_v8
# mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_rr_20min_3code_1.5conv_newclient_v8/vllm

# pkill -f vllm
# ./scripts/start_azure_sim.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_rr_20min_3code_1.5conv_oldclient_v5
# mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_rr_20min_3code_1.5conv_oldclient_v5/sim

# pkill -f vllm
# ./scripts/start_azure_sim.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2_new_client.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_rr_20min_3code_1.5conv_newclient_v5
# mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_rr_20min_3code_1.5conv_newclient_v5/sim


======================
Join Shortest Queue
# ======================
cargo build -p router_v2 --features join-shortest-q,vllm-backend,prefill_tokens_alert,request_in_queue_alert,batch_size_alert,request_num_alert,ngrok --no-default-features

pkill -f vllm
./scripts/start_azure.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_jsq_20min_3code_1.5conv_oldclient_v7
mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_jsq_20min_3code_1.5conv_oldclient_v7/vllm

pkill -f vllm
./scripts/start_azure.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2_new_client.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_jsq_20min_3code_1.5conv_newclient_v7
mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_jsq_20min_3code_1.5conv_newclient_v7/vllm

# pkill -f vllm
# ./scripts/start_azure_sim.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_jsq_20min_3code_1.5conv_oldclient_v3
# # mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_jsq_20min_3code_1.5conv_oldclient_v3/sim

# pkill -f vllm
# ./scripts/start_azure_sim.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2_new_client.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_jsq_20min_3code_1.5conv_newclient_v4
# mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_jsq_20min_3code_1.5conv_newclient_v4/sim


# # ======================
# # Least Workload Queue
# # ======================
cargo build -p router_v2 --features least-work-q,vllm-backend,prefill_tokens_alert,request_in_queue_alert,batch_size_alert,request_num_alert,ngrok --no-default-features

pkill -f vllm
./scripts/start_azure.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_lwl_20min_3code_1.5conv_oldclient_v7
mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_lwl_20min_3code_1.5conv_oldclient_v7/vllm

pkill -f vllm
./scripts/start_azure.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2_new_client.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_lwl_20min_3code_1.5conv_newclient_v7
mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_lwl_20min_3code_1.5conv_newclient_v7/vllm

# pkill -f vllm
# ./scripts/start_azure_sim.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_lwl_20min_3code_1.5conv_oldclient_v3
# mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_lwl_20min_3code_1.5conv_oldclient_v3/sim

# pkill -f vllm
# ./scripts/start_azure_sim.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/lmetric_dp4_4_2_new_client.toml /nvme/zkx/blitz-infer-pack/config/dense_6simulator.toml modified_lwl_20min_3code_1.5conv_newclient_v4
# mv /nvme/lmetric/logs/vllm*.log /nvme/lmetric/logs/modified_lwl_20min_3code_1.5conv_newclient_v4/sim
