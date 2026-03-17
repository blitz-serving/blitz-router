from plot import plot_main

import os


def traverse_logs(root_path, max_depth, current_depth=1):
    if current_depth > max_depth:
        return

    try:
        for entry in os.listdir(root_path):
            full_path = os.path.join(root_path, entry)
            if os.path.isdir(full_path):
                if current_depth == 5:
                    print(full_path)
                    plot_main(
                        f"{full_path}/router.log",
                        f"{full_path}/client.jsonl",
                        f"{full_path}/fig.pdf",
                    )
                traverse_logs(full_path, max_depth, current_depth + 1)
    except PermissionError:
        pass


if __name__ == "__main__":
    # log_dir = f"/nvme/zdy/blitz-remake/log_home/202505082114_e2e_eval/llama3_8b-AzureCode2023-5min-e2e_blitz"
    # log_dir = f"/nvme/zdy/blitz-remake/log_home/202505091129_e2e_eval/mistral_24b-AzureConv2023-8min-e2e_blitz"
    # log_dir = f"/nvme/zdy/blitz-remake/tmp/benchmark/mistral_24b_net"
    # log_dir = f"/nvme/zdy/blitz-remake/tmp/benchmark/llama3_8b_net"
    log_dir = f"/nvme/zdy/blitz-remake/tmp/benchmark/llama3_8b_fast"
    # log_dir = f"/nvme/zdy/blitz-remake/tmp/benchmark/mistral_24b_fast"
    with open(f"{log_dir}/client_prologue.log", "r") as f:
        prologue = f.read()
    with open(f"{log_dir}/router.log", "a") as f:
        f.write(prologue)
    print(f"Plotting {log_dir}/fig.pdf")
    plot_main(
        f"{log_dir}/router.log",
        f"{log_dir}/client.jsonl",
        f"{log_dir}/fig.pdf",
    )
