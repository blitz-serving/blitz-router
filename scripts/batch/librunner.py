from datetime import datetime
import pytz
import subprocess
import shutil
import os
import threading
import time
import re
import json

# import generate_pb2
# import grpc
# import debug.generate_pb2_grpc as generate_pb2_grpc

global_args = {
    "workdir": "/nvme/blitz/log_home",
    "model_path": "/nvme/blitz/model/Llama-2-7b-hf",
    "model_name": "llama2_7b",
    "router_features": "ngrok,sched_naive",
    "scale_factor": 1.0,
    "grpc_base_port": 50051,
    "router_port": 11236,
    "dataset_path": "/nvme/blitz/processed-dataset/AzureCode2023.csv",
    "dataset_type": "processed",
    "client_request_time_in_secs": 180,
    "num_total_blocks": 8000,
    "max_blocks_per_replica": 8000,
    "tokens_prefilled_per_sec": 13000,
    "tokens_transferred_per_sec": 20000,
    "prefill_lower_bound": 0.15,
    "prefill_upper_bound": 0.4,
    "decode_lower_bound": 0.45,
    "decode_upper_bound": 0.8,
    "scale_down_threshold_millis": 1500,
    "migration_lower_bound": 0.0,
    "migration_upper_bound": 0.4,
    "num_hidden_layers": 32,
    "num_gpus_per_node": 32,
    "mock_load_millis": 0,
    "mock_transfer_millis": 0,
    "tensor_parallel_size": 1,
    "max_prefill_num": 3,
    "max_decode_num": 3,
    "min_prefill_num": 1,
    "min_decode_num": 1,
    "init_prefill_num": 1,
    "init_decode_num": 1,
    "cuda_devices": "4,5,6,7",
    "router_config": {
        "init_states": ["Prefill", "Decode", "Inactive", "Inactive"],
        "replicas": [[0], [1], [2], [3]],
    },
    "server_config": {
        "init_states": ["Prefill", "Prefill", "Prefill", "Prefill"],
        "replicas": [[0], [1], [2], [3]],
    },
    "inter_node_server": False,
    "inter_node_config": {
        "gg0021": "172.16.10.21",
        "gg0027": "172.16.10.27",
    },
    "ibv_rate": 0,
}

valid_args = set(global_args.keys())
# server = None


# def reset_server_state():
#     server_conf = get_stubs_config()
#     index = 0
#     for server in server_conf:
#         with grpc.insecure_channel(server) as channel:
#             stub = generate_pb2_grpc.TextGenerationServiceStub(channel)
#             stub.ResetState(generate_pb2.ResetStateRequest())
#             init_state = global_args["router_config"]["init_states"][index]
#             if init_state != "Inactive":
#                 stub.SetStatusReady(generate_pb2.SetStatusReadyRequest())
#         index += 1


def get_abspath(path) -> str:
    if os.path.islink(path) and os.path.isdir(os.readlink(path)):
        return os.path.abspath(os.readlink(path))
    elif os.path.isdir(path):
        return os.path.abspath(path)
    else:
        raise FileNotFoundError(f"{path} is not a valid directory")


def get_profiled_hyperparams(model_name):
    if model_name == "llama2_7b":
        return {
            "tokens_prefilled_per_sec": 13000,
            "tokens_transferred_per_sec": 30000,
            "num_hidden_layers": 32,
            "num_available_blocks": {
                "Full": 8000,
                "Half": 3000,
            },
        }
    elif model_name == "mistral_24b":
        return {
            "tokens_prefilled_per_sec": 6000,
            "tokens_transferred_per_sec": 40000,
            "num_hidden_layers": 40,
            "num_available_blocks": {
                "Full": 8000,
                "Half": 3000,
            },
        }
    elif model_name == "llama3_8b":
        return {
            "tokens_prefilled_per_sec": 12000,
            "tokens_transferred_per_sec": 60000,
            "num_hidden_layers": 32,
            "num_available_blocks": {
                "Full": 30000,
                "Half": 10000,
            },
        }
    elif model_name == "llama2_13b":
        return {
            "tokens_prefilled_per_sec": 9000,
            "tokens_transferred_per_sec": 20000,
            "num_hidden_layers": 40,
            "num_available_blocks": {
                "Full": 4000,
                "Half": 900,
            },
        }
    elif model_name == "mistral_13b":
        return {
            "tokens_prefilled_per_sec": 7000,
            "tokens_transferred_per_sec": 60000,
            "num_hidden_layers": 60,
            "num_available_blocks": {
                "Full": 13000,
                "Half": 3000,
            },
        }
    elif model_name == "llama2_70b":
        return {
            "tokens_prefilled_per_sec": 7500,
            "tokens_transferred_per_sec": 40000,
            "num_hidden_layers": 80,
            "num_available_blocks": {
                "Full": 32000,
                "Half": -1,
            },
        }
    elif model_name == "vit-large":
        return {
            "tokens_prefilled_per_sec": 13000,
            "tokens_transferred_per_sec": 30000,
            "num_hidden_layers": 32,
            "num_available_blocks": {
                "Full": 8000,
                "Half": 3000,
            },
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def set_hyperparams(model_name, memory_available):
    if memory_available not in {"Full", "Half"}:
        raise ValueError(f"Choose memory_available from {'Full', 'Half'}")
    hyper_params = get_profiled_hyperparams(model_name)
    set_args("tokens_prefilled_per_sec", hyper_params["tokens_prefilled_per_sec"])
    set_args("tokens_transferred_per_sec", hyper_params["tokens_transferred_per_sec"])
    set_args("num_hidden_layers", hyper_params["num_hidden_layers"])
    num_available_blocks = hyper_params.get("num_available_blocks")[memory_available]
    set_args("max_blocks_per_replica", num_available_blocks)
    set_args("num_total_blocks", num_available_blocks)


def set_args(key: str, val):
    if key in valid_args:
        global_args[key] = val
    else:
        raise Exception("Invalid arg: " + key)


def get_stubs_config():
    stubs = []
    slots = len(global_args["cuda_devices"].split(","))
    if global_args["inter_node_server"]:
        for _, ip in global_args["inter_node_config"].items():
            for i in range(0, slots):
                port = global_args["grpc_base_port"] + i
                stubs.append(f"http://{ip}:{port}")
    else:
        for i in range(0, slots):
            port = global_args["grpc_base_port"] + i
            stubs.append(f"http://localhost:{port}")
    return stubs


def get_extra_env() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": global_args["cuda_devices"],
        "RUST_BACKTRACE": "full",
        "LOG_LEVEL": "info",
    }


def get_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(get_extra_env())
    return env


def build_client_command() -> list[str]:
    return [
        "cargo",
        "build",
        "--release",
        "--package",
        "request-sim",
        "--bin",
        "client",
    ]


def build_router_command() -> list[str]:
    return [
        "cargo",
        "build",
        "--release",
        "--package",
        "router_v2",
        "--no-default-features",
        "--features",
        global_args["router_features"],
    ]


def dump_hostfile():
    slots = len(global_args["cuda_devices"].split(","))
    with open(f"{global_args['workdir']}/hostfile.txt", "w") as f:
        for hostname, _ in global_args["inter_node_config"].items():
            f.write(f"{hostname} slots={slots} max_slots={slots}\n")


def run_single_node_command():
    return [
        os.path.abspath("./build/release/bin/run_server_disaggregative"),
        "-T",
        f"{global_args['model_path']}/tokenizer.json",
        "-V",
        "''",
        "--host",
        "0.0.0.0",
        "--port",
        str(global_args["grpc_base_port"]),
        "--model-path",
        "''",
        "--model-name",
        global_args["model_name"],
        "-P",
        "fp16",
        "--config",
        f"{global_args['workdir']}/config-server.json",
        "--num-total-blocks",
        str(global_args["num_total_blocks"]),
        "--ibv-rate",
        str(global_args["ibv_rate"]),
    ]


def run_inter_node_server_command() -> list[str]:
    return [
        "mpirun",
        "--allow-run-as-root",
        "--hostfile",
        os.path.abspath(f"{global_args['workdir']}/hostfile.txt"),
        "--map-by",
        "slot",
        "--mca",
        "btl_tcp_if_include=bond0",
        str(
            len(global_args["inter_node_config"])
            * len(global_args["cuda_devices"].split(","))
        ),
        "bash",
        os.path.abspath(f"{global_args['workdir']}/run-single-node.sh"),
    ]


def run_intra_server_command() -> list[str]:
    return [
        "mpirun",
        "-n",
        str(len(global_args["cuda_devices"].split(","))),
        "--allow-run-as-root",
        os.path.abspath("./build/release/bin/run_server_disaggregative"),
        "-T",
        f"{global_args['model_path']}/tokenizer.json",
        "-V",
        "''",
        "--host",
        "localhost",
        "--port",
        str(global_args["grpc_base_port"]),
        "--model-path",
        "''",
        "--model-name",
        global_args["model_name"],
        "-P",
        "fp16",
        "--config",
        f"{global_args['workdir']}/config-server.json",
        "--num-total-blocks",
        str(global_args["num_total_blocks"]),
        "-TP",
        str(global_args["tensor_parallel_size"]),
        "--ibv-rate",
        str(global_args["ibv_rate"]),
    ]


def run_router_command() -> list[str]:
    return [
        "./target/release/router_v2",
        "--hostname",
        "localhost",
        "--port",
        str(global_args["router_port"]),
        "--use-tokenizer",
        "--deployment",
        "disaggregation",
        "--tokenizer-name",
        global_args["model_path"],
        "--client-config",
        f"{global_args['workdir']}/config-stubs.json",
        "--deployment-config-path",
        f"{global_args['workdir']}/config-router.json",
        "--log-path",
        f"{global_args['workdir']}/router.log",
        "--max-input-length",
        "4090",
        "--max-total-tokens",
        "4096",
        "--max-concurrent-requests",
        "4096",
        "--tokens-prefilled-per-sec",
        str(global_args["tokens_prefilled_per_sec"]),
        "--tokens-transferred-per-sec",
        str(global_args["tokens_transferred_per_sec"]),
        "--max-blocks-per-replica",
        str(global_args["max_blocks_per_replica"]),
        "--max-prefill-num",
        str(global_args["max_prefill_num"]),
        "--max-decode-num",
        str(global_args["max_decode_num"]),
        "--min-prefill-num",
        str(global_args["min_prefill_num"]),
        "--min-decode-num",
        str(global_args["min_decode_num"]),
        "--prefill-lower-bound",
        str(global_args["prefill_lower_bound"]),
        "--prefill-upper-bound",
        str(global_args["prefill_upper_bound"]),
        "--decode-lower-bound",
        str(global_args["decode_lower_bound"]),
        "--decode-upper-bound",
        str(global_args["decode_upper_bound"]),
        "--migration-lower-bound",
        str(global_args["migration_lower_bound"]),
        "--migration-upper-bound",
        str(global_args["migration_upper_bound"]),
        "--scale-down-threshold-millis",
        str(global_args["scale_down_threshold_millis"]),
        "--num-hidden-layers",
        str(global_args["num_hidden_layers"]),
        "--num-gpus-per-node",
        str(global_args["num_gpus_per_node"]),
        "--mock-load-millis",
        str(global_args["mock_load_millis"]),
        "--mock-transfer-millis",
        str(global_args["mock_transfer_millis"]),
        "--tensor-parallel-size",
        str(global_args["tensor_parallel_size"]),
    ]


def run_client_command() -> list[str]:
    return [
        "./target/release/client",
        "--tokenizer",
        f"{global_args['model_path']}/tokenizer.json",
        "--endpoint",
        f"http://localhost:{global_args['router_port']}/generate",
        "--protocol",
        "st",
        "--replay-mode",
        "--scale-factor",
        str(global_args["scale_factor"]),
        "--dataset-type",
        global_args["dataset_type"],
        "--dataset-path",
        global_args["dataset_path"],
        "--time-in-secs",
        str(global_args["client_request_time_in_secs"]),
        "--truncate",
        "4095",
        "--output-path",
        os.path.join(f"{global_args['workdir']}", "client.jsonl"),
    ]


def build():
    build_client_result = subprocess.run(
        build_client_command(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert build_client_result.returncode == 0, "Failed to build client"

    build_router_result = subprocess.run(
        build_router_command(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert build_router_result.returncode == 0, "Failed to build router"


def remove_ansi_sequences(text) -> str:
    ansi_escape = re.compile(r"\x1b\[([0-9;]*m)")
    return ansi_escape.sub("", text)


def monitor_server_process(stdout, log_file, event, warm_up_event):
    with open(log_file, "w") as f:
        for line in stdout:
            f.write(line)
            f.flush()
            if "Start gRPC server" in line:
                event.set()
            if "Warmup with num tokens" in line:
                warm_up_event.set()


def monitor_client_process(stdout):
    log_file = f"{global_args['workdir']}/client.log"
    prologue_file = f"{global_args['workdir']}/client_prologue.txt"
    with open(log_file, "w") as f:
        for line in stdout:
            f.write(remove_ansi_sequences(line))
            f.flush()
            if "Client start" in line:
                prologue = open(prologue_file, "w")
                prologue.write(remove_ansi_sequences(line))
                prologue.flush()
                prologue.close()


def get_git_info() -> str:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode("utf-8")
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode("utf-8")
        modified_files = subprocess.check_output(
            ["git", "status", "--porcelain"]
        ).decode("utf-8")
        return f"Branch: {branch}\nCommit: {commit}\nStatus:\n{modified_files}"
    except subprocess.CalledProcessError as e:
        return "Error occurred while getting git info"


def get_diff() -> str:
    try:
        result = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return "Error occurred while getting git diff"


def copy_untracked_files(destination):
    try:
        if not os.path.exists(destination):
            os.makedirs(destination)
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
        )
        untracked_files = result.stdout.splitlines()

        for file_path in untracked_files:
            # Create the destination path
            dest_path = os.path.join(destination, file_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            # Copy the file to the destination
            shutil.copy2(file_path, dest_path)
    except Exception as _:
        pass


def run_with_retry(tag) -> bool:
    if run(tag):
        return True
    else:
        print(f"[{get_current_timestamp()}] [{tag}] Failed. Retry schedued", flush=True)
        return run(tag)


def run(tag) -> bool:
    status = True
    server = None
    router = None
    client = None
    keyboard_interrupted = False

    export_lines = [f'export {key}="{value}"' for key, value in get_extra_env().items()]

    try:
        print(f"[{get_current_timestamp()}] [{tag}] Dumping configuration files")

        with open(f"{global_args['workdir']}/git-info.txt", "w") as f:
            f.write(get_git_info())
        with open(f"{global_args['workdir']}/git-diff.patch", "w") as f:
            f.write(get_diff())
        copy_untracked_files(f"{global_args['workdir']}/untracked")

        with open(f"{global_args['workdir']}/config-router.json", "w") as json_file:
            json.dump(
                global_args["router_config"], json_file, indent=4, ensure_ascii=False
            )
        with open(f"{global_args['workdir']}/config-server.json", "w") as json_file:
            json.dump(
                global_args["server_config"], json_file, indent=4, ensure_ascii=False
            )
        with open(f"{global_args['workdir']}/config-stubs.json", "w") as json_file:
            json.dump(get_stubs_config(), json_file, indent=4, ensure_ascii=False)

        with open(f"{global_args['workdir']}/build-client.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write(" ".join(build_client_command()))
            f.write("\n")
        with open(f"{global_args['workdir']}/build-router.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write(" ".join(build_router_command()))
            f.write("\n")

        if global_args["inter_node_server"]:
            dump_hostfile()
            single_node_command = run_single_node_command()
            with open(f"{global_args['workdir']}/run-single-node.sh", "w") as file:
                file.write("#!/bin/bash\n")
                file.write("\n".join(export_lines))
                file.write("\n")
                file.write(" ".join(single_node_command))
                file.write("\n")
            server_command = run_inter_node_server_command()
            time.sleep(5)
        else:
            server_command = run_intra_server_command()
        with open(f"{global_args['workdir']}/run-server.sh", "w") as file:
            file.write("#!/bin/bash\n")
            file.write("\n".join(export_lines))
            file.write("\n")
            file.write(" ".join(server_command))
            file.write("\n")

        router_command = run_router_command()
        with open(f"{global_args['workdir']}/run-router.sh", "w") as file:
            file.write("#!/bin/bash\n")
            file.write("\n".join(export_lines))
            file.write("\n")
            file.write(" ".join(router_command))
            file.write("\n")

        client_command = run_client_command()
        with open(f"{global_args['workdir']}/run-client.sh", "w") as file:
            file.write("#!/bin/bash\n")
            file.write("\n".join(export_lines))
            file.write("\n")
            file.write(" ".join(client_command))
            file.write("\n")

        print(f"[{get_current_timestamp()}] [{tag}] Building...")
        env = get_env()
        build()

        print(f"[{get_current_timestamp()}] [{tag}] Running...")

        server = subprocess.Popen(
            server_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        server_start_event = threading.Event()
        warm_up_event = threading.Event()
        monitor_server_thread = threading.Thread(
            target=monitor_server_process,
            args=(
                server.stdout,
                f"{global_args['workdir']}/server.log",
                server_start_event,
                warm_up_event,
            ),
        )
        monitor_server_thread.start()
        if server_start_event.wait(timeout=60) is False:
            raise Exception("Server failed to start")
        print(f"[{get_current_timestamp()}] [{tag}] Server started")
        time.sleep(1)

        # Start the router
        router = subprocess.Popen(
            router_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=get_env(),
        )

        time.sleep(3)
        print(f"[{get_current_timestamp()}] [{tag}] Router started")
        time.sleep(3)

        # Starter the client
        client = subprocess.Popen(
            client_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        monitor_client_thread = threading.Thread(
            target=monitor_client_process,
            args=(client.stdout,),
        )
        monitor_client_thread.start()

        print(f"[{get_current_timestamp()}] [{tag}] Client started")
        client_returned = None
        while True:
            client_returned = client.poll()
            if client_returned is not None:
                break
            elif server.poll() is not None and router.poll() is not None:
                raise Exception("Server or router terminated unexpectedly")
            else:
                time.sleep(1)
    except Exception as e:
        print(f"[{get_current_timestamp()}] [{tag}] Failed: {e}")
        status = False
    except KeyboardInterrupt:
        print(f"[{get_current_timestamp()}] [{tag}] Aborted")
        status = False
        keyboard_interrupted = True
    else:
        if client_returned != 0:
            print(
                f"[{get_current_timestamp()}] [{tag}] Client exited with non-zero code"
            )
            status = False
        else:
            print(f"[{get_current_timestamp()}] [{tag}] Passed")
            status = True
    finally:
        if server is not None:
            server.terminate()
        if router is not None:
            router.terminate()
        if client is not None:
            client.terminate()
        print(f"[{get_current_timestamp()}] [{tag}] Finished")
        if keyboard_interrupted:
            raise KeyboardInterrupt()
        return status


def get_current_time():
    return datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d-%H%M")


def get_current_timestamp():
    return datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")


def remove(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
