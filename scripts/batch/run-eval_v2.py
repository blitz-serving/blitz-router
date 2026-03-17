# search lower bound and upper bound
import time
from librunner import (
    run_with_retry,
    set_args,
    get_current_time,
    get_current_timestamp,
    remove,
    set_hyperparams,
    get_abspath,
)
from environs import Env
from plot import plot_main
from email_alert import send_emails
import os
import yaml


failed_tests = []
passed_tests = []

email_list = [
    "grizzy@sjtu.edu.cn",
    "healthcliff-ding@sjtu.edu.cn",
    "green_egg@sjtu.edu.cn",
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def epilogue():
    if len(failed_tests) > 0:
        failed_msg = "Failed tests:\n{}\n".format("\n".join(failed_tests))
        passed_msg = "Passed tests:\n{}\n".format("\n".join(passed_tests))
        msg = failed_msg + passed_msg
    else:
        msg = "All tests passed\n"
    print(msg)
    env = Env()
    env.read_env()
    smtp_host = env("SMTP_HOST", None)
    account = env("SMTP_ACCOUNT", None)
    password = env("SMTP_PASSWORD", None)
    if password and account and smtp_host:
        send_emails(account, password, smtp_host, msg, email_list)


def setup_init_config(init_states, machines, replicas):
    init_config = {
        "init_states": init_states,
        "replicas": replicas,
        "machines": machines,
    }
    return init_config


def eval_all():
    task_start_time = get_current_time()

    config = load_config("scripts/batch/config_v2.yaml")

    cuda_devices = config["cuda_devices"]
    set_args("cuda_devices", ",".join([str(i) for i in cuda_devices]))
    dataset_home = config["dataset_home"]
    default_settings = config["default_settings"]

    inter_node_server = config["inter_node_server"]["enabled"]
    if inter_node_server:
        # TODO: update inter node config
        inter_node_config = config["inter_node_server"]["config"]
        inter_node_size = len(inter_node_config)
        # init_states = ["Inactive"] * 16
        # init_states[0] = "Prefill"
        # init_states[1] = "Decode"
        # init_config = setup_init_config(cuda_devices, init_states, [0] * 16, 16)
        set_args("inter_node_server", True)
        set_args("inter_node_config", inter_node_config)
        set_args("max_prefill_num", 10)
        set_args("max_decode_num", 10)
        set_args("min_prefill_num", 1)
        set_args("min_prefill_num", 1)
    else:
        set_args("inter_node_server", False)
        set_args("max_prefill_num", default_settings["max_prefill_num"])
        set_args("max_decode_num", default_settings["max_decode_num"])
        set_args("min_prefill_num", default_settings["min_prefill_num"])
        set_args("min_decode_num", default_settings["min_decode_num"])

    for key, value in default_settings.items():
        if (
            "scale_down_threshold_millis" in key
            or key == "prefill_lower_bound"
            or key == "prefill_upper_bound"
        ):
            continue
        set_args(key, value)

    model_configs = config["model_configs"]
    feature_configs = config["feature_configs"]

    prefill_bound_num = len(default_settings["prefill_lower_bound"])
    for i in range(prefill_bound_num):
        p_lower_bound = default_settings["prefill_lower_bound"][i]
        p_upper_bound = default_settings["prefill_upper_bound"][i]
        set_args("prefill_lower_bound", p_lower_bound)
        set_args("prefill_upper_bound", p_upper_bound)
        set_args("decode_upper_bound", default_settings["decode_upper_bound"])
        set_args("decode_lower_bound", default_settings["decode_lower_bound"])
        memory_available = "Full"
        for model_name, model_config in model_configs.items():
            tp_size = model_config["tensor_parallel_size"]
            set_args("tensor_parallel_size", tp_size)
            if inter_node_server:
                device_on_each_server = len(cuda_devices)
                cuda_devices = [
                    i for i in range(int(inter_node_size * device_on_each_server))
                ]
            # init router_config and server_config
            assert len(cuda_devices) % tp_size == 0
            router_init_stat = model_config["init_status"]
            router_len = len(router_init_stat)
            for _ in range(router_len, int(len(cuda_devices) / tp_size)):
                router_init_stat.append("Inactive")

            init_prefill_num = default_settings["init_prefill_num"]
            init_decode_num = default_settings["init_decode_num"]
            num_gpus_per_node = default_settings["num_gpus_per_node"]
            for i in range(init_prefill_num - 1):
                try:
                    idx = router_init_stat.index("Inactive")
                    router_init_stat[idx] = "Prefill"
                except ValueError:
                    break

            for i in range(init_decode_num - 1):
                try:
                    idx = router_init_stat.index("Inactive")
                    router_init_stat[idx] = "Decode"
                except ValueError:
                    break

            router_init_config = setup_init_config(
                init_states=router_init_stat,
                replicas=[
                    [i + j for j in range(tp_size)]
                    for i in range(0, len(cuda_devices), tp_size)
                ],
                machines=[
                    cuda_devices[i] // num_gpus_per_node
                    for i in range(0, len(cuda_devices), tp_size)
                ],
            )

            server_init_stat = [ele for ele in router_init_stat for _ in range(tp_size)]
            server_init_config = setup_init_config(
                init_states=server_init_stat,
                replicas=[[i] for i in range(len(cuda_devices))],
                machines=[x // num_gpus_per_node for x in cuda_devices],
            )

            set_args("router_config", router_init_config)
            set_args("server_config", server_init_config)

            set_args("model_name", model_name)
            set_hyperparams(model_name, memory_available)

            if tp_size > 1:
                set_args("ibv_rate", 100)
            else:
                set_args("ibv_rate", 0)

            for dataset_name in model_config["traces"][memory_available]:
                dataset_path = os.path.join(dataset_home, f"{dataset_name}.csv")
                set_args("dataset_path", dataset_path)

                for feature_tag, feature in feature_configs.items():
                    set_args("router_features", feature)
                    if "motiv" in feature_tag:
                        set_args(
                            "mock_load_millis",
                            config["extra_settings"]["mock_load_millis"],
                        )
                    elif "sllm" in feature_tag:
                        for scale_down_threshold in default_settings[
                            "baseline_scale_down_threshold_millis"
                        ]:
                            set_args(
                                "scale_down_threshold_millis", scale_down_threshold
                            )
                            tag = f"{feature_tag}-{dataset_name}-{model_name}"
                            relative_log_home = f"./log_home"
                            absolute_log_home = get_abspath(relative_log_home)
                            if inter_node_server:
                                node_level = "inter-node"
                            else:
                                node_level = "intra-node"
                            relative_workdir = f"{relative_log_home}/{task_start_time}-{node_level}-eval-{memory_available}/{model_name}/{dataset_name}/{feature_tag}/pup_{p_upper_bound}_plow_{p_lower_bound}/scale_down_{scale_down_threshold}"
                            absolute_workdir = f"{absolute_log_home}/{task_start_time}-{node_level}-eval-{memory_available}/{model_name}/{dataset_name}/{feature_tag}/pup_{p_upper_bound}_plow_{p_lower_bound}/scale_down_{scale_down_threshold}"
                            set_args("workdir", absolute_workdir)

                            print(
                                f"[{get_current_timestamp()}] [{tag}] Output to {relative_workdir}",
                                flush=True,
                            )
                            test_wrapper(tag, absolute_workdir, relative_workdir)
                            time.sleep(10)
                    elif "blitz" in feature_tag:
                        for scale_down_threshold in default_settings[
                            "blitz_scale_down_threshold_millis"
                        ]:
                            set_args(
                                "scale_down_threshold_millis", scale_down_threshold
                            )
                            tag = f"{feature_tag}-{dataset_name}-{model_name}"
                            relative_log_home = f"./log_home"
                            absolute_log_home = get_abspath(relative_log_home)
                            if inter_node_server:
                                node_level = "inter-node"
                            else:
                                node_level = "intra-node"
                            relative_workdir = f"{relative_log_home}/{task_start_time}-{node_level}-eval-{memory_available}/{model_name}/{dataset_name}/{feature_tag}/pup_{p_upper_bound}_plow_{p_lower_bound}/scale_down_{scale_down_threshold}"
                            absolute_workdir = f"{absolute_log_home}/{task_start_time}-{node_level}-eval-{memory_available}/{model_name}/{dataset_name}/{feature_tag}/pup_{p_upper_bound}_plow_{p_lower_bound}/scale_down_{scale_down_threshold}"
                            set_args("workdir", absolute_workdir)

                            print(
                                f"[{get_current_timestamp()}] [{tag}] Output to {relative_workdir}",
                                flush=True,
                            )
                            test_wrapper(tag, absolute_workdir, relative_workdir)
                            # time.sleep(10)
                    else:
                        assert(0)
    epilogue()


def test_wrapper(tag, absolute_workdir, relative_workdir):
    remove(absolute_workdir)
    os.makedirs(absolute_workdir, exist_ok=True)
    result = run_with_retry(tag)
    if result is False:
        failed_tests.append(relative_workdir)
        print(f"[{get_current_timestamp()}] [{tag}] Failed", flush=True)
    else:
        passed_tests.append(relative_workdir)
        with open(f"{absolute_workdir}/client_prologue.txt", "r") as f:
            prologue = f.read()
        with open(f"{absolute_workdir}/router.log", "a") as f:
            f.write(prologue)
        plot_main(
            f"{absolute_workdir}/router.log",
            f"{absolute_workdir}/client.jsonl",
            f"{absolute_workdir}/fig.pdf",
        )
    print(f"================================================", flush=True)


if __name__ == "__main__":
    env = Env()
    env.read_env()
    smtp_host = env("SMTP_HOST", None)
    account = env("SMTP_ACCOUNT", None)
    print(f"SMTP_HOST={smtp_host}")
    print(f"SMTP_ACCOUNT={account}")
    eval_all()
