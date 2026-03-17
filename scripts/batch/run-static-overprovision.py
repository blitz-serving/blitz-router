import time
from librunner import (
    get_current_time,
    set_hyperparams,
    remove,
    run_with_retry,
    set_args,
)
from environs import Env
from plot import plot_main
from email_alert import send_emails
import os


def run_motivation_test():
    set_args("cuda_devices", "0,1,2,3,4,5,6,7")
    set_args("router_features", "ngrok,sched_naive,manually_scale")
    current_time = get_current_time()

    dn = "AzureCode2023-5min"
    router_config = {
        "init_states": [
            "Prefill",
            "Prefill",
            "Prefill",
            "Decode",
            "Prefill",
            "Decode",
            "Prefill",
            "Decode",
        ],
        "replicas": [[0], [1], [2], [3], [4], [5], [6], [7]],
    }
    server_config = {
        "init_states": [
            "Prefill",
            "Prefill",
            "Prefill",
            "Decode",
            "Prefill",
            "Decode",
            "Prefill",
            "Decode",
        ],
        "replicas": [[0], [1], [2], [3], [4], [5], [6], [7]],
    }
    set_args("router_config", router_config)
    set_args("server_config", server_config)
    for model_name in ["llama2_7b", "llama3_8b", "llama2_13b", "mistral_13b"]:
        set_args("model_name", model_name)
        set_hyperparams(model_name, "Full")
        for sf in [1.2, 1.4, 1.6]:
            set_args("scale_factor", sf)
            tag = f"{dn}-{model_name}-{sf}x-scale"
            relative_workdir = f"./log/{current_time}/{model_name}/code-5p3d/{sf}x"
            workdir = os.path.abspath(relative_workdir)
            dataset_path = f"/nvme/workdir/wht/processed-dataset/{dn}.csv"
            set_args("dataset_path", dataset_path)
            set_args("client_request_time_in_secs", int(5 * 60 / sf))
            set_args("workdir", workdir)
            single_test(tag, workdir)
            time.sleep(5)
            print(f"================================================", flush=True)

    dn = "AzureConv2023-3x-scale-5min-02"
    router_config = {
        "init_states": [
            "Prefill",
            "Decode",
            "Prefill",
            "Decode",
            "Prefill",
            "Decode",
            "Decode",
            "Decode",
        ],
        "replicas": [[0], [1], [2], [3], [4], [5], [6], [7]],
    }
    server_config = {
        "init_states": [
            "Prefill",
            "Decode",
            "Prefill",
            "Decode",
            "Prefill",
            "Decode",
            "Decode",
            "Decode",
        ],
        "replicas": [[0], [1], [2], [3], [4], [5], [6], [7]],
    }
    set_args("router_config", router_config)
    set_args("server_config", server_config)
    for model_name in ["llama2_7b", "llama3_8b", "llama2_13b", "mistral_13b"]:
        set_args("model_name", model_name)
        set_hyperparams(model_name, "Full")
        for sf in [1.2, 1.4, 1.6]:
            set_args("scale_factor", sf)
            tag = f"{dn}-{model_name}-{sf}x-scale"
            relative_workdir = f"./log/{current_time}/{model_name}/conv-3p5d/{sf}x"
            workdir = os.path.abspath(relative_workdir)
            dataset_path = f"/nvme/workdir/wht/processed-dataset/{dn}.csv"
            set_args("dataset_path", dataset_path)
            set_args("client_request_time_in_secs", int(5 * 60 / sf))
            set_args("workdir", workdir)
            single_test(tag, workdir)
            time.sleep(5)
            print(f"================================================", flush=True)


def single_test(tag, workdir):
    remove(workdir)
    os.makedirs(workdir, exist_ok=True)
    result = run_with_retry(tag)
    if result is False:
        env = Env()
        env.read_env()
        smtp_host = env("SMTP_HOST", None)
        account = env("SMTP_ACCOUNT", None)
        password = env("SMTP_PASSWORD", None)
        if password and account and smtp_host:
            send_emails(
                account,
                password,
                smtp_host,
                f"Test {workdir} failed",
                [
                    "grizzy@sjtu.edu.cn",
                    "healthcliff-ding@sjtu.edu.cn",
                    "green_egg@sjtu.edu.cn",
                ],
            )
    else:
        with open(f"{workdir}/client_prologue.txt", "r") as f:
            prologue = f.read()
        with open(f"{workdir}/router.log", "a") as f:
            f.write(prologue)
        plot_main(
            f"{workdir}/router.log",
            f"{workdir}/client.jsonl",
            f"{workdir}/fig.pdf",
        )


if __name__ == "__main__":
    env = Env()
    env.read_env()
    smtp_host = env("SMTP_HOST", None)
    account = env("SMTP_ACCOUNT", None)
    print(f"SMTP_HOST={smtp_host}")
    print(f"SMTP_ACCOUNT={account}")
    run_motivation_test()
