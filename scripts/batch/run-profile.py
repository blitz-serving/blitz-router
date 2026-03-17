import time
from librunner import (
    remove,
    run_with_retry,
    set_args,
    set_hyperparams,
)
from environs import Env
from plot import plot_main
from email_alert import send_email
import os


def run_motivation_test():
    router_config = {
        "init_states": [
            "Prefill",
            "Decode",
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
            "Decode",
            "Prefill",
            "Decode",
            "Prefill",
            "Decode",
            "Prefill",
            "Decode",
        ],
        "replicas": [[0], [1], [2], [3], [4], [5], [6], [7]],
    }

    set_args("scale_factor", 1.0)
    set_args("cuda_devices", "0,1,2,3,4,5,6,7")
    set_args("max_prefill_num", 6)
    set_args("max_decode_num", 6)
    set_args("prefill_lower_bound", 0.1)
    set_args("prefill_upper_bound", 0.4)
    set_args("decode_lower_bound", 0.35)
    set_args("decode_upper_bound", 0.65)
    set_args("scale_down_threshold_millis", 2000)
    set_args("max_prefill_num", 6)
    set_args("max_decode_num", 6)
    set_args("migration_lower_bound", 0.0)
    set_args("migration_upper_bound", 1.0)
    set_args("router_config", router_config)
    set_args("server_config", server_config)

    dataset_name = "profile"
    set_args("client_request_time_in_secs", 110)
    set_args("dataset_path", "./temp/profile.csv")
    set_args("router_features", "ngrok,sched_naive,manually_scale")

    for model in ["llama2_7b", "llama3_8b", "llama2_13b", "mistral_13b"]:
        set_hyperparams(model, "Full")
        tag = model
        workdir = f"./log/{dataset_name}/{tag}"
        set_args("model_name", model)
        set_args("workdir", workdir)
        single_test(tag, workdir)
        time.sleep(5)


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
            send_email(
                account,
                password,
                smtp_host,
                f"Test {workdir} failed",
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
