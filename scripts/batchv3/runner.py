import argparse
import subprocess
import tomli as tomllib
import time
import os
import signal
from utils.blocker import block_until_keyword


background_procs = []

rt_registry = {}
def register(name):
    def decorator(func):
        rt_registry[name] = func
        return func

    return decorator


def load_toml_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)
    
def insert_envs(old_cmd, app_running_config):
    envs = app_running_config.get("envs", {})
    for key, value in iter(envs.items()):
        old_cmd = f"{key}={value} {old_cmd}"
    return old_cmd


def process_macro(app_name, app_general, app_self_cfg, add_executable=True):
    macro_rules = app_general.get("macro_rules", {})

    # generate args from rules
    args = []
    for cli_key, config_key in macro_rules.items():
        parts = config_key.split(".")
        value = app_self_cfg
        for part in parts:
            value = value.get(part, "")
            if value is None:
                raise ValueError(
                    f"Cannot find value for key: {app_name}.config.{config_key}"
                )
            if value == "":
                value = "\'\'"
        args.append(f"{cli_key} {value}")

    # start building command
    executable = app_general["executable"]
    extra_args = app_general.get("extra_args", [])

    if add_executable:
        run_cmd = " ".join(executable + extra_args + args)
    else:
        run_cmd = " ".join(extra_args + args)

    return run_cmd

@register("ssh")
def gen_ssh_cmd(app_cmd, ssh_global_config, app_running_config):
    if (
        "remote_user" not in app_running_config.keys()
        or "remote_host" not in app_running_config.keys()
    ):
        raise (f"Failed to get remote user or remote host")

    ssh_init_cmd = process_macro(
        "ssh",
        app_general=ssh_global_config,
        app_self_cfg=app_running_config,
        add_executable=False,
    )
    user, host = app_running_config["remote_user"], app_running_config["remote_host"]
    run_in_bg = app_running_config.get("background", False)
    if run_in_bg:
        app_cmd = f"nohup {app_cmd} > /dev/null 2>&1 &"
    full_cmd = " ".join(
        ["ssh", f'"{user}@{host}"'] + [ssh_init_cmd] + [f'"{app_cmd}"', "</dev/null"]
    )
    if run_in_bg:
        full_cmd = f"{full_cmd} &"
        
    full_cmd = insert_envs(full_cmd, app_running_config)

    return full_cmd

@register("mpi")
def gen_mpi_cmd(app_cmd, mpi_global_config, app_running_config):

    mpi_init_cmd = process_macro("mpi", app_general=mpi_global_config, app_self_cfg=app_running_config)
    if app_running_config.get("inter_node", False) == True:
        # add host config
        host_cfgs = []
        hosts = app_running_config.get("hosts", {})
        for machine, slot in iter(hosts.items()):
            host_cfgs.append(f"{machine}:{slot}")
        host_cmd = ','.join(host_cfgs)
        host_cmd = f"--host {host_cmd}"
        # add app_cmd
        app_cmd = insert_envs(app_cmd, app_running_config)
        # replace mpirun with mpirun+host_cfg
        mpi_init_cmd = mpi_init_cmd.replace("mpirun", f"mpirun {host_cmd}", 1)
        start_num = app_running_config.get("start_num", 1)
        full_cmd = " ".join([mpi_init_cmd, str(start_num), "bach -c \'", app_cmd, "\'"])
    else:
        full_cmd = " ".join([mpi_init_cmd, app_cmd])
    full_cmd = insert_envs(full_cmd, app_running_config)
    return full_cmd

@register("raw")
def gen_raw_cmd(app_cmd, raw_global_config, app_running_config):
    run_in_bf = app_running_config.get("background", False)
    full_cmd = insert_envs(app_cmd, app_running_config)
    if run_in_bf:
        return f"{full_cmd} &"
    return full_cmd

def run_apps(rt, rt_config, app_config):
    all_apps = rt_config["config"]
    for init_config in all_apps:
        app_name = init_config["app"]
        if app_name not in app_config.keys():
            raise (f"Failed to find app {app_name}")
        app_cmd = process_macro(
            app_name=app_name,
            app_general=app_config[app_name],
            app_self_cfg=app_config[app_name].get("config", {}),
        )
        rt_func = rt_registry.get(rt)
        cmd = rt_func(app_cmd, rt_config, init_config)
        print(cmd)
        # continue
        run_in_bg = init_config.get("background", False)
        keyword = init_config.get("block_keyword", None)
        if run_in_bg and keyword:
            print(f"Wait until {keyword}")
            proc = subprocess.Popen(cmd,
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    preexec_fn=os.setsid
                                    )
            block_until_keyword(proc = proc, keyword=keyword)
            background_procs.append(proc)
        elif run_in_bg:
            proc = subprocess.Popen(cmd, shell= True, stderr=subprocess.DEVNULL,
                                    preexec_fn=os.setsid)
            background_procs.append(proc)
        else:
            result = subprocess.run(cmd, shell=True, check=True)
            if result.stdout is not None:
                print("stdout:", result.stdout)
            if result.stderr is not None:
                print("stderr:", result.stderr)
        time.sleep(2)


def check_and_get_runtime(config):
    rt_iter = config.get("runtime", {})
    for rt, rt_config in iter(rt_iter.items()):
        yield rt, rt_config


def main():
    # arg parse
    parser = argparse.ArgumentParser(
        description="Generate launcher script from TOML config."
    )
    parser.add_argument("--toml", required=True, help="Path to the TOML config file")
    args = parser.parse_args()

    # run all apps
    config = load_toml_config(args.toml)
    for rt_name, rt_config in check_and_get_runtime(config):
        all_apps = config.get("app", {})
        run_apps(rt=rt_name, rt_config=rt_config, app_config=all_apps)
        
    for proc in background_procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # 或 SIGKILL
            print(f"Killed PID={proc.pid}")
        except ProcessLookupError:
            print(f"PID={proc.pid} already exited")


if __name__ == "__main__":
    main()
