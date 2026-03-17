import argparse
import subprocess
import tomli as tomllib
import time
import os
import signal
import re
from typing import Any
import copy  # 用于深拷贝
import json
import sys
from functools import reduce

# buffer background processes for cleanup on exit
background_procs = []

# runtime registry
rt_registry = {}


def register(name):
    def decorator(func):
        rt_registry[name] = func
        return func

    return decorator


def param_expand(s: str, params: dict) -> str:
    lo = 0
    hi = 0
    n = len(s)
    s1 = ""
    while hi < n:
        # pre: lo |-> Some(c) or $
        # invariant: lo \le hi
        # invariant: lo |-> $ => lo == hi
        if s[hi] == "$" and s[hi + 1] == "{":
            # flush last str segment
            s1 += s[lo:hi]
            # expand current param
            lo = hi
            # pre: lo |-> $
            hi += 2
            while s[hi] != "}":
                hi += 1
            var_name = s[lo + 2 : hi]
            # replace, if not expanded, keep original
            s1 += params.get(var_name, f"${{{var_name}}}")
            lo = hi + 1
            hi = lo
        else:
            hi += 1
    if s1 == "":
        return s
    else:
        # pre: hi == n
        s1 += s[lo:hi]
        print(f'"{s}" |-> {s1}')
        return s1


def resolve_inheritance(app_name: str, app_config: dict, seen: tuple = None) -> dict:
    """
    递归解析继承链，返回合并后的 app 配置。
    使用 tuple 做 seen 防止循环继承。
    """
    if seen is None:
        seen = ()
    if app_name in seen:
        raise ValueError(
            f"Circular inheritance detected: {' -> '.join(seen + (app_name,))}"
        )

    app_def = app_config.get(app_name)
    if app_def is None:
        raise ValueError(f"App '{app_name}' not found in config.")

    # 创建深拷贝，避免污染原始定义
    result = copy.deepcopy(app_def)

    # 处理继承
    if "inherit" in app_def:
        parent_name = app_def["inherit"]
        parent = resolve_inheritance(parent_name, app_config, seen + (app_name,))
        # 先合并父级
        for k, v in parent.items():
            if k not in result:
                result[k] = copy.deepcopy(v)
            elif k == "config" or k == "macro_rules":
                # 合并字典：子级优先
                merged = copy.deepcopy(v)
                merged.update(result[k])
                result[k] = merged
            elif k == "extra_args":
                # 列表：父级在前，子级追加（或覆盖？这里我们让子级完全定义）
                # 通常子级自己定义 extra_args 就覆盖
                if app_def.get(k) is None:
                    result[k] = copy.deepcopy(v)
            # 其他字段如 executable，子级优先，无需合并
        # 删除 inherit 字段
        result.pop("inherit", None)

    return result


def set_global_variables(variables: dict, global_kv: dict):
    for key, value in variables.items():
        if isinstance(value, str):
            variables[key] = param_expand(value, global_kv)


def load_toml_config(path: str, global_kv: dict = {}):
    with open(path, "rb") as f:
        config = tomllib.load(f)

    # 提取变量
    variables = config.get("variables", {})

    set_global_variables(variables, global_kv)
    # 先变量替换，再处理继承（变量替换要在继承前？否，要在继承后每层都做？）
    # 我们选择：先做变量替换整个 config，再处理 inherit
    config = resolve_variables(config, variables)

    # 处理所有 app 的 inherit 继承关系
    resolved_app_config = {}
    app_config_raw = config.get("app", {})

    for app_name, app_def in app_config_raw.items():
        resolved_app_config[app_name] = resolve_inheritance(app_name, app_config_raw)

    config["app"] = resolved_app_config
    return config


def resolve_variables(obj, variables: dict) -> Any:
    """递归替换字符串中的 ${key} 为变量值"""
    if isinstance(obj, str):
        return param_expand(obj, variables)
    elif isinstance(obj, list):
        return [resolve_variables(item, variables) for item in obj]
    elif isinstance(obj, dict):
        return {k: resolve_variables(v, variables) for k, v in obj.items()}
    else:
        return obj


def insert_envs(old_cmd: str, app_running_config: dict) -> str:
    envs = app_running_config.get("envs", {})
    env_str = ""
    for key, value in envs.items():
        env_str += f"{key}={value} "
    return env_str + old_cmd


def process_macro(
    app_name: str, app_general: dict, app_self_cfg: dict, add_executable: bool = True
) -> str:
    macro_rules = app_general.get("macro_rules", {})
    args = []

    for cli_key, config_key in macro_rules.items():
        value = app_self_cfg.get(config_key, "")
        if value is None or value == "":
            value = "''"
        args.append(f"{cli_key} {value}")

    executable = app_general["executable"]
    # print(f"exe {executable}")

    if isinstance(executable, str):
        cmd_parts = [executable]
    elif isinstance(executable, list):
        cmd_parts = [" ".join([str(x) for x in executable])]
    else:
        cmd_parts = [str(executable)]

    extra_args = app_general.get("extra_args", [])
    if isinstance(extra_args, str):
        extra_args = [extra_args]

    if add_executable:
        cmd_parts = cmd_parts + extra_args + args
    else:
        cmd_parts = extra_args + args

    # print(f"cmd: {" ".join(cmd_parts)}")

    return " ".join(cmd_parts)


@register("raw")
def gen_raw_cmd(app_cmd: str, rt_config: dict, app_running_config: dict) -> str:
    app = app_running_config["app"]
    stubs = []
    stub_path = "/nvme/zkx/blitz-infer-pack/exps/blitz-run/configs/config-stubs.json"
    if app == "vllm_template":
        base_port = app_running_config.get("base_port", 22222)
        offset = app_running_config.get("port_offset", 0)
        if offset > 0:
            with open(stub_path, "r") as f:
                data = json.load(f)
                stubs = data
        # print(f"{base_port}, {offset=}")
        # print(f"{rt_config=} {app_running_config=}")
        # print(f"before replace {app_cmd=}")
        port = base_port + offset
        stubs.append(f"http://localhost:{port}")
        # TODO, some hard-code path here..
        log_file = f"/nvme/lmetric/logs/vllm{offset+1}.log"
        # 替换模板
        app_cmd = app_cmd.replace("${port}", str(port))
        app_cmd = app_cmd.replace("${log_file}", log_file)

        # TODO, some hard-code path here..
        with open(stub_path, "w") as f:
            json.dump(stubs, f, indent=2)

    # 插入环境变量
    full_cmd = insert_envs(app_cmd, app_running_config)

    # print(f"full cmd {full_cmd=} \n")
    # 后台运行
    if app_running_config.get("background", False):
        return f"{full_cmd} &"
    return full_cmd


@register("ssh")
def gen_ssh_cmd(app_cmd: str, ssh_global_config: dict, app_running_config: dict) -> str:
    if (
        "remote_user" not in app_running_config
        or "remote_host" not in app_running_config
    ):
        raise ValueError("Missing 'remote_user' or 'remote_host'.")

    user, host = app_running_config["remote_user"], app_running_config["remote_host"]
    run_in_bg = app_running_config.get("background", False)
    if run_in_bg:
        app_cmd = f"nohup {app_cmd} > /dev/null 2>&1 &"

    full_cmd = f'ssh "{user}@{host}" {app_cmd} </dev/null'
    if run_in_bg:
        full_cmd += " &"
    return insert_envs(full_cmd, app_running_config)


@register("mpi")
def gen_mpi_cmd(app_cmd: str, mpi_global_config: dict, app_running_config: dict) -> str:
    hosts = app_running_config.get("hosts", {})
    host_list = [f"{k}:{v}" for k, v in hosts.items()]
    host_cmd = f"--host {','.join(host_list)}" if host_list else ""

    app_cmd = insert_envs(app_cmd, app_running_config)
    mpirun_cmd = f"mpirun {host_cmd}".strip() if host_cmd else "mpirun"
    full_cmd = f"{mpirun_cmd} {app_cmd}"
    return full_cmd


def block_until_keyword(proc: subprocess.Popen, keyword: str):
    """等待输出中出现关键字"""
    try:
        while True:
            line = proc.stdout.readline().decode("utf-8")
            if not line:
                break
            if keyword in line:
                print(f"Found keyword: '{keyword}'")
                break
    except:
        pass


def run_apps(rt: str, rt_config: dict, app_config: dict):
    all_apps = rt_config.get("config", [])
    for init_config in all_apps:
        app_name = init_config["app"]
        if app_name not in app_config:
            raise ValueError(f"App '{app_name}' not found.")

        app_general = app_config[app_name]
        print(f"{app_general=}\n")
        app_self_cfg = app_general.get("config", {})
        print(f"{app_self_cfg=}\n")
        output_path = app_self_cfg.get("log_path", None)
        if output_path:
            print(f"make dirs {output_path=}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        app_cmd = process_macro(app_name, app_general, app_self_cfg)
        print(f"{app_cmd=}\n")

        rt_func = rt_registry.get(rt)
        if not rt_func:
            raise ValueError(f"Unknown runtime: {rt}")

        cmd = rt_func(app_cmd, rt_config, init_config)

        print(f"RUN: {cmd}")

        run_in_bg = init_config.get("background", False)
        keyword = init_config.get("block_keyword")

        if run_in_bg and keyword:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            block_until_keyword(proc, keyword)
            background_procs.append(proc)
        elif run_in_bg:
            proc = subprocess.Popen(
                cmd, shell=True, stderr=subprocess.DEVNULL, preexec_fn=os.setsid
            )
            background_procs.append(proc)
        else:
            subprocess.run(cmd, shell=True, check=True)

        time.sleep(2)


def check_and_get_runtime(config: dict):
    for rt, cfg in config.get("runtime", {}).items():
        yield rt, cfg


def main():
    parser = argparse.ArgumentParser(description="Run apps from TOML config")
    parser.add_argument("--toml", required=True, help="Path to TOML file")
    # 允许用户传入任意名称的全局变量， 比如 --output_dir /nvme/lmetric/logs/dp8_2code_2conv/lwl,将所有这样的全局变量存入global_kv
    # global kv 能够替换全局变量中的place holder
    # 用 parse_known_args 捕获未知参数
    args, unknown = parser.parse_known_args()

    # 解析类似 --key=value 的参数
    # this kind of args are not splitted by space
    global_kv = {}
    for arg in unknown:
        if arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            global_kv[key] = value

    print(global_kv)

    global background_procs
    background_procs = []

    try:
        config = load_toml_config(args.toml, global_kv)
        for rt_name, rt_config in check_and_get_runtime(config):
            # smart runner need one foreground app as blocker  
            def fold_is_bg(acc, cfg) -> bool:
                return cfg.get("background", False) and acc
            if reduce(fold_is_bg, rt_config.get("config", []), True):
                raise ValueError("No foreground app found in runtime config!")
            # postcondition: there must be one Forground app each run
            run_apps(rt=rt_name, rt_config=rt_config, app_config=config["app"])
    except ValueError as e:
        print(f"Fail to run app: {e}")
    except KeyboardInterrupt as e:
        # NOTE: you must use SIGINT rather than SIGTERM!
        # to bypass kill-session, emulate a user via C-c
        print(f"Keyboard interupt: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 清理后台进程
        for proc in background_procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=3)
                print(f"Stopped PID={proc.pid}")
            except:
                pass


if __name__ == "__main__":
    main()
