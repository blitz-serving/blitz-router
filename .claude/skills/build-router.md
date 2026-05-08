---
name: build-router
description: Build blitz-router with correct features and Python environment for PyO3
---

# Building blitz-router

## Two Build Modes

### Inner-cluster mode (no Python dependency)

```bash
cargo build --release -p router --features lmetric-q
```

- Default `--chat-template-mode none` — prompts arrive pre-rendered
- No Python headers needed at build time, no Python needed at runtime
- Use this when blitz-router sits behind a frontend that applies chat templates

### Gateway mode (PyO3 embedded Python)

```bash
cargo build --release -p router --features python-chat-template,lmetric-q
```

- Enables `--chat-template-mode python` — router renders chat templates via embedded Python jinja2
- **Requires at build time**: Python dev headers (`Python.h`) + `libpython3.x.so`
- **Requires at runtime**: `libpython3.x.so` + `jinja2` pip package

## PyO3 Python Discovery

PyO3 finds Python in this priority order:

1. **`PYO3_PYTHON` env var** — explicit path, e.g., `PYO3_PYTHON=/usr/bin/python3.10`
2. **`pkg-config`** — searches for `python-3.x.pc`
3. **PATH search** — tries `python3`, `python3.x` in PATH

### Verifying Python is ready for PyO3

```bash
# Check Python version (need 3.8+)
python3 --version

# Check dev headers exist
ls /usr/include/python3.*/Python.h

# Check shared library exists
ls /usr/lib/*/libpython3.*.so*

# Check jinja2 is installed (runtime requirement)
python3 -c "import jinja2; print(jinja2.__version__)"
```

### Installing Python dev headers

```bash
# Ubuntu/Debian
sudo apt-get install -y python3-dev
pip3 install jinja2

# If specific version needed
sudo apt-get install -y python3.10-dev
```

## Feature Flags Reference

`vllm-backend` is implied by `default` and by `colocation`/`zmq-backend` — you do not normally need to pass it explicitly.

| Feature | Purpose | Requires |
|---------|---------|----------|
| `vllm-backend` | HTTP+SSE backend (yaullm) | - |
| `zmq-backend` | ZMQ backend (headless yaullm) | zeromq, rmp-serde |
| `python-chat-template` | PyO3 chat template rendering | Python dev headers |
| `simulator` | Latency-prediction subsystem (orthogonal to policies) | - |

## Scheduling Policy (pick exactly one)

Policies are organized under `router/src/policies/` by upstream baseline system. Pick exactly one of the 18 features below; if none is given, the build defaults to `join-shortest-weight-q`.

| Module | Features |
|---|---|
| `simple.rs` | `random-q`, `round-robin-q`, `least-wait-token-q`, `bounded-most-hit-q` |
| `vllm.rs` | `join-shortest-weight-q` |
| `bailian.rs` | `bailian-impl-q` |
| `aibrix.rs` | `aibrix-q` |
| `dynamo.rs` | `dynamo-q`, `dynamo-po-q` |
| `lmetric.rs` | `lmetric-q` |
| `preble/` | `preble-q` |
| `llm_d/` | `most-hit-q`, `least-waiting-q`, `least-bs-q`, `least-active-q`, `least-token-load-q`, `most-hit-load-q`, `most-hit-load-active-q` |
