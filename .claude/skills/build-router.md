---
name: build-router
description: Build blitz-router with correct features and Python environment for PyO3
---

# Building blitz-router

## Two Build Modes

### Inner-cluster mode (no Python dependency)

```bash
cargo build --release -p router --features vllm-backend,join-shortest-q
```

- Default `--chat-template-mode none` — prompts arrive pre-rendered
- No Python headers needed at build time, no Python needed at runtime
- Use this when blitz-router sits behind a frontend that applies chat templates

### Gateway mode (PyO3 embedded Python)

```bash
cargo build --release -p router --features vllm-backend,python-chat-template,join-shortest-q
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

| Feature | Purpose | Requires |
|---------|---------|----------|
| `vllm-backend` | HTTP+SSE backend (yaullm) | - |
| `zmq-backend` | ZMQ backend (headless yaullm) | zeromq, rmp-serde |
| `python-chat-template` | PyO3 chat template rendering | Python dev headers |
| `join-shortest-q` | Scheduling policy | - |
| `bounded-most-hit-q` | Cache-aware scheduling | - |

## Scheduling Policy (pick exactly one)

Must enable exactly one scheduling policy feature:
`random-q`, `round-robin-q`, `join-shortest-q`, `bounded-most-hit-q`, `least-wait-token-q`, `join-shortest-q-tuple`, `join-shortest-q-weight`, `bailian-impl-q`
