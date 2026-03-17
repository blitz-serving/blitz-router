#!/bin/bash
cargo build --release --package router_v2 --no-default-features --features ngrok,impl_sllm,cache_replace,mutate
