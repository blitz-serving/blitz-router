# radixtree

Patricia-trie prefix-matching crate consumed by `blitz-router`.

## Public surface

- `BlockHash` — interface trait the router's KV-cache prefix matcher
  consumes. Two implementations live behind it: `RadixTreeBlockHash`
  (here) and `HashTableBlockHash` (in `router/src/scheduler/kvcache.rs`).
- `RadixTreeBlockHash` — production L3-lowered Patricia trie with
  `V = Bids` (multi-bid per position via `SmallVec<[u64;1]>`),
  `SpinLock`-guarded mutation, `AtomicUsize` node count, monotonic
  `epoch`. Concurrent-safe; survives chunked-prefill duplicate-prefix
  workloads (the dynamo-q under-prediction bug fix).
- `RadixTreeReqIdHash` — simpler Box-based trie with `V = ReqId` for
  the simulator's L1 incremental mirror. Different requirements
  (eviction arbitration via in-flight set, drift tolerance) → kept as
  a separate impl rather than unified.
- `Bids` — public type alias `SmallVec<[u64;1]>`.
- `verified` (gated by `verify` feature) — Verus-verified L0 spec.

## Lowering history

The user originally migrated the router's KV-cache index from a flat
`HashMap` to a hand-rolled Patricia trie. The first hand-rolled
version (raw pointers, no concurrency) **segfaulted** in production
under multi-tenant chunked-prefill workloads. To recover, the team
wrote a Verus-verified L0 spec (see [`src/verified.rs`](src/verified.rs))
and progressively lowered it through bijective translations:

| Level | Form | What changed |
|-------|------|--------------|
| L0 | `Vec<Box<Node>>`, linear child scan | Verus-proved baseline |
| L1 | + sorted children, binary search | O(log n) child lookup |
| L2 | `Vec<*mut Node>` raw pointers | drop `Box` overhead |
| L3 | `Children::{Small, Large}` split | HashMap upgrade past `SMALL_MAX` |

All four levels live behind a `RadixTree` trait in
[`benches/lowering_levels.rs`](benches/lowering_levels.rs); the
Criterion harness in [`benches/workloads.rs`](benches/workloads.rs)
exercises each on KV-cache-shaped workloads so regressions at any
level are caught.

The L3 production form lives in
[`src/block_hash.rs`](src/block_hash.rs) (with the concurrency +
multi-bid additions on top of L3). [`src/req_id_hash.rs`](src/req_id_hash.rs)
intentionally stops at "simple Rust trie" — the simulator mirror's
working set is too small (~6.4k nodes) to warrant L3-style raw
pointers.

## Build & test

```bash
cargo check -p radixtree
cargo test  -p radixtree
cargo bench -p radixtree --bench workloads
```

To re-verify the L0 spec against Verus (requires a `verus` toolchain):

```bash
verus radixtree/src/verified.rs --crate-type=lib
```
