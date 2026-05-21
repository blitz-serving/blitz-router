# Preble — Abstract Design

This is the canonical mental model of what Preble computes. It is the spec
the Rust port (`router/src/scheduler/policies/preble/`) reproduces and the
spec a future refactor should target. It is not a description of the
current code's storage layout; see §"Notes on alphabet and topology" at
the end for the bridge to the implementation.

The original AIBrix Go reference is at
[`workspace/aibrix/pkg/plugins/gateway/algorithms/prefix_cache_preble.go`](../../aibrix/pkg/plugins/gateway/algorithms/prefix_cache_preble.go)
plus
[`pkg/utils/prefixcacheindexer/tree.go`](../../aibrix/pkg/utils/prefixcacheindexer/tree.go).
The two AIBrix-specific implementation defects we deliberately do NOT
reproduce are documented in agent memory; this doc is design-level only.

---

## Definitions

A **radix tree** T over an alphabet has nodes connected by edges. Each
edge carries a label that is a non-empty sequence of symbols from the
alphabet. With path compression, an edge can have a label of length
greater than 1; without compression every edge has label length exactly 1.

For a non-root node N:

- **prefix(N)** is the concatenation of edge labels on the path from root
  to N. It is a sequence of alphabet symbols.
- **ndepth(N)** is the number of edges on the path from root to N
  (equivalently, the number of nodes on that path excluding root).
- **tdepth(N)** is the length of prefix(N), equal to the sum of edge-label
  lengths from root to N.

These two depths are not equal in general. With path compression they
can differ; without compression they coincide.

In a tree, every non-root node has a unique path from root, so non-root
nodes are in bijection with stored prefixes. Operations are phrased on
prefixes; the node is just the storage location of the prefix's data.

**Per-prefix data Preble maintains** (call this **State A**):

- `owners(π) ⊆ Pods` — pods that have served some request through prefix π.
- `count(π) ∈ ℕ` — number of in-window requests whose full input was
  exactly π. Note "full input was exactly π", not "passed through π" —
  this is what the implementation enforces by only updating the count map
  for the leaf returned by the full-input INSERT call.
- `lastAccess(π)` — wall-clock time of the last traversal through π.
- Aggregate cost-model state at the leaf only: `hit_tokens(π)`,
  `prompt_tokens(π)`, `decoding_size(π)`, `total_decode_lengths(π)`.

**Window constants:** `W_count = 3 min` (sliding window for count decay),
`W_lru = 5 min` (LRU TTL for ownership decay).

---

## Operation 1 — INSERT(τ, P)

Inputs: full token sequence τ of a routed request, the chosen pod P.

1. Materialize τ in T. Ensure there is a node L with prefix(L) = τ;
   create intermediate nodes / split as needed. **L's tdepth equals
   |τ|** — the leaf is at the FULL input depth, not at the
   matched-prefix depth.

2. For every prefix σ that is a prefix of τ (including σ = τ, excluding
   σ = empty): `owners(σ) ← owners(σ) ∪ {P}`, `lastAccess(σ) ← now`.
   **Owners propagate to every ancestor**, not just the leaf.

3. Increment `count(τ)` by 1. Append `(τ, P, now)` to the sliding-window
   log. Update the leaf's aggregate cost-model fields with this
   request's tokens / hits / output length.

Two key consequences:

- `count` is incremented only at the prefix τ — never at any proper
  prefix of τ. So `count(σ) > 0` iff σ was the full input of at least
  one recent request.
- P is added to `owners(σ)` for every prefix σ of τ — every level of
  granularity. This is what makes Stage 1's ancestor walk meaningful.

---

## Operation 2 — EXPIRE

The abstract spec has two decay paths; the blitz-router implementation
collapses them by leveraging engine SSE feedback that the Go reference
did not have.

**Sliding window decay (W_count = 3 min).** For each `(τ, P, t)` record
in the window log with `now − t > W_count`: decrement `count(τ)` and
remove the record.

**LRU decay (W_lru = 5 min) — abstract spec only.** For any prefix σ
with `now − lastAccess(σ) > W_lru`: remove σ and every prefix that
has σ as a strict prefix from T entirely. **Not implemented in
blitz-router.** The Go reference uses LRU because Go has no
engine-side feedback — it must guess when cached prefixes go stale.
blitz-router's colocation event loop (`router/src/engine/colocation.rs`
lines 606, 654) consumes engine SSE updates (`evicted_block_ids`,
`cur_used_block_ids`) and calls `sctx.block_hash.{insert,remove}`
accordingly. The KV-cache tree state IS the ground truth of "what is
currently cached on this replica"; Preble does not maintain a
parallel stale-state guess.

The 1 Hz background eviction tick is also unnecessary in this
implementation. The sliding-window decay is **lazy** — `expire(now)`
on the per-replica `SlidingWindow` pops front-of-deque entries that
are older than `W_count` whenever an aggregate is read. Amortized
O(1) per push; no background thread.

---

## Operation 3 — QUERY(τ)

Inputs: full token sequence τ of a new request awaiting routing.

1. **Descend.** Find the deepest existing prefix M of τ in T. If T is
   empty along τ from depth 1, M = empty.

2. **Filter.** Let m = |M|. If `m / |τ| > 0.5`, go to Stage 1; else
   go to Stage 2.

3. **Stage 1.** Walk the chain `M, parent(M), ..., root`. Let M\* be
   the deepest prefix in this chain with `owners(M*) ≠ ∅`. Among the
   pods in `owners(M*)`, select the one minimizing `LOAD`. Ties broken
   arbitrarily.

4. **Stage 2.** Among all pods, select the one minimizing `COST`.

---

## LOAD(P) — Stage 1's tie-break metric

Sum, over all prefixes σ in T such that `count(σ) > 0` AND
`P ∈ owners(σ)`, of `count(σ)`.

In words: sum the leaf-counts of every recent-terminal prefix that P
co-owns.

**Two crucial properties:**

**L1.** The condition `count(σ) > 0` restricts to prefixes that were
the FULL input of at least one recent request. Internal prefixes —
those that are proper prefixes of recent inputs but never themselves
a request's full input — contribute zero. Their count is zero
regardless of how many pods own them.

**L2. LOAD over-counting (abstract spec only — eliminated in
blitz-router).** If two requests with the same full input τ_0 were
routed to different pods P and Q, the abstract spec puts both in
`owners(τ_0)` and gives `count(τ_0) ≥ 2`, so each pod's LOAD is
incremented by `count(τ_0)` even though each only served one
request. The blitz-router implementation eliminates this by
construction: each replica has its own engine-driven tree, and
LOAD(P) = `|W_P|` is the count of requests actually routed to P
(see §Implementation). The L2 over-count is a property of the
abstract spec's owner-set design, not a correctness requirement.

> **Concrete example (abstract spec).** Three requests in the window
> with identical input τ_0, routed to pods P1, P2, P3.
> `count(τ_0) = 3`, `owners(τ_0) = {P1, P2, P3}`.
> `LOAD_abstract(P1) = LOAD_abstract(P2) = LOAD_abstract(P3) = 3`.
> blitz-router gives `LOAD(P1) = LOAD(P2) = LOAD(P3) = 1` —
> per-replica counts attribute correctly.

---

## COST(P) — Stage 2's selection metric

For each prefix σ with `count(σ) > 0`:

- `prefill_cost(σ) ≈ miss_rate(σ) × count(σ) × prefill_time_polynomial(tdepth(σ))`
- `decode_cost(σ) = output_len(σ) × time_per_token`

where `prefill_time_polynomial` is a static cost model (Mistral-7B
coefficients on V100/A6000, hardcoded — not calibrated for the deployed
model/hardware), and `time_per_token` is the hardcoded constant
0.15 s. The `avgTimePerTokenPerPod` lookup in the AIBrix Go reference
is dead code in production (the map is only written by tests).

For a pod P:

`COST(P) = Σ over prefixes σ in T with count(σ) > 0 AND P ∈ owners(σ)`
of `(prefill_cost(σ) / |owners(σ)| + decode_cost(σ))`.

Same iteration structure as `LOAD` but with per-owner amortization on
the prefill term (divides by `|owners(σ)|`). Decode is not amortized —
the same value contributes to every owner.

---

## Implementation in blitz-router

State A and State B are split along the per-replica boundary that
already exists in `router/src/scheduler/state.rs:230`
(`ScheduleContext { lmetric, block_hash }`):

- **State A — per-prefix.** Lives in each replica's
  `RadixTreeBlockHash` (the existing engine-driven KV-cache prefix
  matcher). The tree models exactly what is currently cached on that
  replica, kept faithful by the colocation event loop's
  `insert`/`remove` calls. There is no per-replica "owner set" on
  tree nodes — a replica P "owns" prefix σ ⇔ σ exists in P's tree.
- **State B — per-pod.** Lives in `pod_load: usize` and
  `pod_cost: f64`, both derived from a single
  `SlidingWindow<f64, Sum>` (3 min) that holds per-request cost
  contributions. `pod_load = window.len()` (count is free from the
  deque after expire); `pod_cost = window.aggregate().0` (the
  incrementally-maintained `Sum`). Both are O(1) reads.

The composition lives in `router/src/scheduler/policies/preble/
block_hash.rs` as `PrebleBlockHash`. Under `--features preble-q`
the `PrefixBlockHash` alias points at this type
(`router/src/scheduler/kvcache.rs`), so the colocation event loop's
existing `BlockHash` calls keep the tree faithful, while the Preble
DSL helpers read `pod_load` / `pod_cost` via inherent methods on the
concrete type.

`SlidingWindow<T, A: Aggregate>` is a generic primitive in
`radixtree/src/types/sliding_window.rs`. Future policies that need
"sum/count over recent N seconds" reuse it.

### Paper alignment of the implementation

- `pod_load(P) = |W_P|` matches paper §3.2 directly. No L2
  over-counting.
- `pod_cost(P) = Σ_{r ∈ W_P}(PT_r + DT_r)` matches the paper's
  `L_i = Σ_{r∈W}(PT_r + DT_r)` formula — per-request prefill+decode
  contributions attributed to the routed replica.
- The paper §4.1 explicitly says *"the global scheduler maintains a
  current load count for each GPU by updating it every time a new
  request is assigned to it or when it evicts a tree node"* —
  per-GPU scalars, incrementally maintained. blitz-router's
  `SlidingWindow.push` (on assign) + lazy `expire` (on read) is
  exactly that.
- The 5-min LRU subtree drop and the 1 Hz background eviction loop
  are not implemented; engine SSE feedback (`evicted_block_ids`)
  drives tree state directly, which is closer to ground truth than
  the time-based stale guess.

---

## Notes on alphabet and topology (Go vs blitz-router)

The abstract operations above are stated over an unspecified alphabet.

- **Go reference** uses the *token alphabet*. tdepth is in tokens.
  The tree is path-compressed.
- **blitz-router** uses the *block-hash alphabet*. Each edge carries
  one block hash; the token interpretation is recovered externally
  by multiplying by `block_size` in the policy layer. The
  implementation reuses the existing
  `radixtree::RadixTreeBlockHash` (Verus-derived L3 Patricia trie),
  the same type the rest of the router uses for KV-cache prefix
  matching.

The two trees have different alphabets but represent the same
prefix → cached-state map modulo block-alignment slack at the tail
(up to `block_size − 1` tokens dropped from incomplete final blocks,
since `BlockHashState` only hashes complete blocks). Semantic
equivalence on Preble's queries is preserved; topological identity
is not.

---
