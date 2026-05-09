# Scheduling Policy DSL — Schema

This document specifies (a) the DSL surface syntax and (b) the typed schema (Request / ScheduleContext / GlobalContext / named functions / reducers) every policy lowers against. Companion docs:

- `policies.md` — canonical DSL listings for every policy (the catalog).
- `implementation.md` — how the `policy!` proc macro lowers DSL into `impl Policy` (rewrite table + lint).

## 1. Motivation

Each policy is a single `policy! { ... }` invocation — typically ≤ 30 lines — that the proc macro lowers into an `impl Policy for X` block. The macro enforces an allowlist lint (`implementation.md` §2.2) so every impl corresponds 1:1 to a spec-form DSL listing in `policies.md` §2 via the rewrite table in `implementation.md` §2.1. The gap between *what a policy claims to compute* and *what the deployed code actually computes* collapses to a mechanical codegen pass that is audited once. Reviewers can hold an entire policy in working memory.

## 2. Surface syntax

Five constructs:

```
<expr>     ::= Filter <pred> <expr> <expr>          -- lossless three-arg
             | Select min|max|rand by <fn>           -- list-to-one combinator
             | With <name> = <reducer-expr>
                  [, <name> = <reducer-expr>]* in <expr>

<pred>     ::= <bool-expr over sctx.f, gctx.f, req.f, named-fn(...), bound τ̄>
<fn>       ::= <T-expr   over sctx.f, gctx.f, req.f, named-fn(...), bound τ̄>

<reducer>  ::= Mean .field | Std .field | Sum .field
             | Min .field  | Max .field | Count
             | <arith over reducers and constants>

<after>    ::= after: default
             | after: default; <stmt> [; <stmt>]*
             | after: <stmt> [; <stmt>]*
<stmt>     ::= gctx.f <- <expr>
             | sctx.lmetric.f += <expr>
             | entry.f := <expr>
```

A `policy` declaration:

```
policy <name>-q (gctx: <GlobalContextType>):
    <expr>
    <after>
```

For policies with no global state: `gctx: ()` (the unit type).

Determinism is syntactic: a policy is deterministic iff its `<expr>` contains no `Select rand`. The codegen reads this once at compile time and emits the appropriate path; there is no runtime tag.

`Filter <pred> <expr_in> <expr_out>` semantics:
- Partition `[sctx]` by `<pred>(sctx, ...)`.
- If the passing set is non-empty: evaluate `<expr_in>` on it.
- If the passing set is empty (lossless fallback): evaluate `<expr_out>` on the rejected set (which equals `[sctx]`).

The fallback guarantees no request is dropped due to filter starvation. `<expr_in>` and `<expr_out>` are required to be syntactically distinct in all but trivial cases — see §8.

## 3. Three scopes (data layout)

| Scope | Type | Cardinality | Lifetime | Mutation channels |
|---|---|---|---|---|
| **Request** | `&ValidGenerateRequest` | 1 per call | one schedule | none (read-only input) |
| **ScheduleContext** | `&[Arc<Mutex<ScheduleContext>>]` | N (one per replica) | live across calls | `apply_schedule_decision` (commit-driven) + `colocation::handle_metric` (SSE-driven) |
| **GlobalContext** | `&mut Self::GlobalContext` | 1 per system | live across calls | `after:` clause only |

Scope is what makes the design markovian: state lives in exactly one scope and one transition relation per scope. `ScheduleContext` and `GlobalContext` together form the system state at scheduling time.

## 4. Field schema

### 4.1 Request fields exposed to DSL

| Name | Rust source | Type | Semantics |
|---|---|---|---|
| `req.input_tokens` | `request.input_tokens` | `&[u32]` | tokenized prompt |
| `req.tokens` | `request.input_tokens.len()` | `usize` | token count (sugar) |
| `req.id` | `request.request_id` | `u64` | request ID |

### 4.2 ScheduleContext fields exposed to DSL

| Name | Rust source | Type | Semantics |
|---|---|---|---|
| `sctx.bs` | `lmetric.bs` | `usize` | active batch size (running + queued) |
| `sctx.waiting` | `lmetric.waiting_reqs` | `usize` | requests in waiting queue |
| `sctx.all_tokens` | `lmetric.all_tokens` | `usize` | total active tokens (prefill + decode) |
| `sctx.idx` | implicit replica index | `usize` | this replica's index in `[sctx]` |
| `sctx.block_hash` | `block_hash` field | `&dyn BlockHash` | radix tree (only via named fns, not raw access) |
| `sctx.block_size` | `block_hash.block_size()` | `usize` | per-engine block size |

Notes:
- `sctx.block_hash` is **not** a directly readable DSL field — the radix tree is large and policy `<fn>`s should not iterate it. Access is mediated by the named functions in §5 (`hit_blocks`, `match_blocks`, etc.), each of which performs exactly one `block_hash.get(...)` call.
- The lmetric backing field `queued_tokens: isize` (in `Observation`; backed by `lmetric.prefill_tokens`) is **deliberately NOT in this table**. Canonical DSL access is the `queued_tokens(sctx) → usize` named-fn (§5), which clamps to 0 and casts. The struct field can transiently be negative under commit/return races (hence `isize`); direct access bypasses the clamp, and the type-system mismatch (`isize + usize`) catches most misuse (`sctx.queued_tokens + sctx.all_tokens` does not typecheck). Name shadow: `sctx.queued_tokens` (field, `isize`, impl-form-legal but discouraged) ≠ `queued_tokens(sctx)` (fn, `usize`, spec-form canonical) — `policies.md` §2 listings use only the latter.
- `sctx.idx` is implicit: the codegen iterating over `[sctx]` knows the index without policy declaration.

### 4.3 GlobalContext field schema

`GlobalContext` is **per-policy**. Each stateful policy defines its own struct. Stateless policies use `()`.

Currently defined:

| Policy | GlobalContext struct | Fields |
|---|---|---|
| `round-robin-q` | `RRGCtx` | `next_replica_id: usize` |
| `preble-q` | `PrebleGCtx` | `H: SlidingWindowHistogram` |
| all others | `()` | (none) |

`Count` (number of replicas) is a system constant accessible from `<fn>`/`<pred>` without going through gctx; codegen substitutes the `[sctx].len()` at the right call site.

## 5. Named pure function library

These functions are the *only* way `<fn>`/`<pred>` may consume per-request × per-replica information that requires more than a single field read. Every function is pure, deterministic, and reads finitely many fields from its arguments.

| Name | Signature | Body | Reads |
|---|---|---|---|
| `new_tokens(req, sctx)` | `(Req, ScheduleContext) → usize` | `req.tokens - hit_blocks(req, sctx) * sctx.block_size` | `req.input_tokens`, `sctx.block_hash`, `sctx.block_size` |
| `new_blocks(req, sctx)` | `(Req, ScheduleContext) → usize` | `req.tokens.div_ceil(sctx.block_size).saturating_sub(hit_blocks(req, sctx))` (returns 0 if `block_size == 0`) | `req.input_tokens`, `sctx.block_hash`, `sctx.block_size` |
| `queued_tokens(sctx)` | `ScheduleContext → usize` | `sctx.queued_tokens.max(0) as usize` | `sctx.queued_tokens` (the `isize` `Observation` field — note name shadows this fn; see §4.2 Notes) |
| `prefill_tokens(req, sctx)` | `(Req, ScheduleContext) → usize` | `queued_tokens(sctx) + new_tokens(req, sctx)` | (composition) |
| `hit_blocks(req, sctx)` | `(Req, ScheduleContext) → usize` | `sctx.block_hash.get(req.block_hash_state.get_hashes())` | `req.block_hash_state`, `sctx.block_hash` |
| `hit_pct(req, sctx)` | `(Req, ScheduleContext) → f32` | `(hit_blocks(req, sctx) * sctx.block_size) as f32 / req.tokens as f32` | (composition) |
| `match_blocks(req, sctx)` | `(Req, ScheduleContext) → usize` | alias for `hit_blocks` (Preble-flavored naming) | (composition) |
| `decode_blocks(sctx)` | `ScheduleContext → usize` | `sctx.all_tokens / sctx.block_size` | `sctx.all_tokens`, `sctx.block_size` |
| `preble_cost(req, sctx, gctx_H)` | `(Req, ScheduleContext, &SlidingWindowHistogram) → f32` | per Preble cost formula (see `policies/preble/cost_model.rs`) | request features, replica load, histogram state |

This list is **closed** for the initial DSL. Adding a new function requires a separate review (it expands the per-policy reachability set).

## 6. Reducer vocabulary

Closed set of six. Each takes a projection over `[ScheduleContext]` and returns a scalar. The projection may be either:

- a `.field` access (a §4.2 ScheduleContext field), e.g. `Min .bs`, or
- a §5 named-fn applied with the per-replica `sctx`, e.g. `Min hit_blocks(req, ·)`. The `·` denotes the iteration position; `req` (and any other top-level arg) is captured from the surrounding policy scope.

Both projection forms iterate over `[sctx]` and return a single scalar. The named-fn form is what `bailian-impl-q` and `most-hit-load-q` rely on for per-component min-max normalization across candidate replicas.

| Reducer | Type | Body |
|---|---|---|
| `Mean .f` | `f32` | `sum(s.f for s in sctxs) / sctxs.len()` |
| `Std .f` | `f32` | sample stddev (n-1 denominator, matching aibrix) |
| `Sum .f` | numeric | sum |
| `Min .f` | numeric | min |
| `Max .f` | numeric | max |
| `Count` | `usize` | `sctxs.len()` (no projection) |

Reducers compose via a restricted arithmetic algebra: `+ − × ÷` over reducers and float/int constants. **No** non-reducer function calls inside `<reducer-expr>`. Reducers cannot reference `req` or `gctx` (they are pure functions of `[sctx]`).

`With` introduces named bindings into the lexical scope of `<expr>`:

```
With τ = Mean .bs + 1.0 · Std .bs in
  Filter (sctx.bs ≤ τ) ...
```

This makes the cross-replica aggregate `τ` *visible* to `<pred>`/`<fn>` without permitting them to compute aggregates themselves.

`With` bindings are **computed once per schedule call** and broadcast to every per-replica `<fn>`/`<pred>` invocation — they are not re-evaluated per replica. (See §10.1 for the rationale.)

## 7. Algebra bounds for `<fn>` and `<pred>`

`<fn>`, `<pred>` expressions are restricted to:

- Field reads: `sctx.f`, `gctx.f`, `req.f`, `bound_τ`
- Constants: integer / float literals, named constants (e.g. `BOUND`, `BAILIAN_ALPHA`, `IMBALANCE_LIMIT`)
- Named function calls from §5 with the exact signatures listed
- Arithmetic: `+ − × ÷ % == ≠ ≤ ≥ < >`, boolean `and or not`
- Tuple construction: `(a, b)`, `(a, b, c)`. Tuples compare lexicographically. (`AssignScore<M, W>` already does this; the DSL preserves the convention.)
- Conditional: `if <pred> then <expr> else <expr>` (no general `match`, no recursion, no closures)

Explicitly **forbidden**: any iteration over `[sctx]` (use a reducer in `With`); any direct access to `sctx.block_hash` (use a named fn); any function call not in §5; allocation / I/O.

This is the audit surface: any DSL `<fn>` is a finite straight-line expression over the schema. The codegen lowers it to a function with no allocations and no calls outside the named-fn library.

## 8. `after:` clause static check

Codegen rejects any policy whose `after:` body does not satisfy at least one of:

1. **Default-shorthand**: body syntactically begins with the `default` keyword (optionally followed by additional statements: `after: default; <stmt>; <stmt>`).
2. **Full enumeration**: body contains all six canonical mutations explicitly:
   - `sctx.lmetric.bs += 1`
   - `sctx.lmetric.waiting_reqs += 1`
   - `sctx.lmetric.prefill_tokens += new_tokens(req, sctx)`
   - `sctx.lmetric.all_tokens += req.tokens`
   - `entry.pred_hits := hit_blocks(req, sctx)`  *(or the cached version when epoch-skip applies; codegen handles this)*
   - `entry.epoch := sctx.block_hash.epoch`

A policy can extend `default` (option 1) with arbitrary additional statements (e.g. preble's `gctx.H` update), but cannot omit any of the six.

The check is purely syntactic at the AST level; no semantic equivalence reasoning is required.

## 9. Out of scope

The DSL governs **commit-driven** state transitions only (synchronous post-decision updates expressed in `after:`).

It does **not** govern:

- **SSE-driven mutations** to `sctx.block_hash` (insert / remove / alias-bid tracking) — these are triggered by yaullm step events in `colocation::handle_metric` and are entirely independent of policy choice. Their correctness is the scope of `verify_staleness` + the alias-bid soundness of `RadixTreeBlockHash` (see issue #10 for the canonical example of this layering boundary).
- **Cross-system state machine** of replica lifecycle (`Inactive`, `LoadingPrefill`, `Prefill`, etc.) — these are governed by `spec/abort-recovery/AbortRecovery.tla` and `colocation.rs`, not by policy DSL.
- **Validation, tokenization, batching** — pre- and post-scheduling pipeline stages.

Conversely, anything inside `after:` that mutates `sctx` or `gctx` IS under DSL governance and subject to the §8 static check.

## 10. Decisions

1. **`With` binding scope**: bindings are computed **once per schedule call** and broadcast to every per-replica `<fn>`/`<pred>` evaluation. Reducers operate over `[sctx]`, so computing them once is the only sound option (per-replica re-evaluation would either be redundant or, if the projection ever depended on the iterating replica, change the reducer's meaning).

2. **`chosen` binding in `after:`**: `chosen: usize` (replica index). Rationale: of the policies in `policies.md`, none has an `after:` body that *reads* `chosen.f` — `default` *writes* sctx fields, and preble uses `chosen` as an opaque ID into `gctx.H.insert(chosen, req)`. Choosing `&ScheduleContext` would require holding the scoring-time lock guard across `after:` evaluation, which introduces lock-across-await hazards for any future stateful policy whose `after:` body itself awaits (preble's histogram update is a candidate). With `usize`, codegen synthesizes a fresh lock acquisition for any `after:` statement that needs to read sctx fields — the same pattern `apply_schedule_decision` already uses for the framework default block.

3. **Tuple comparison via negation trick**: keep. `Select min by (-hit_pct(req, sctx), sctx.bs)` is the canonical form for "primary key max, secondary key min" in aibrix-q. No `tiebreak` keyword introduced.

4. **Tunable constant location**: hybrid (a) + cfg-gated (c).
   - **(a) Storage**: every tunable lives as `pub static` in `router/src/metrics.rs` (or a dedicated `policies/constants.rs` if the list grows). This is the single source of truth and the value at build time.
   - **(c) Override**: the CLI parser conditionally registers `--<tunable-name> <type>` arguments under `#[cfg(feature = "<policy>-q")]`. Builds without that policy's feature flag never see the corresponding CLI argument and never carry the override code path. This matches the `BAILIAN_ALPHA`/`BAILIAN_BETA`/`BAILIAN_GAMMA` placement in `metrics.rs:25-27` while adding a startup-time override channel for ablation without recompilation.

5. **Empty `[sctx]`**: codegen emits `panic!("schedule called with empty replica set")` for any reducer or selector that would dereference an empty list. The framework's lossless admission control guarantees `[sctx].len() ≥ 1` at every `schedule()` invocation; violating this is a contract break, not a recoverable error.

6. **`dynamo-po-q` = prefill-only**: the prefill term in the score (`w · new_tokens(req, sctx)`) contains *only* per-request uncached tokens, decoupled from the engine's queued prefill state (which is folded into the separate `+ sctx.all_tokens` term). Contrast with `dynamo-q` whose prefill term is `w · prefill_tokens(req, sctx) = w · (queued + new)` — engine state is mixed *into* the prefill scoring. The "po" suffix names this contrast.
