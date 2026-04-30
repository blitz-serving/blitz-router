# Scheduling Policy DSL — Schema & Trait Specification

> **Status**: draft, awaiting review before Phase 2 (proc-macro implementation) begins.
>
> **Scope**: this document specifies (a) the DSL surface syntax, (b) the typed schema (Request / ScheduleContext / GlobalContext / named functions / reducers) every policy lowers against, and (c) the single Rust trait that codegen targets. It does **not** cover the proc-macro implementation, the static checker, or per-policy migration order — those live in the implementation PLAN.

## 1. Motivation

The current `router/src/policies/` is ~880 LOC of handwritten policy implementations sitting on ~270 LOC of trait scaffolding (`QueuePlusPlus`, `AssignScore`, `NaiiveLattice`, `DeterministicPolicy`, `StochasticPolicy`, `ScheduleStep`, `SamplerFn`, plus dispatch helpers). Per-policy review requires holding ~80 LOC of trait impls in your head and tracing through the dispatch maze in `mod.rs`.

The DSL replaces this with ~5 lines per policy + a single ~5-line trait. Reviewers can hold an entire policy in working memory; the paper's algorithm boxes become byte-identical to the DSL listings; the gap between *what we claim a policy does* and *what the deployed code actually computes* collapses to a mechanical codegen pass that is itself audited once.

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

The fallback guarantees no request is dropped due to filter starvation. `<expr_in>` and `<expr_out>` are required to be syntactically distinct in all but trivial cases — see §10.

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
| `sctx.queued_pre` | `lmetric.prefill_tokens` | `usize` | queued prefill tokens |
| `sctx.all_tokens` | `lmetric.all_tokens` | `usize` | total active tokens (prefill + decode) |
| `sctx.idx` | implicit replica index | `usize` | this replica's index in `[sctx]` |
| `sctx.block_hash` | `block_hash` field | `&dyn BlockHash` | radix tree (only via named fns, not raw access) |
| `sctx.block_size` | `block_hash.block_size()` | `usize` | per-engine block size |

Notes:
- `sctx.block_hash` is **not** a directly readable DSL field — the radix tree is large and policy `<fn>`s should not iterate it. Access is mediated by the named functions in §5 (`hit_blocks`, `match_blocks`, etc.), each of which performs exactly one `block_hash.get(...)` call.
- `sctx.idx` is implicit: the codegen iterating over `[sctx]` knows the index without policy declaration.

### 4.3 GlobalContext field schema

`GlobalContext` is **per-policy**. Each stateful policy defines its own struct. Stateless policies use `()`.

Currently anticipated:

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
| `queued_tokens(sctx)` | `ScheduleContext → usize` | `sctx.queued_pre` | `sctx.queued_pre` |
| `prefill_tokens(req, sctx)` | `(Req, ScheduleContext) → usize` | `queued_tokens(sctx) + new_tokens(req, sctx)` | (composition) |
| `hit_blocks(req, sctx)` | `(Req, ScheduleContext) → usize` | `sctx.block_hash.get(req.block_hash_state.get_hashes())` | `req.block_hash_state`, `sctx.block_hash` |
| `hit_pct(req, sctx)` | `(Req, ScheduleContext) → f32` | `(hit_blocks(req, sctx) * sctx.block_size) as f32 / req.tokens as f32` | (composition) |
| `match_blocks(req, sctx)` | `(Req, ScheduleContext) → usize` | alias for `hit_blocks` (Preble-flavored naming) | (composition) |
| `decode_blocks(sctx)` | `ScheduleContext → usize` | `sctx.all_tokens / sctx.block_size` | `sctx.all_tokens`, `sctx.block_size` |
| `preble_cost(req, sctx, gctx_H)` | `(Req, ScheduleContext, &SlidingWindowHistogram) → f32` | per Preble cost formula (see `policies/preble/cost_model.rs`) | request features, replica load, histogram state |

This list is **closed** for the initial DSL. Adding a new function requires a separate review (it expands the per-policy reachability set).

## 6. Reducer vocabulary

Closed set of six. Each takes a projection `.field` over `[ScheduleContext]` and returns a scalar.

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

`With` bindings are **computed once per schedule call** and broadcast to every per-replica `<fn>`/`<pred>` invocation — they are not re-evaluated per replica. (See §12.1 for the rationale.)

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

## 8. The 11 policies (canonical DSL listings)

```
policy random-q (gctx: ()):
    Select rand by 1
    after: default

policy round-robin-q (gctx: RRGCtx):
    Select min by (if sctx.idx == gctx.next_replica_id then 0 else 1)
    after: default; gctx.next_replica_id <- (gctx.next_replica_id + 1) % Count

policy join-shortest-q (gctx: ()):
    Select min by 4 · sctx.waiting + sctx.bs
    after: default

policy least-wait-token-q (gctx: ()):
    Select min by prefill_tokens(req, sctx)
    after: default

policy bounded-most-hit-q (gctx: ()):
    Filter (sctx.queued_pre < BOUND)
      (Select max by hit_blocks(req, sctx))
      (Select min by prefill_tokens(req, sctx))
    after: default

policy dynamo-q (gctx: ()):
    Select min by w · (prefill_tokens(req, sctx) / sctx.block_size)
                  + decode_blocks(sctx)
    after: default

policy dynamo-po-q (gctx: ()):                          # was dynamo-decoupled-q
    Select min by w · new_tokens(req, sctx) + sctx.all_tokens
    after: default

policy lmetric-q (gctx: ()):
    Select min by prefill_tokens(req, sctx) · (sctx.bs + 1)
    after: default

policy bailian-impl-q (gctx: ()):
    With M_bs  = Max .bs,
         M_tok = Max .all_tokens in
    Select rand by α · hit_pct(req, sctx)
                  + β · (1 - sctx.bs / M_bs)
                  + γ · (1 - sctx.all_tokens / M_tok)
    after: default

policy aibrix-q (gctx: ()):
    With τ   = Mean .bs + 1.0 · Std .bs,
         lo  = Min .bs,
         gap = Max .bs - lo in
    Filter (sctx.bs == lo and gap > 8)
      (Filter (sctx.bs ≤ τ)
         (Select min by (-hit_pct(req, sctx), sctx.bs))
         (Select max by (-hit_pct(req, sctx), sctx.bs)))
      (Filter (sctx.bs ≤ τ)
         (Select min by (-hit_pct(req, sctx), sctx.bs))
         (Select max by (-hit_pct(req, sctx), sctx.bs)))

policy preble-q (gctx: PrebleGCtx):
    Filter (match_blocks(req, sctx) / req.tokens > 0.5)
      (Select max by match_blocks(req, sctx))
      (Select min by preble_cost(req, sctx, gctx.H))
    after: default; gctx.H <- gctx.H.insert(chosen, req)
```

`chosen` in the `after:` clause is the `usize` index of the selected replica (see §12.2). It is a reserved name; codegen binds it after the `<expr>` evaluates.

Note: `bounded-most-hit-q` differs from current `policies/bounded_most_hit.rs` by routing the fallback to a least-wait-token branch (per the "attention black hole" fix), not to identical max-hits scoring. Migration of this policy is a behavior change, not a no-op refactor.

## 9. Lowering target — the `Policy` trait

```rust
pub(crate) trait Policy {
    type GlobalContext: Default + Send + Sync + 'static;

    fn schedule(
        req: &ValidGenerateRequest,
        sctxs: &[Arc<Mutex<ScheduleContext>>],
        gctx: &mut Self::GlobalContext,
    ) -> impl Future<Output = Option<usize>> + Send;
}
```

That's the entire policy-side surface. Codegen produces `impl Policy for <Name>Q { fn schedule(...) { /* generated body */ } }` per policy.

The framework dispatches a single `TaskAssigner = QueueRunner<P: Policy>` (selected via cargo feature gating, exactly as today). The 270 LOC of trait scaffolding (`QueuePlusPlus`, `AssignScore`, `NaiiveLattice`, marker traits, sampler indirection) goes away.

### `simple.rs` convention

Trivial policies — those whose DSL expression fits in a single `Select` or simple `Filter` and that have no e2e test dependencies — live together in `router/src/policies/simple.rs` instead of one file each. Initial residents (subject to migration):

- `random-q`
- `round-robin-q`
- `join-shortest-q`
- `least-wait-token-q`
- `bounded-most-hit-q`

Non-simple policies retain their own files: `dynamo/`, `preble/`, `bailian.rs`, `aibrix/`, `lmetric.rs`.

The codegen treats `simple.rs` and per-policy files identically; the partition is for human ergonomics only (review burden, drift visibility).

## 10. `after:` clause static check

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

## 11. Out of scope

The DSL governs **commit-driven** state transitions only (synchronous post-decision updates expressed in `after:`).

It does **not** govern:

- **SSE-driven mutations** to `sctx.block_hash` (insert / remove / alias-bid tracking) — these are triggered by yaullm step events in `colocation::handle_metric` and are entirely independent of policy choice. Their correctness is the scope of `verify_staleness` + the alias-bid soundness of `RadixTreeBlockHash` (see issue #10 for the canonical example of this layering boundary).
- **Cross-system state machine** of replica lifecycle (`Inactive`, `LoadingPrefill`, `Prefill`, etc.) — these are governed by `formal/tlaplus/CompletionLoop.tla` and `colocation.rs`, not by policy DSL.
- **Validation, tokenization, batching** — pre- and post-scheduling pipeline stages.

Conversely, anything inside `after:` that mutates `sctx` or `gctx` IS under DSL governance and subject to the §10 static check.

## 12. Decisions (resolved during review)

1. **`With` binding scope**: bindings are computed **once per schedule call** and broadcast to every per-replica `<fn>`/`<pred>` evaluation. Reducers operate over `[sctx]`, so computing them once is the only sound option (per-replica re-evaluation would either be redundant or, if the projection ever depended on the iterating replica, change the reducer's meaning).

2. **`chosen` binding in `after:`**: `chosen: usize` (replica index). Rationale: of the 11 policies, none has an `after:` body that *reads* `chosen.f` — `default` *writes* sctx fields, and preble uses `chosen` as an opaque ID into `gctx.H.insert(chosen, req)`. Choosing `&ScheduleContext` would require holding the scoring-time lock guard across `after:` evaluation, which introduces lock-across-await hazards for any future stateful policy whose `after:` body itself awaits (preble's histogram update is a candidate). With `usize`, codegen synthesizes a fresh lock acquisition for any `after:` statement that needs to read sctx fields — the same pattern `apply_schedule_decision` already uses for the framework default block.

3. **Tuple comparison via negation trick**: keep. `Select min by (-hit_pct(req, sctx), sctx.bs)` is the canonical form for "primary key max, secondary key min" in aibrix-q. No `tiebreak` keyword introduced.

4. **Tunable constant location**: hybrid (a) + cfg-gated (c).
   - **(a) Storage**: every tunable lives as `pub static` in `router/src/metrics.rs` (or a dedicated `policies/constants.rs` if the list grows). This is the single source of truth and the value at build time.
   - **(c) Override**: the CLI parser conditionally registers `--<tunable-name> <type>` arguments under `#[cfg(feature = "<policy>-q")]`. Builds without that policy's feature flag never see the corresponding CLI argument and never carry the override code path. This matches the current `BAILIAN_ALPHA`/`BAILIAN_BETA`/`BAILIAN_GAMMA` placement in `metrics.rs:25-27` while adding a startup-time override channel that the camera-ready paper can use for ablation without recompilation.

5. **Empty `[sctx]`**: codegen emits `panic!("schedule called with empty replica set")` for any reducer or selector that would dereference an empty list. The framework's lossless admission control guarantees `[sctx].len() ≥ 1` at every `schedule()` invocation; violating this is a contract break, not a recoverable error.

6. **`dynamo-po-q` = prefill-only**: the cargo feature renames from `dynamo-decoupled-q` to `dynamo-po-q`. Semantics: the prefill term in the score (`w · new_tokens(req, sctx)`) contains *only* per-request uncached tokens, decoupled from the engine's queued prefill state (which is folded into the separate `+ sctx.all_tokens` term). Contrast with `dynamo-q` whose prefill term is `w · prefill_tokens(req, sctx) = w · (queued + new)` — engine state is mixed *into* the prefill scoring. The "po" suffix names this contrast.

---

Reviewer to check: §3 (scope cardinalities), §5 (named-fn library completeness), §6 (reducer set sufficiency for all 11 policies), §8 (DSL listings against intended algorithm), §9 (trait signature), §10 (static check formulation). Decisions in §12 are settled. Approval of this document gates Phase 2 (proc-macro implementation).
