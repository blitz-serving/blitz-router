# Scheduling Policy DSL — Implementation

This document specifies (a) the single Rust trait that codegen targets and (b) the proc-macro implementation surface (rewrite table + lint allowlist) that maps spec-form DSL to impl-form Rust. For the DSL surface syntax + schema see `schema.md`; for canonical per-policy listings see `policies.md`.

## 1. Lowering target — the `Policy` trait

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

That's the entire policy-side surface. Codegen produces `impl Policy for <Name>Q { fn schedule(...) { /* generated body */ } }` per policy. The framework dispatches a single `TaskAssigner = PolicyRunner<P: Policy>` selected via cargo feature gating.

## 2. Implementation surface and DSL ↔ impl interchange

The `policy!` proc macro does **not** parse the spec-form DSL syntax (`schema.md` §2) literally. Instead, the macro accepts a Rust-syntactic body restricted to a closed allowlist of constructs, and the spec-form DSL ↔ Rust impl correspondence is a **fixed mechanical rewrite** that a reviewer can apply in either direction. The macro is ~150 LOC of allowlist lint plus boilerplate emission rather than a recursive-descent parser, trading surface-syntax fidelity for implementation simplicity.

### 2.1 Rewrite table

Each row is a 1:1 syntactic correspondence between the spec-form DSL (left, what `policies.md` §2 prints) and the implementation-form Rust accepted inside `policy!` (right, what the source actually contains). The mapping is total in both directions: every spec construct has exactly one impl form, and the macro lint (§2.2) rejects any impl construct not in this table.

| # | Spec DSL form | Implementation form |
|---|---|---|
| 1 | `Filter (P) { A } else { B }` | `filter_then(target, \|o\| P, \|t\| { A }, \|t\| { B })` |
| 2 | `Select min by F` | `select_min_by(target, \|o\| F)` |
| 3 | `Select max by F` | `select_max_by(target, \|o\| F)` |
| 4 | `Select rand by F` | `select_rand_by(target, \|o\| F)` |
| 5 | `With τ = E in body` | `{ let τ = E; body }` |
| 6 | `Mean .f` | `mean_of_usize(&observations, \|o\| o.f)` |
| 7 | `Std .f` | `std_of_usize(&observations, \|o\| o.f)` |
| 8 | `Sum .f` | `sum_of_usize(&observations, \|o\| o.f)` |
| 9 | `Min .f` | `min_of_usize(&observations, \|o\| o.f)` |
| 9b | `Min named_fn(req, ·)` | `min_of_usize(&observations, \|o\| named_fn(req, o))` |
| 10 | `Max .f` | `max_of_usize(&observations, \|o\| o.f)` |
| 10b | `Max named_fn(req, ·)` | `max_of_usize(&observations, \|o\| named_fn(req, o))` |
| 11 | `Count` | `observations.len()` |
| 12 | `after default` | (auto-emitted by macro; absent from body) |
| 13 | `after default; gctx.X ← E` | `policy! { ..., after { gctx.X = E; } }` |

`target` is the bound name of the current candidate set inside the closure scope; for the outermost expression it is `&observations`. `t` is the inner candidate set inside `filter_then`'s on_pass / on_fail branches. `req` and `gctx` are bound at the top of the generated `schedule` body.

### 2.2 Proc-macro lint (allowlist)

Inside the `policy!` body, the macro accepts only:

- Function calls to the **closed allowlist**:
  - Combinators: `filter_then`, `select_min_by`, `select_max_by`, `select_rand_by`
  - Reducers: `mean_of_usize`, `std_of_usize`, `sum_of_usize`, `min_of_usize`, `max_of_usize`
  - Named pure fns (`schema.md` §5): `new_tokens`, `new_blocks`, `queued_tokens`, `prefill_tokens`, `hit_blocks`, `match_blocks`, `hit_pct`, `decode_blocks`, `preble_cost`
- `let` bindings (any name, any RHS satisfying these rules transitively)
- Closures `|name| ...` and `|name1, name2, ...| ...`
- Field access: `o.bs`, `req.input_tokens`, `gctx.field`, `t[idx]`, etc.
- Method calls on `&[Observation]` / `Option`: `.len()`, `.is_empty()`, `.as_ref()`, etc. (also allowlisted)
- Arithmetic and comparison operators
- Boolean operators
- Tuple construction
- `if`/`else` expressions
- Block expressions `{ ... }`
- Numeric and string literals; named constants imported via `use`

The macro **rejects** (with a compile error pointing to the offending source line):

- Any function call not in the allowlist (catches `unsafe { ... }` calls, arbitrary stdlib usage, custom helpers smuggled in via `use`)
- `for` / `while` / `loop` constructs
- `match` (use `if`/`else` chains via `select_*_by`'s comparator instead)
- `unsafe` blocks
- `return` / `break` / `continue`
- Macro invocations other than `policy!` itself

The lint is a `syn::visit::Visit` walk of the body's `syn::Expr` tree. Implementation lives in `policy-dsl/src/check.rs`.

#### 2.2.1 Known gap: function items as path arguments

The current lint overrides `visit_expr_call` only. That catches every `foo(args)` expression and checks `foo`'s last path segment against the allowlist. It does **not** catch function items passed as a *value* — i.e. as a path argument to a higher-order combinator:

```rust
// Both forms appear in real policies:
select_min_by(t, preble_cost)                    // function ITEM passed as arg
select_min_by(t, |o| preble_cost(o))             // function CALL inside closure
```

The first form (function item) is an `Expr::Path` in argument position, never visited by `visit_expr_call`. `preble_cost` is never checked against `ALLOWED_FNS`. The second form goes through `visit_expr_call` normally.

In practice this is mitigated by Rust's type system: the function item must satisfy the combinator's higher-order parameter type (e.g. `Fn(&Observation) -> S: PartialOrd`), which sharply limits what can be passed. But the allowlist invariant ("any legal body mechanically maps back to a spec-form DSL expression") is weakened — a reviewer cannot assume that every named helper appearing in a body has been audited against `ALLOWED_FNS`. They must additionally check path-argument positions by hand.

**Convention until this is fixed**: every helper that may be passed as a function item must also be added to `ALLOWED_FNS`. This is currently honored by `preble_cost`, `preble_load`, `preble_bs_sum`, `preble_tps_count`, `preble_owned_match_blocks` — all reachable as both call expressions and as path arguments to `select_*_by`. The convention is by author discipline, not by the linter.

**Fix design (deferred)**: extend `Linter` with a `visit_expr` override that, when seeing an `Expr::Path` in argument position of a known higher-order combinator, applies the same allowlist check as `visit_expr_call`. The hard part is identifying which call-argument positions are "higher-order" — naive coverage (every path argument) would reject `preble_global_match_blocks(observations, prefix)` (where `prefix` is a path to a local binding). The cleanest carve-out is a per-combinator argument-position whitelist: e.g. `select_min_by`'s second argument is higher-order, `filter_then`'s arguments 2/3/4 are higher-order; everything else is data. This is mechanical but adds ~50 LOC to `check.rs`.

### 2.3 Reverse interchange (impl → DSL)

A reviewer auditing a `policy!` invocation reads the Rust body, applies §2.1's table right-to-left mechanically, and recovers the spec-form DSL. The lint guarantees this is always well-defined: every legal body is built from table entries, and each entry has a unique spec form. There is no construct in the impl that "cannot be expressed in DSL" — the lint rejects such bodies at compile time.

This eliminates the standard "is the implementation faithful to the spec?" trust gap: instead of relying on author discipline, the macro enforces faithfulness statically. Readers of `policies.md` §2 and readers of `router/src/policies/*.rs` are looking at the same machine, with the rewrite table as the public bijection between the two views.

### 2.4 Where each piece lives

- **Runtime helpers** (combinators, reducers, named pure fns, `apply_default_after`): `router/src/policies/dsl_runtime.rs`. Audited once, depended on by every policy.
- **`Policy` trait**: `router/src/policies/policy_trait.rs`. 5 lines.
- **`policy!` macro** (parse + lint + emit): `policy-dsl/`. ~150 LOC total.
- **Per-policy invocations**: `router/src/policies/{simple,vllm,bailian,aibrix,dynamo,lmetric}.rs`, `router/src/policies/preble/`, `router/src/policies/llm_d/{most_hit,most_hit_load,most_hit_load_active,least_active,least_bs,least_token_load,least_waiting}.rs`. Each ≤ 30 lines.
- **`PolicyRunner<P: Policy>`** (queue management dispatching to `P::schedule`): `router/src/policies/policy_runner.rs`.
