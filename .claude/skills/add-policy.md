---
name: add-policy
description: Add a new scheduling policy to blitz-router (cargo feature + policy! invocation + all required doc surfaces). Use when the user asks to add/port/implement a new policy.
---

# Adding a Scheduling Policy

This is the canonical workflow for adding one new policy. Mirrors the "Policy added / renamed / deleted" entry of the Doc-Code Consistency checklist in `CLAUDE.md`. Skipping any step leaves drift the next agent will hit.

## 0. Scope check

Before starting, confirm:
- The new policy can be expressed in the DSL — `Filter` / `Select` / `With` / named-fns from `docs/dsl/schema.md` §5 + reducers from §6. If it requires session-affinity, history beyond `PrebleGCtx`, or any external I/O, it cannot be a DSL policy and this skill does not apply.
- The cargo-feature name ends in `-q`. Convention: `<descriptor>-q`, kebab-case. Struct name is the same in PascalCase ending in `Q` (e.g. `least-active-q` → `LeastActiveQ`).

## 1. Pick the module

Open `docs/dsl/policies.md` §1 (Module organisation by upstream baseline system). Decide which existing module the policy belongs to:

- vLLM-derived → `router/src/scheduler/policies/vllm.rs`
- bailian-derived → `router/src/scheduler/policies/bailian.rs`
- AIBrix-derived → `router/src/scheduler/policies/aibrix.rs`
- Dynamo-derived → `router/src/scheduler/policies/dynamo.rs`
- our system (lmetric) → `router/src/scheduler/policies/lmetric.rs`
- Preble-derived → `router/src/scheduler/policies/preble/` (a directory)
- llm-d-derived → `router/src/scheduler/policies/llm_d/<file>.rs` (one file per policy in the directory)
- no upstream-system origin AND a single `Select` or shallow `Filter` → `router/src/scheduler/policies/simple.rs`

If the policy comes from a NEW upstream baseline system (not in the table above), create a new module file/dir at `router/src/scheduler/policies/<system>.rs` and update `policies.md` §1's table in the same commit.

## 2. Add the cargo feature

Edit `router/Cargo.toml`. Add the feature line under the matching baseline-system block (see the comments around lines 108–148). Example for an llm-d-derived policy:

```toml
# llm_d/ — llm-d baselines (single-scorer ablations + multi-scorer combos)
...
new-policy-q = []
```

## 3. Write the `policy!` invocation

In the chosen module file (or a new file inside `llm_d/` / `preble/`), add:

```rust
policy! {
    name: NewPolicyQ,                    // PascalCase struct name
    gctx: (),                            // or a per-policy GCtx struct
    body: {
        // impl-form Rust per docs/dsl/implementation.md §2.1 rewrite table
    },
    // optional after_extra: for stateful policies
}
```

Constraints (enforced by the `policy!` proc-macro lint, `policy-dsl/src/check.rs`):
- Only call functions in the allowlist (`docs/dsl/implementation.md` §2.2). Combinators: `filter_then`, `select_{min,max,rand}_by`. Reducers: `{mean,std,sum,min,max}_of_usize`. Named pure fns: from `docs/dsl/schema.md` §5.
- No `for`/`while`/`loop`/`match`/`unsafe`/`return`/`break`/`continue`.
- Helper: `root_target(&observations)` to materialise the initial candidate set.

If the policy needs a new named-fn or reducer, that is a separate review (see CLAUDE.md "New / renamed / removed named-fn or reducer" checklist).

If a new file in `llm_d/` / `preble/`: also add `pub(crate) mod <file>;` + re-export to that directory's `mod.rs`.

## 4. File header doc comment

Add at the top of the policy's source file (or above the `policy!` invocation if sharing a module like `simple.rs`):

```rust
//! `new-policy-q` — one-line summary of what the policy computes.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2):
//!
//! ```text
//! [the §2 listing for this policy]
//! ```
//!
//! ## Origin (if from an upstream baseline)
//! Brief mapping rationale: which upstream scorer/algorithm this ports
//! and any divergences (tiebreak, signal source, etc.).
```

## 5. Wire up `mod.rs`

Edit `router/src/scheduler/policies/mod.rs`. **Three** places to update:

1. Module declaration (only if you added a new file/module, not when extending `simple.rs`):
   ```rust
   pub(crate) mod <module>;
   ```
2. Re-export (always):
   ```rust
   #[allow(unused_imports)]
   pub(crate) use <module>::NewPolicyQ;
   ```
   For `llm_d/` policies, add to the `pub(crate) use llm_d::{ ... }` group.
3. `TaskAssigner` cfg-arm:
   ```rust
   #[cfg(feature = "new-policy-q")]
   pub(crate) type TaskAssigner = PolicyRunner<NewPolicyQ>;
   ```
   AND add `feature = "new-policy-q",` to the catch-all default's `not(any(...))` exclusion list at the bottom of the file (so default mode doesn't accidentally co-select).

## 6. Add the §2 listing in `docs/dsl/policies.md`

Append a `policy <new-policy-q> (gctx: ...): ...` block to the §2 fenced code listing, following the spec-form syntax in `docs/dsl/schema.md` §2. Use exactly the same operator spelling (`Select min by`, `Filter (P) ...`, `With τ = ... in`) as sibling entries — the rewrite-table audit is purely syntactic.

If the policy comes from a new baseline-system module, also update §1's module-organisation table.

## 7. CLAUDE.md scheduling-policies list

Add a one-line bullet under the matching `**<module>.rs**` block in CLAUDE.md's "Scheduling Policies" section. Format:

```markdown
- `new-policy-q` — one-line summary. Reference upstream scorer name if applicable (`<file>.rs`).
```

If a new module: also add a `**<module>.rs**` heading section.

## 8. Verify

```bash
cargo check -p router --no-default-features \
  --features vllm-backend,radixtree-blockhash,default-hash-algo,determinent-schedule,new-policy-q
```

Then a default build (catch-all default still works):

```bash
cargo check -p router
```

Both must pass with no errors. Existing 6 warnings about unused simulator vars are pre-existing and unrelated.

## 9. Final coherence sweep

Before declaring done, grep for the new policy name across the repo:

```bash
rg new-policy-q --glob '!target/**'
```

Expected hits: `Cargo.toml` (1), `mod.rs` (TaskAssigner cfg + catch-all exclusion = 2), policy source file (1), file header doc (1), `docs/dsl/policies.md` (1), `CLAUDE.md` (1). Anything else is either a sibling-policy mention (fine) or a stale reference you missed (fix).

If the policy is camera-ready-relevant, also update the corresponding sweep config in [MetricsTestRunner](https://github.com/blitz-serving/MetricsTestRunner) (`sweeps/lmetric_camera_ready_*.toml`).
