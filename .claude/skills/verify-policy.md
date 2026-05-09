---
name: verify-policy
description: Verify a policy! invocation matches its docs/dsl/policies.md §2 spec-form listing via the rewrite table. Use for code review, audit, or when the user suspects impl-vs-spec drift.
---

# Verifying a Policy Implementation

Mechanically check that one `policy!` invocation in `router/src/scheduler/policies/...` is the impl-form image of its spec-form listing in `docs/dsl/policies.md` §2 under the rewrite table in `docs/dsl/implementation.md` §2.1.

The procedure is purely syntactic — no semantic equivalence reasoning needed. A reviewer applies the rewrite table right-to-left and compares.

## 0. Scope and limits

This skill verifies that **what the source claims** (the impl-form `policy!` body) matches **what the spec claims** (the §2 listing for the same policy name). It does NOT verify either against the upstream paper / system — that is a separate, citation-grounded audit (see `.claude/memory/dynamo_pd_colocated_terms.md` and issue #11 for why sibling-doc agreement is not a soundness check).

If the user suspects the SPEC is wrong (vs. upstream), this skill cannot help — escalate to the upstream-source review pattern in CLAUDE.md.

## 1. Inputs

Identify the policy name (e.g. `lmetric-q`, `dynamo-po-q`). From it derive:

- **Source location**: search `router/src/scheduler/policies/` for `name: <PascalCaseQ>` in a `policy!` block. Use:
  ```bash
  rg -l 'name: <PascalCaseQ>' router/src/scheduler/policies/
  ```
- **Spec-form location**: `docs/dsl/policies.md` §2 — find the `policy <name>-q (gctx: ...):` block.

If either is missing, the policy is not implemented OR not specified — that is the drift; report and stop.

## 2. Read both forms

Read the entire `policy!` invocation (including `gctx:`, `body:`, optional `after_extra:`) AND the entire §2 block. Hold both side-by-side.

## 3. Apply the rewrite table

Reference: `docs/dsl/implementation.md` §2.1.

Walk the impl-form body top-down. For every construct, find its row in the table and write down the spec-form equivalent. Reject the body immediately if it contains a construct not in the table — the lint is supposed to catch this at compile time, but a recently-added construct or a hand-edited macro could slip through.

| Pattern in impl-form Rust | Spec-form to write |
|---|---|
| `select_min_by(target, \|o\| F)` | `Select min by F` |
| `select_max_by(target, \|o\| F)` | `Select max by F` |
| `select_rand_by(target, \|o\| F)` | `Select rand by F` |
| `filter_then(target, \|o\| P, \|t\| { A }, \|t\| { B })` | `Filter (P) (A) (B)` |
| `{ let τ = E; body }` | `With τ = E in body` |
| `min_of_usize(&observations, \|o\| o.f)` | `Min .f` |
| `min_of_usize(&observations, \|o\| named_fn(req, o))` | `Min named_fn(req, ·)` |
| (similarly `max_/mean_/std_/sum_of_usize`) | (correspondingly) |
| `observations.len()` | `Count` |
| (no `after_extra`) | `after default` |
| `after_extra: { gctx.X = E; }` | `after default; gctx.X ← E` |

`gctx.<field>` and `req.<field>` reads in impl-form become `gctx.<field>` / `req.<field>` in spec-form. `o.<field>` in impl-form becomes `sctx.<field>` in spec-form (the closure variable `o` is the iterating per-replica observation).

## 4. Compare

Whitespace, operator spelling (`·` vs `*`, `≤` vs `<=`), and field-vs-named-fn shadow (`sctx.queued_tokens` field vs `queued_tokens(sctx)` fn — see `docs/dsl/schema.md` §4.2 Note) all matter. The §2 spec is the source of truth for the syntactic surface; the impl is checked against it.

Output one of:
- ✅ Match — report `policy <name>-q OK` and stop.
- ❌ Drift — report:
  1. The exact impl-form expression that mismatched.
  2. What the rewrite table maps it to.
  3. What the §2 listing actually says.
  4. The minimal patch (impl side OR spec side) — usually the IMPL is the source of truth for runtime behaviour, so prefer fixing the spec UNLESS the diff exposes a real bug in the impl.

## 5. Doc-comment consistency (optional but recommended)

The file header `//! Spec-form DSL (...)` block usually quotes the §2 listing inline. If it does, check that the inline quote matches the §2 listing too. A drift here means the header was hand-edited without re-pasting from §2.

## 6. Cross-policy notes

Some policies have known caveats — apply the right reading lens:

- `bounded-most-hit-q` (`simple.rs`): the else-branch routes to `Select min by prefill_tokens`, NOT to a second `Select max by hit_blocks`. This is the "attention black hole" guard noted in `policies.md` §2.
- `dynamo-q` vs `dynamo-po-q`: the per-request prefill term differs by `new_tokens` vs `prefill_tokens` — a swap is the historical drift from issue #11. ALWAYS cross-check against `selector.rs:150` in upstream Dynamo if both formulas look plausible.
- `most-hit-load-q` / `most-hit-load-active-q`: tunable weights live in `router/src/scheduler/state.rs` (`MOST_HIT_LOAD_W_*`); spec-form uses placeholder names `w_hit`, `w_load`, etc. Don't flag this as drift.
