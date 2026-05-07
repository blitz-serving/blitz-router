# Scheduling Policies — Canonical DSL Listings

This document is the catalog of every scheduling policy in the router. Each policy has a one-block spec-form DSL listing in §2 and a corresponding `policy! { ... }` invocation in `router/src/policies/...`. The 1:1 correspondence is enforced by the macro lint described in `implementation.md` §2.2.

For the schema (fields, named functions, reducers) every listing references, see `schema.md`.

## 1. Module organisation by upstream baseline system

Trivial policies — those with no upstream-system origin AND whose DSL expression fits in a single `Select` or simple `Filter` — live together in `router/src/policies/simple.rs` instead of one file each.

`simple.rs` residents:

- `random-q`
- `round-robin-q`
- `least-wait-token-q`
- `bounded-most-hit-q`

Policies derived from an upstream baseline system live in a per-system module (one file or one directory). The `<name>-q` cargo feature still routes 1:1 to a single `policy!` invocation; the partition is for review ergonomics (a reviewer can hold "everything llm-d-derived" or "everything Dynamo-derived" in working memory at once).

| Module | Cargo features |
|---|---|
| `vllm.rs` | `join-shortest-weight-q` |
| `bailian.rs` | `bailian-impl-q` |
| `aibrix.rs` | `aibrix-q` |
| `dynamo.rs` | `dynamo-q`, `dynamo-po-q` |
| `lmetric.rs` | `lmetric-q` |
| `preble/` | `preble-q` |
| `llm_d/` | `most-hit-q`, `least-waiting-q`, `least-bs-q`, `least-active-q`, `least-token-load-q`, `most-hit-load-q`, `most-hit-load-active-q` |

llm-d policies that cannot be expressed in the DSL (e.g. session-aware) are NOT ported; reference: `workspace/llm-d-scheduler/`.

The codegen treats single-file modules and directory modules identically; the partition is for human ergonomics only (review burden, drift visibility).

## 2. Canonical DSL listings

```
policy random-q (gctx: ()):
    Select rand by 1
    after: default

policy round-robin-q (gctx: RRGCtx):
    Select min by (if sctx.idx == gctx.next_replica_id then 0 else 1)
    after: default; gctx.next_replica_id <- (gctx.next_replica_id + 1) % Count

policy join-shortest-weight-q (gctx: ()):
    Select min by 4 · sctx.waiting + sctx.bs
    after: default

policy least-wait-token-q (gctx: ()):
    Select min by prefill_tokens(req, sctx)
    after: default

policy bounded-most-hit-q (gctx: ()):
    Filter (queued_tokens(sctx) < BOUND)
      (Select max by hit_blocks(req, sctx))
      (Select min by prefill_tokens(req, sctx))
    after: default

policy dynamo-q (gctx: ()):                              # Dynamo Decode-node logit
    Select min by w · (new_tokens(req, sctx) / sctx.block_size)
                  + new_blocks(req, sctx) + decode_blocks(sctx)
    after: default

policy dynamo-po-q (gctx: ()):                           # Dynamo Prefill-node logit
                                                         # ("po" = prefill-only node;
                                                         # was dynamo-decoupled-q)
    Select min by w · (prefill_tokens(req, sctx) / sctx.block_size)
                  + floor(prefill_tokens(req, sctx) / sctx.block_size)
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

policy most-hit-q (gctx: ()):                            # llm-d kvcache baseline
    Select max by hit_blocks(req, sctx)                  # see most_hit.rs header
    after: default

policy least-waiting-q (gctx: ()):                       # llm-d load-aware-scorer single
    Select min by sctx.waiting                           # = queue-depth-scorer single
    after: default

policy least-bs-q (gctx: ()):                            # llm-d running-requests-scorer
    Select min by sctx.bs                                # single (composite: bs = running + queued)
    after: default

policy least-active-q (gctx: ()):                        # llm-d kv-cache-utilization-scorer
    Select min by sctx.all_tokens                        # single (cap-free; argmin invariant)
    after: default

policy least-token-load-q (gctx: ()):                    # llm-d token-load-scorer single
    Select min by queued_tokens(sctx) + sctx.all_tokens
    after: default

policy most-hit-load-q (gctx: ()):                       # llm-d precise + load-aware
    With M_h = Max hit_blocks(req, ·),
         m_h = Min hit_blocks(req, ·) in
    Select max by w_hit  · ((hit_blocks(req, sctx) − m_h) / (M_h − m_h))
                + w_load · (if sctx.waiting == 0 then 0.5
                            else 0.5 · (1 − min(sctx.waiting, T) / T))
    after: default
    # defaults: w_hit=10, w_load=1, T=128 (see metrics.rs)

policy most-hit-load-active-q (gctx: ()):                # llm-d precise + load-aware + kv-util
    With M_h = Max hit_blocks(req, ·), m_h = Min hit_blocks(req, ·),
         M_a = Max .all_tokens,        m_a = Min .all_tokens in
    Select max by w_hit  · ((hit_blocks(req, sctx) − m_h) / (M_h − m_h))
                + w_load · (if sctx.waiting == 0 then 0.5
                            else 0.5 · (1 − min(sctx.waiting, T) / T))
                + w_kv   · (1 − (sctx.all_tokens − m_a) / (M_a − m_a))
    after: default
    # defaults: w_hit=10, w_load=1, w_kv=1, T=128 (see metrics.rs)
```

`chosen` in the `after:` clause is the `usize` index of the selected replica (see `schema.md` §10.2). It is a reserved name; codegen binds it after the `<expr>` evaluates.

Note: the fallback branch in `bounded-most-hit-q` routes to least-wait-token semantics (the "attention black hole" guard — under load, a fresh replica with `hit_blocks = 0` must remain reachable), not to identical max-hits scoring.
