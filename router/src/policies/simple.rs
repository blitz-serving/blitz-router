//! "Simple" policies — those whose DSL body fits in a single `Select` or
//! a shallow `Filter`, with no per-policy state beyond what fits in
//! `RRGCtx`. Per `docs/dsl-schema.md` §9, these live together for review
//! ergonomics rather than each having their own file.
//!
//! Policies in this file:
//!   - `RandomQ`           (`random-q`)
//!   - `RoundRobinQ`       (`round-robin-q`)  — uses `RRGCtx`
//!   - `JShortestQ`        (`join-shortest-q`)  — vLLM `4·waiting + bs`
//!   - `JShortestQWeight`  (`join-shortest-q-weight`) — alias of `JShortestQ`
//!   - `JLeastWaitTokenQ`  (`least-wait-token-q`)
//!   - `JBoundMostHitQ2`   (`bounded-most-hit-q`)  — includes the
//!     attention-black-hole fallback fix (§8 / §13.1)
//!
//! Each invocation maps 1:1 to a paper-form DSL listing in
//! `docs/dsl-schema.md` §8 via the rewrite table in §13.1.

use crate::metrics::WAITINGT_PREFILL_TOKEN_BOUND;
use policy_dsl::policy;

// =========================================================================
// Round-robin global context (cross-replica counter).
// =========================================================================

#[derive(Default, Debug)]
pub(crate) struct RRGCtx {
    pub(crate) next_replica_id: usize,
}

// =========================================================================
// random-q :=  Select rand by 1
// =========================================================================
policy! {
    name: RandomQ,
    gctx: (),
    body: { select_rand_by(&root_target(&observations), |_o| 1.0_f32) },
}

// =========================================================================
// round-robin-q :=  Select min by (if sctx.idx == gctx.next then 0 else 1)
//                   after default; gctx.next ← (gctx.next + 1) % Count
// =========================================================================
policy! {
    name: RoundRobinQ,
    gctx: RRGCtx,
    body: {
        let next = gctx.next_replica_id;
        select_min_by(&root_target(&observations), |o| if o.idx == next { 0_u32 } else { 1_u32 })
    },
    after_extra: {
        let count = observations.len().max(1);
        gctx.next_replica_id = (gctx.next_replica_id + 1) % count;
    }
}

// =========================================================================
// join-shortest-q  :=  Select min by 4·sctx.waiting + sctx.bs   (vLLM)
// join-shortest-q-weight :=  same formula  (kept as separate alias for the
//                            cargo feature flag — both compile to the same
//                            code, exactly one is enabled per build)
// =========================================================================
policy! {
    name: JShortestQ,
    gctx: (),
    body: { select_min_by(&root_target(&observations), |o| 4 * o.waiting + o.bs) },
}

policy! {
    name: JShortestQWeight,
    gctx: (),
    body: { select_min_by(&root_target(&observations), |o| 4 * o.waiting + o.bs) },
}

// =========================================================================
// least-wait-token-q  :=  Select min by prefill_tokens(req, sctx)
// =========================================================================
policy! {
    name: JLeastWaitTokenQ,
    gctx: (),
    body: { select_min_by(&root_target(&observations), |o| prefill_tokens(req, o)) },
}

// =========================================================================
// bounded-most-hit-q  :=  Filter (queued_tokens(sctx) < BOUND)
//                            { Select max by hit_blocks(req, sctx) }
//                          else
//                            { Select min by prefill_tokens(req, sctx) }
//
// Note: the else-branch routes to *least-wait-token* semantics, NOT to
// max-hits again. This is the "attention black hole" fix from §8 — under
// load, a fresh replica with hit_blocks=0 must still be reachable.
// =========================================================================
policy! {
    name: JBoundMostHitQ2,
    gctx: (),
    body: {
        filter_then(
            &root_target(&observations),
            |o| queued_tokens(o) < WAITINGT_PREFILL_TOKEN_BOUND,
            |t| select_max_by(t, |o| hit_blocks(req, o)),
            |t| select_min_by(t, |o| prefill_tokens(req, o)),
        )
    },
}
