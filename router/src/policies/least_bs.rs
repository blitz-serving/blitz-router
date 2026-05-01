//! `least-bs-q` — pick the replica with the smallest active batch size.
//!
//! Paper-form DSL (`docs/dsl-schema.md` §8):
//!
//! ```text
//! Select min by sctx.bs
//! ```
//!
//! ## Naming honesty
//!
//! `sctx.bs` (per §4.2) = `lmetric.bs` = **active batch size = running +
//! queued** (queued inside the engine's batch, distinct from
//! `sctx.waiting` which is the router-level waiting queue). Naming this
//! policy after `bs` rather than after "running" avoids implying it
//! measures only the running fraction — it is a composite signal.
//!
//! ## llm-d origin
//!
//! Closest single-scorer port of llm-d's `running-requests-scorer`, which
//! upstream normalizes `RunningRequestsSize` (pure running, no queued)
//! inversely. Our `sctx.bs = running + queued`, so this policy ranks by a
//! strictly larger composite quantity. For comparison-against-llm-d
//! purposes this is the closest mapping our metric layer admits without
//! adding a pure-running field to `Observation`. Flag this widening if
//! the camera-ready paper draws conclusions about pure RunningRequestsSize
//! signal versus queue-aware composite signal.
//!
//! Distinct from:
//!   - `join-shortest-q` (`4·waiting + bs`, additionally weights
//!     `sctx.waiting`)
//!   - `least-waiting-q` (`sctx.waiting` only — different "queue")

use policy_dsl::policy;

policy! {
    name: LeastBsQ,
    gctx: (),
    body: {
        select_min_by(&root_target(&observations), |o| o.bs)
    },
}
