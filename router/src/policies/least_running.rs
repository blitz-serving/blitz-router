//! `least-running-q` — pick the replica with the fewest in-flight requests.
//!
//! Paper-form DSL (`docs/dsl-schema.md` §8):
//!
//! ```text
//! Select min by sctx.bs
//! ```
//!
//! ## llm-d origin
//!
//! Our port of llm-d's `running-requests-scorer` single-scorer ablation.
//! Upstream normalizes `RunningRequestsSize` inversely (`1 − norm(bs)`);
//! argmax of that is argmin `sctx.bs`. `RunningRequestsSize` maps to our
//! `sctx.bs` (active batch size = running + queued).
//!
//! Native name: "least-running" reflects the signal — fewest currently
//! running. Distinct from:
//!   - `join-shortest-q` (`4·waiting + bs`, weights waiting too)
//!   - `least-waiting-q` (`waiting` only, ignores bs)
//!
//! This one is purely `bs`.

use policy_dsl::policy;

policy! {
    name: LeastRunningQ,
    gctx: (),
    body: {
        select_min_by(&root_target(&observations), |o| o.bs)
    },
}
