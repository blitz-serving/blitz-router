//! `least-waiting-q` — pick the replica with the fewest waiting requests.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2):
//!
//! ```text
//! Select min by sctx.waiting
//! ```
//!
//! ## llm-d origin
//!
//! Our port of two llm-d single-scorer ablations that collapse to the same
//! argmin in our environment:
//!
//!   - `load-aware-scorer` (single): score `if waiting==0 then 0.5 else
//!     0.5·(1 − min(waiting, T)/T)` for default T=128. argmax of this is
//!     argmin `sctx.waiting` (the cap is monotonic, doesn't change ranking).
//!   - `queue-depth-scorer` (single): score `1 − norm(waiting)`. argmax of
//!     this is also argmin `sctx.waiting`.
//!
//! Both upstream scorers measure the same signal (`WaitingQueueSize`); as
//! single-scorer policies they are the same algorithm. We name it after
//! what it COMPUTES (`least-waiting`) rather than after llm-d's plugin
//! labels ("load-aware" tells the reader nothing about which signal).
//!
//! Distinct from `join-shortest-weight-q` in `vllm.rs` which uses
//! `4·waiting + bs` (combines waiting AND batch size). This one uses
//! `waiting` only.

use policy_dsl::policy;

policy! {
    name: LeastWaitingQ,
    gctx: (),
    body: {
        select_min_by(&root_target(&observations), |o| o.waiting)
    },
}
