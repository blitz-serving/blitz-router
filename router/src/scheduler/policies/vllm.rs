//! Scheduling policies derived from the vLLM baseline.
//!
//! Currently one policy: `join-shortest-weight-q`. The "weight" qualifier
//! names the formula's defining feature — `sctx.waiting` carries weight 4
//! in the score versus weight 1 on `sctx.bs`, so the policy weights the
//! waiting queue more than the active batch. Without "weight" the name
//! would be ambiguous with other "shortest"-style policies (`least-bs-q`,
//! `least-waiting-q`) that drop one term entirely.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2):
//!
//! ```text
//! Select min by 4 · sctx.waiting + sctx.bs
//! ```
//!
//! ## vLLM origin
//!
//! Mirrors vLLM's stock dispatch tendency to pick the engine with the
//! lowest combined queue+batch pressure, with the `4×` weight on waiting
//! reflecting that a request stuck in the waiting queue costs more than
//! an extra slot in the active batch (the latter only adds parallelism
//! pressure; the former adds head-of-line latency).

use policy_dsl::policy;

policy! {
    name: JShortestWeightQ,
    gctx: (),
    body: { select_min_by(&root_target(&observations), |o| 4 * o.waiting + o.bs) },
}
