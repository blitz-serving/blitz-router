//! `lmetric-q` — the policy named after the lmetric paper.
//!
//! Paper-form DSL (`docs/dsl-schema.md` §8):
//!
//! ```text
//! Select min by prefill_tokens(req, sctx) · (sctx.bs + 1)
//! ```
//!
//! The `(bs + 1)` term ensures idle replicas (`bs == 0`) are still
//! distinguishable by their prefill load alone — without `+1` an idle
//! replica would always tie at score 0 regardless of queued prefill.

use policy_dsl::policy;

policy! {
    name: LmetricQ,
    gctx: (),
    body: { select_min_by(&root_target(&observations), |o| prefill_tokens(req, o) * (o.bs + 1)) },
}
