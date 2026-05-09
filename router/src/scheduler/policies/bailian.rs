//! `bailian-impl-q` — Bailian's three-component normalize-and-sample policy.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2):
//!
//! ```text
//! With M_bs  = Max .bs,
//!      M_tok = Max .all_tokens in
//! Select rand by α · hit_pct(req, sctx)
//!               + β · (1 - sctx.bs / M_bs)
//!               + γ · (1 - sctx.tokens / M_tok)
//! ```
//!
//! The actual normalization in `select_rand_by`'s closure is a
//! per-component `(value - lo) / (hi - lo)` clamp, matching the existing
//! sampler (which used `inf_w`/`sup_w` from `NaiiveLattice` join/meet).
//! Reducers compute `lo` and `hi` once per call via `min_of_*` /
//! `max_of_*` and the With-bound names are read inside the closure.

use crate::scheduler::state::{BAILIAN_ALPHA, BAILIAN_BETA, BAILIAN_GAMMA};
use policy_dsl::policy;

policy! {
    name: BailianImplQ,
    gctx: (),
    body: {
        let lo_hit = min_of_f32(&observations, |o| hit_pct(req, o));
        let hi_hit = max_of_f32(&observations, |o| hit_pct(req, o));
        let lo_bs  = min_of_usize(&observations, |o| o.bs) as f32;
        let hi_bs  = max_of_usize(&observations, |o| o.bs) as f32;
        let lo_tok = min_of_usize(&observations, |o| o.all_tokens) as f32;
        let hi_tok = max_of_usize(&observations, |o| o.all_tokens) as f32;
        let eps = 1e-6_f32;
        let dx_hit = (hi_hit - lo_hit).abs().max(eps);
        let dx_bs  = (hi_bs - lo_bs).abs().max(eps);
        let dx_tok = (hi_tok - lo_tok).abs().max(eps);
        select_rand_by(&root_target(&observations), |o| {
            let n0 = ((hit_pct(req, o) - lo_hit) / dx_hit).clamp(0.0, 1.0);
            let n1 = ((hi_bs - o.bs as f32) / dx_bs).clamp(0.0, 1.0);
            let n2 = ((hi_tok - o.all_tokens as f32) / dx_tok).clamp(0.0, 1.0);
            let p = n0 * BAILIAN_ALPHA + n1 * BAILIAN_BETA + n2 * BAILIAN_GAMMA;
            if p.is_finite() && p > 0.0 { p } else { 0.0 }
        })
    },
}
