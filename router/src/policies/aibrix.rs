//! `aibrix-q` — port of AIBrix's `prefix_cache.go` routing.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2): two-level lossless `Filter`
//! over an outer load-imbalance gate and an inner stddev threshold,
//! with tuple-keyed `Select min` (`(-hit_pct, bs)` to get max hit, ties
//! broken by smaller bs).
//!
//! Both branches of the outer Filter are syntactically identical because
//! the imbalance gate restricts the candidate set rather than changing
//! the inner algorithm — `filter_then`'s lossless fallback handles the
//! "balanced (no element passes lo predicate)" case by re-running the
//! inner Filter on all replicas.

use policy_dsl::policy;

const AIBRIX_IMBALANCE_ABS_COUNT: usize = 8;
const AIBRIX_STDDEV_FACTOR: f32 = 1.0;

policy! {
    name: AibrixQ,
    gctx: (),
    body: {
        let lo = min_of_usize(&observations, |o| o.bs);
        let hi = max_of_usize(&observations, |o| o.bs);
        let gap = hi.saturating_sub(lo);
        let mean_bs = mean_of_usize(&observations, |o| o.bs);
        let std_bs  = std_of_usize(&observations, |o| o.bs);
        let tau = mean_bs + AIBRIX_STDDEV_FACTOR * std_bs;
        filter_then(
            &root_target(&observations),
            |o| o.bs == lo && gap > AIBRIX_IMBALANCE_ABS_COUNT,
            |t| filter_then(
                t,
                |o| (o.bs as f32) <= tau,
                |tt| select_min_by(tt, |o| (-hit_pct(req, o), o.bs)),
                |tt| select_max_by(tt, |o| (-hit_pct(req, o), o.bs)),
            ),
            |t| filter_then(
                t,
                |o| (o.bs as f32) <= tau,
                |tt| select_min_by(tt, |o| (-hit_pct(req, o), o.bs)),
                |tt| select_max_by(tt, |o| (-hit_pct(req, o), o.bs)),
            ),
        )
    },
}
