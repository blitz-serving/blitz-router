//! `most-hit-load-q` — precise prefix-cache locality + load-aware tiebreak.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2):
//!
//! ```text
//! With M_h = Max hit_blocks(req, ·), m_h = Min hit_blocks(req, ·) in
//! Select max by w_hit · ((hit_blocks(req, sctx) − m_h) / (M_h − m_h))
//!             + w_load · (if sctx.waiting == 0 then 0.5
//!                         else 0.5 · (1 − min(sctx.waiting, T) / T))
//! after: default
//! ```
//!
//! Defaults: `w_hit = 10`, `w_load = 1`, `T = 128`. See
//! `metrics.rs::MOST_HIT_LOAD_W_HIT` / `MOST_HIT_LOAD_W_LOAD` /
//! `LOAD_AWARE_QUEUE_T`.
//!
//! ## llm-d origin
//!
//! Our port of llm-d's two-scorer combination (precise-prefix-cache +
//! load-aware), the most natural extension of the kvcache baseline:
//!
//!   - `precise-prefix-cache-scorer` (weight=10) — per-pod KV-block hits
//!     min-max-normalized to `[0, 1]`.
//!   - `load-aware-scorer` (weight=1) —
//!     `if waiting==0 then 0.5 else 0.5·(1 − min(waiting, T)/T)`,
//!     default `T=128`.
//!
//! Picker: `max-score-picker` (argmax; ties: llm-d shuffles before stable
//! sort, we use `Iterator::max_by` deterministic last-maximum — see
//! `most_hit.rs` header for the divergence note).
//!
//! Bijectively mirrors llm-d's `scheduler_profile.go::runScorerPlugins`
//! pipeline:
//!   for each scorer s:
//!     weighted_score[ep] += enforce_score_range(s) · w
//! Both `n_hit` (∈ `[0, 1]` by min-max) and `n_load` (∈ `[0, 0.5]` by
//! formula) are already in `[0, 1]` so the `enforce_score_range` clamp
//! is a no-op and elided.

use crate::metrics::{LOAD_AWARE_QUEUE_T, MOST_HIT_LOAD_W_HIT, MOST_HIT_LOAD_W_LOAD};
use policy_dsl::policy;

policy! {
    name: MostHitLoadQ,
    gctx: (),
    body: {
        let lo_hit = min_of_usize(&observations, |o| hit_blocks(req, o)) as f32;
        let hi_hit = max_of_usize(&observations, |o| hit_blocks(req, o)) as f32;
        let eps = 1e-6_f32;
        let dx_hit = (hi_hit - lo_hit).abs().max(eps);
        select_max_by(&root_target(&observations), |o| {
            let n_hit = ((hit_blocks(req, o) as f32 - lo_hit) / dx_hit).clamp(0.0, 1.0);
            let waiting = o.waiting as f32;
            let n_load = if waiting == 0.0 {
                0.5
            } else {
                0.5 * (1.0 - waiting.min(LOAD_AWARE_QUEUE_T) / LOAD_AWARE_QUEUE_T)
            };
            let s = MOST_HIT_LOAD_W_HIT * n_hit + MOST_HIT_LOAD_W_LOAD * n_load;
            (s * 1000.0) as i64
        })
    },
}
