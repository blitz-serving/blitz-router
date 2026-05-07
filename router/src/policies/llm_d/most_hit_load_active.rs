//! `most-hit-load-active-q` — precise prefix-cache + load-aware +
//! kv-cache-utilization.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2):
//!
//! ```text
//! With M_h = Max hit_blocks(req, ·), m_h = Min hit_blocks(req, ·),
//!      M_a = Max .all_tokens,        m_a = Min .all_tokens in
//! Select max by w_hit  · ((hit_blocks(req, sctx) − m_h) / (M_h − m_h))
//!             + w_load · (if sctx.waiting == 0 then 0.5
//!                         else 0.5 · (1 − min(sctx.waiting, T) / T))
//!             + w_kv   · (1 − (sctx.all_tokens − m_a) / (M_a − m_a))
//! after: default
//! ```
//!
//! Defaults: `w_hit = 10`, `w_load = 1`, `w_kv = 1`, `T = 128`. See
//! `metrics.rs::MOST_HIT_LOAD_ACTIVE_W_*` and `LOAD_AWARE_QUEUE_T`.
//!
//! ## llm-d origin
//!
//! Our port of llm-d's three-scorer combination:
//!
//!   - `precise-prefix-cache-scorer` (weight=10): per-pod KV-block hits
//!     min-max-normalized to `[0, 1]`.
//!   - `load-aware-scorer` (weight=1):
//!     `if waiting==0 then 0.5 else 0.5·(1 − min(waiting, T)/T)`,
//!     `T = 128`.
//!   - `kv-cache-utilization-scorer` (weight=1): `1 − KVUsagePercent`.
//!     We don't track per-engine KV capacity at the metric layer; we
//!     substitute `1 − norm(sctx.all_tokens)` which is cap-free yet
//!     monotonically equivalent for any fixed CAP.
//!
//! Picker: `max-score-picker` (argmax; ties: deterministic last-maximum,
//! see `most_hit.rs` header).
//!
//! Each scorer contributes in `[0, 1]` before weighting (n_hit by min-max
//! of `hit_blocks`, n_load ≤ 0.5 by load-aware formula, n_kv by
//! `1 − min-max(all_tokens)`), matching llm-d's `enforce_score_range`
//! guarantee. The `enforce_score_range` clamp is a no-op and elided.

use crate::metrics::{
    LOAD_AWARE_QUEUE_T, MOST_HIT_LOAD_ACTIVE_W_HIT, MOST_HIT_LOAD_ACTIVE_W_KV,
    MOST_HIT_LOAD_ACTIVE_W_LOAD,
};
use policy_dsl::policy;

policy! {
    name: MostHitLoadActiveQ,
    gctx: (),
    body: {
        let lo_hit = min_of_usize(&observations, |o| hit_blocks(req, o)) as f32;
        let hi_hit = max_of_usize(&observations, |o| hit_blocks(req, o)) as f32;
        let lo_act = min_of_usize(&observations, |o| o.all_tokens) as f32;
        let hi_act = max_of_usize(&observations, |o| o.all_tokens) as f32;
        let eps = 1e-6_f32;
        let dx_hit = (hi_hit - lo_hit).abs().max(eps);
        let dx_act = (hi_act - lo_act).abs().max(eps);
        select_max_by(&root_target(&observations), |o| {
            let n_hit = ((hit_blocks(req, o) as f32 - lo_hit) / dx_hit).clamp(0.0, 1.0);
            let waiting = o.waiting as f32;
            let n_load = if waiting == 0.0 {
                0.5
            } else {
                0.5 * (1.0 - waiting.min(LOAD_AWARE_QUEUE_T) / LOAD_AWARE_QUEUE_T)
            };
            let n_kv = ((hi_act - o.all_tokens as f32) / dx_act).clamp(0.0, 1.0);
            let s = MOST_HIT_LOAD_ACTIVE_W_HIT * n_hit
                  + MOST_HIT_LOAD_ACTIVE_W_LOAD * n_load
                  + MOST_HIT_LOAD_ACTIVE_W_KV * n_kv;
            (s * 1000.0) as i64
        })
    },
}
