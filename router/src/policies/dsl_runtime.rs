//! Runtime support for codegen-emitted policy bodies.
//!
//! The DSL macro emits `schedule()` functions that:
//!   1. Capture per-replica `Observation` snapshots under one lock each
//!      via `capture_observations`.
//!   2. Evaluate the DSL body purely against those snapshots.
//!   3. Apply the `after:` clause via `apply_default_after` plus any
//!      policy-specific gctx mutations.
//!
//! See `docs/dsl/schema.md` §4 (schema), §5 (named fns), §8 (after).

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::kvcache::BlockHash;
use crate::policies::Entry;
use crate::validation::ValidGenerateRequest;
use crate::{LMetricInc, ScheduleContext};

/// Per-replica snapshot captured under a single `sctx.lock().await`.
///
/// All DSL `<fn>` and `<pred>` bodies operate on `Observation` fields,
/// not on the live `ScheduleContext`. This guarantees the policy
/// evaluation is allocation-free and lock-free after the initial pass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Observation {
    pub idx: usize,
    pub bs: usize,
    pub waiting: usize,
    pub queued_tokens: isize,
    pub all_tokens: usize,
    pub block_size: usize,
    pub hit_blocks: usize,
    pub epoch: u64,
}

/// Lock each sctx briefly, capture all fields needed by any policy.
///
/// Same-lock pairing of `hit_blocks` with `epoch` is the soundness
/// condition for the epoch-skip optimization in `apply_default_after`.
pub(crate) async fn capture_observations(
    entry: &Entry,
    all_sctx: &[Arc<Mutex<ScheduleContext>>],
) -> Vec<Observation> {
    use futures::future::join_all;
    let block_size = entry.block_hash_state.get_block_size();
    let hashes = entry.block_hash_state.get_hashes();
    let captures = all_sctx.iter().enumerate().map(|(idx, arc)| async move {
        let sctx = arc.lock().await;
        let hit_blocks = sctx.block_hash.get(hashes);
        let epoch = sctx.block_hash.epoch();
        Observation {
            idx,
            bs: sctx.lmetric.bs,
            waiting: sctx.lmetric.waiting_reqs,
            queued_tokens: sctx.lmetric.prefill_tokens,
            all_tokens: sctx.lmetric.all_tokens,
            block_size,
            hit_blocks,
            epoch,
        }
    });
    join_all(captures).await
}

// =========================================================================
// Named pure-fn library (docs/dsl/schema.md §5)
// =========================================================================

/// New (uncached) prefill tokens contributed by `req` if routed to `sctx`.
#[inline]
pub(crate) fn new_tokens(req: &ValidGenerateRequest, sctx: &Observation) -> usize {
    req.input_tokens.len()
        .saturating_sub(sctx.hit_blocks * sctx.block_size)
}

/// New (uncached) blocks `req` would allocate at `sctx`. Equals
/// `⌈req.tokens / sctx.block_size⌉ − hit_blocks(req, sctx)`. Distinct from
/// `new_tokens / block_size` at partial-block boundaries (the trailing
/// partial token always rounds up to one extra block here).
#[inline]
pub(crate) fn new_blocks(req: &ValidGenerateRequest, sctx: &Observation) -> usize {
    if sctx.block_size == 0 {
        return 0;
    }
    req.input_tokens.len()
        .div_ceil(sctx.block_size)
        .saturating_sub(sctx.hit_blocks)
}

/// Already-queued prefill tokens at this replica, request-independent.
#[inline]
pub(crate) fn queued_tokens(sctx: &Observation) -> usize {
    sctx.queued_tokens.max(0) as usize
}

/// Total prefill tokens at this replica if `req` were routed here.
#[inline]
pub(crate) fn prefill_tokens(req: &ValidGenerateRequest, sctx: &Observation) -> usize {
    queued_tokens(sctx) + new_tokens(req, sctx)
}

/// Number of cached prefix blocks that `req` already has at this replica.
#[inline]
pub(crate) fn hit_blocks(_req: &ValidGenerateRequest, sctx: &Observation) -> usize {
    sctx.hit_blocks
}

/// Preble-flavored alias for `hit_blocks`.
#[inline]
pub(crate) fn match_blocks(req: &ValidGenerateRequest, sctx: &Observation) -> usize {
    hit_blocks(req, sctx)
}

/// Fraction of `req`'s input tokens already cached at this replica.
#[inline]
pub(crate) fn hit_pct(req: &ValidGenerateRequest, sctx: &Observation) -> f32 {
    if req.input_tokens.is_empty() {
        return 0.0;
    }
    (sctx.hit_blocks * sctx.block_size) as f32 / req.input_tokens.len() as f32
}

/// Approximation of decode-phase block load at this replica.
#[inline]
pub(crate) fn decode_blocks(sctx: &Observation) -> usize {
    if sctx.block_size == 0 {
        0
    } else {
        sctx.all_tokens / sctx.block_size
    }
}

/// Preble cost model query. Reads the `SlidingWindowHistogram` out of
/// `gctx` (the policy's `GlobalContext` — see `policies::preble::PrebleGCtx`)
/// and adds the per-replica score adjustment that PrebleQ overlays on
/// top of `(new_tokens + all_tokens)`.
///
/// Splitting out as a named pure fn lets the policy DSL stay a one-liner
/// while concentrating the `match_ratio > 0.5` bonus + the histogram
/// cost lookup in one auditable place.
pub(crate) fn preble_cost(
    req: &ValidGenerateRequest,
    sctx: &Observation,
    gctx: &crate::policies::preble::PrebleGCtx,
) -> i64 {
    let new_pre = new_tokens(req, sctx);
    let all = sctx.all_tokens;
    let mut score = (new_pre + all) as i64;
    let input_len = req.input_tokens.len();
    let match_tokens = sctx.hit_blocks * sctx.block_size;
    let match_ratio = match_tokens as f64 / input_len.max(1) as f64;
    if match_ratio > 0.5 {
        let bonus = (match_ratio * input_len as f64 * 0.5) as i64;
        score -= bonus;
    }
    if let Some(histogram) = gctx.histogram() {
        let costs = histogram.get_allocation_cost_per_replica();
        if sctx.idx < costs.len() {
            score += (costs[sctx.idx] * 1000.0) as i64;
        }
    }
    score
}

/// `after_extra` hook for PrebleQ: updates the sliding-window histogram
/// in `gctx` with the routing decision so future `preble_cost` calls
/// reflect it. Lazy-initializes the histogram on first call.
pub(crate) fn preble_update_after(
    entry: &Entry,
    chosen: &Observation,
    num_replicas: usize,
    gctx: &mut crate::policies::preble::PrebleGCtx,
) {
    crate::policies::preble::update_histogram_into(
        gctx,
        entry.block_hash_state.get_hashes(),
        chosen.hit_blocks,
        entry.request.input_tokens.len(),
        chosen.block_size,
        chosen.idx,
        num_replicas,
    );
}

// =========================================================================
// Combinators — Filter / Select min / Select max / Select rand
// =========================================================================

/// `Filter (P) { A } else { B }`: partition `target` by `pred`, evaluate
/// `on_pass` on the passing subset; if empty, evaluate `on_fail` on the
/// rejected subset (which equals `target`). Lossless: never returns `None`
/// just because the filter excluded everything.
pub(crate) fn filter_then<P, F, G>(
    target: &[&Observation],
    pred: P,
    on_pass: F,
    on_fail: G,
) -> Option<usize>
where
    P: Fn(&Observation) -> bool,
    F: FnOnce(&[&Observation]) -> Option<usize>,
    G: FnOnce(&[&Observation]) -> Option<usize>,
{
    let (passing, rejected): (Vec<&Observation>, Vec<&Observation>) =
        target.iter().copied().partition(|o| pred(*o));
    if !passing.is_empty() {
        on_pass(&passing)
    } else {
        on_fail(&rejected)
    }
}

/// `Select min by F`: pick the index with the smallest score in `target`.
pub(crate) fn select_min_by<F, S>(target: &[&Observation], score: F) -> Option<usize>
where
    F: Fn(&Observation) -> S,
    S: PartialOrd,
{
    target
        .iter()
        .min_by(|a, b| {
            score(a)
                .partial_cmp(&score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|o| o.idx)
}

/// `Select max by F`: pick the index with the largest score in `target`.
pub(crate) fn select_max_by<F, S>(target: &[&Observation], score: F) -> Option<usize>
where
    F: Fn(&Observation) -> S,
    S: PartialOrd,
{
    target
        .iter()
        .max_by(|a, b| {
            score(a)
                .partial_cmp(&score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|o| o.idx)
}

/// `Select rand by F`: weighted categorical sample over `target`. The
/// degenerate `Select rand by 1` shape (uniform random) is the case where
/// `score` returns a constant.
pub(crate) fn select_rand_by<F>(target: &[&Observation], score: F) -> Option<usize>
where
    F: Fn(&Observation) -> f32,
{
    if target.is_empty() {
        return None;
    }
    let weights: Vec<(usize, f32)> = target.iter().map(|o| (o.idx, score(o))).collect();
    Some(weighted_pick(&weights))
}

// =========================================================================
// Helper: convert &[Observation] (root candidate set) → &[&Observation]
// (the form combinators accept). Called once at the top of each schedule.
// =========================================================================

#[inline]
pub(crate) fn root_target(observations: &[Observation]) -> Vec<&Observation> {
    observations.iter().collect()
}

// =========================================================================
// Reducers — closed set of six (docs/dsl/schema.md §6)
// =========================================================================

#[inline]
pub(crate) fn mean_of_usize(obs: &[Observation], proj: impl Fn(&Observation) -> usize) -> f32 {
    if obs.is_empty() { return 0.0; }
    obs.iter().map(|o| proj(o) as f32).sum::<f32>() / obs.len() as f32
}

#[inline]
pub(crate) fn std_of_usize(obs: &[Observation], proj: impl Fn(&Observation) -> usize + Copy) -> f32 {
    if obs.len() <= 1 { return 0.0; }
    let m = mean_of_usize(obs, proj);
    let var: f32 = obs.iter()
        .map(|o| { let x = proj(o) as f32 - m; x * x })
        .sum::<f32>() / (obs.len() - 1) as f32;
    var.sqrt()
}

#[inline]
pub(crate) fn min_of_usize(obs: &[Observation], proj: impl Fn(&Observation) -> usize) -> usize {
    obs.iter().map(proj).min().unwrap_or(0)
}

#[inline]
pub(crate) fn max_of_usize(obs: &[Observation], proj: impl Fn(&Observation) -> usize) -> usize {
    obs.iter().map(proj).max().unwrap_or(0)
}

#[inline]
#[allow(dead_code)] // closed reducer vocabulary; kept for future policies
pub(crate) fn sum_of_usize(obs: &[Observation], proj: impl Fn(&Observation) -> usize) -> usize {
    obs.iter().map(proj).sum()
}

// f32 variants for policies whose reducers operate over f32 quantities (e.g., hit_pct, normalized scores in bailian, aibrix's stddev over request counts as f32).

#[inline]
#[allow(dead_code)] // closed reducer vocabulary; kept for future policies
pub(crate) fn mean_of_f32(obs: &[Observation], proj: impl Fn(&Observation) -> f32) -> f32 {
    if obs.is_empty() { return 0.0; }
    obs.iter().map(proj).sum::<f32>() / obs.len() as f32
}

#[inline]
#[allow(dead_code)] // closed reducer vocabulary; kept for future policies
pub(crate) fn std_of_f32(obs: &[Observation], proj: impl Fn(&Observation) -> f32 + Copy) -> f32 {
    if obs.len() <= 1 { return 0.0; }
    let m = mean_of_f32(obs, proj);
    let var: f32 = obs.iter()
        .map(|o| { let x = proj(o) - m; x * x })
        .sum::<f32>() / (obs.len() - 1) as f32;
    var.sqrt()
}

#[inline]
pub(crate) fn min_of_f32(obs: &[Observation], proj: impl Fn(&Observation) -> f32) -> f32 {
    obs.iter().map(proj).fold(f32::INFINITY, f32::min)
}

#[inline]
pub(crate) fn max_of_f32(obs: &[Observation], proj: impl Fn(&Observation) -> f32) -> f32 {
    obs.iter().map(proj).fold(f32::NEG_INFINITY, f32::max)
}

// =========================================================================
// Categorical (weighted) sampling for `Select rand by f`
// =========================================================================

/// Pick a replica index from `weights` proportional to weight.
/// Falls back to the first if all weights are non-positive.
pub(crate) fn weighted_pick(weights: &[(usize, f32)]) -> usize {
    debug_assert!(!weights.is_empty());
    let total: f32 = weights.iter().map(|(_, w)| w.max(0.0)).sum();
    if total <= 0.0 {
        return weights[0].0;
    }
    use rand::Rng;
    let mut r = rand::thread_rng().gen::<f32>() * total;
    for &(idx, w) in weights {
        let w = w.max(0.0);
        if r <= w {
            return idx;
        }
        r -= w;
    }
    weights.last().unwrap().0
}

// =========================================================================
// `after: default` — the canonical commit-driven mutation block
// =========================================================================

/// Apply the canonical post-decision mutations to `chosen`'s sctx.
///
/// Includes the epoch-skip optimization: if the radix tree hasn't
/// mutated since the scoring-time observation, reuse the cached
/// `hit_blocks`. Otherwise re-lookup under the apply-time lock.
pub(crate) async fn apply_default_after(
    entry: &Entry,
    all_sctx: &[Arc<Mutex<ScheduleContext>>],
    chosen: usize,
    cached: &Observation,
) {
    let request = &entry.request;
    let ScheduleContext { lmetric, block_hash } =
        &mut *all_sctx[chosen].lock().await;
    let current_epoch = block_hash.epoch();
    let hit_nblks = if cached.epoch == current_epoch {
        cached.hit_blocks
    } else {
        block_hash.get(entry.block_hash_state.get_hashes())
    };
    entry.block_hash_state.set_pred_block_hits(hit_nblks);
    entry.block_hash_state.set_decision_epoch(current_epoch);
    let new_ntkns = request.input_tokens.len()
        .saturating_sub(hit_nblks * entry.block_hash_state.get_block_size());

    tracing::info!(
        target: "scheduling",
        request_id = request.request_id,
        engine = chosen,
        predicted_hits = hit_nblks,
        radix_epoch = current_epoch,
        new_tokens = new_ntkns,
        "DECISION"
    );

    *lmetric += LMetricInc {
        bs_inc: 1,
        waiting_reqs_inc: 1,
        prefill_tokens_inc: new_ntkns,
        all_tokens_inc: request.input_tokens.len(),
    };
}
