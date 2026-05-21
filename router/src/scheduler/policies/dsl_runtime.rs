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

use crate::scheduler::kvcache::BlockHash;
use super::Entry;
use crate::gateway::validation::ValidGenerateRequest;
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
    /// Preble: per-replica `pod_load` snapshot (count of recent
    /// requests routed to this replica in the 3-min window). Captured
    /// from `sctx.block_hash.load(now)` only under `--features
    /// preble-q`; zero under any other feature flag.
    #[cfg(feature = "preble-q")]
    pub preble_load: usize,
    /// Preble: per-replica `pod_cost` snapshot (sum of
    /// `(PT_r + DT_r)` over the 3-min window). Captured from
    /// `sctx.block_hash.cost(now)` only under `--features preble-q`.
    #[cfg(feature = "preble-q")]
    pub preble_cost: f64,
    /// Preble-BS: sum of batch-size samples in the per-replica 3-min
    /// window. Lower = less loaded. Captured from
    /// `sctx.block_hash.bs_sum(now)` only under `--features
    /// preble-bs-q`.
    #[cfg(feature = "preble-bs-q")]
    pub preble_bs_sum: f64,
    /// Preble-TPS: count of forward steps in the per-replica 3-min
    /// window. Higher = more throughput, less loaded. Captured from
    /// `sctx.block_hash.tps_count(now)` only under `--features
    /// preble-tps-q`.
    #[cfg(feature = "preble-tps-q")]
    pub preble_tps_count: usize,
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
        // Need `&mut` under preble-q because `block_hash.load(now)` /
        // `.cost(now)` lazily expire the sliding window. Under any
        // other feature this is read-only and the `mut` binding is a
        // harmless no-op.
        #[allow(unused_mut)]
        let mut sctx = arc.lock().await;
        let hit_blocks = sctx.block_hash.get(hashes);
        let epoch = sctx.block_hash.epoch();
        #[cfg(any(feature = "preble-q", feature = "preble-bs-q", feature = "preble-tps-q"))]
        let now = std::time::Instant::now();
        Observation {
            idx,
            bs: sctx.lmetric.bs,
            waiting: sctx.lmetric.waiting_reqs,
            queued_tokens: sctx.lmetric.prefill_tokens,
            all_tokens: sctx.lmetric.all_tokens,
            block_size,
            hit_blocks,
            epoch,
            #[cfg(feature = "preble-q")]
            preble_load: sctx.block_hash.load(now),
            #[cfg(feature = "preble-q")]
            preble_cost: sctx.block_hash.cost(now),
            #[cfg(feature = "preble-bs-q")]
            preble_bs_sum: sctx.block_hash.bs_sum(now),
            #[cfg(feature = "preble-tps-q")]
            preble_tps_count: sctx.block_hash.tps_count(now),
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

/// Preble load-balancing-branch cost: per-replica `pod_cost` from
/// the 3-min sliding window snapshotted into `Observation` at
/// `capture_observations` time. Scaled by 1000 to integer for
/// `select_min_by` stability; matches the previous helper's units.
///
/// The Preble paper's `L_i = Σ_{r ∈ W_i}(PT_r + DT_r)` is exactly this
/// — per-request prefill+decode contributions over the window,
/// attributed to the replica that served each. Reading is O(1).
///
/// **Removed in the per-replica refactor:** the previous overlay
/// `(new_tokens + all_tokens) - bonus` is gone. With per-replica trees
/// and per-replica scalars, the bare `pod_cost` is the paper-faithful
/// metric; the overlay was a workaround for the Go reference's
/// over-counting LOAD that this refactor eliminated structurally.
#[cfg(feature = "preble-q")]
pub(crate) fn preble_cost(sctx: &Observation) -> i64 {
    (sctx.preble_cost * 1000.0) as i64
}

/// Per-replica `pod_load` for Preble's KV$-aware-branch tie-break
/// (longest match → lowest load). Reads the snapshot captured at
/// `capture_observations` time. O(1).
///
/// Equivalent to `|W_P|` — the count of recent requests routed to this
/// replica in the 3-min window. Eliminates the LOAD over-counting
/// (Property L2) that the global-tree-with-owners design suffered from.
#[cfg(feature = "preble-q")]
pub(crate) fn preble_load(sctx: &Observation) -> usize {
    sctx.preble_load
}

/// Longest prefix-match length (in blocks) across ALL replicas — used
/// only for the KV$-aware-vs-load-balancing branch-split threshold.
/// Each replica's tree is engine-driven (faithful to actually-cached
/// state), so the max over replicas is the cluster-wide deepest
/// prefix. Bijective with Go's global `len(matchedTokens) /
/// len(tokens)` gating decision.
#[cfg(any(feature = "preble-q", feature = "preble-bs-q", feature = "preble-tps-q"))]
pub(crate) fn preble_global_match_blocks(
    observations: &[Observation],
    _prefix: &[u64],
) -> usize {
    // `observations[i].hit_blocks` was populated by
    // `sctx.block_hash.get(prefix)` for each replica's tree. The max
    // is the deepest prefix any replica currently caches.
    observations.iter().map(|o| o.hit_blocks).max().unwrap_or(0)
}

/// Per-replica longest-match depth (in blocks) — i.e. how deep this
/// replica's engine-driven tree caches the request's prefix. With
/// per-replica trees, "owned" = "in this replica's tree", so this is
/// just `o.hit_blocks`.
#[cfg(any(feature = "preble-q", feature = "preble-bs-q", feature = "preble-tps-q"))]
pub(crate) fn preble_owned_match_blocks(sctx: &Observation) -> usize {
    sctx.hit_blocks
}

/// Preble-BS load-balancing branch: sum of batch-size samples in
/// the per-replica 3-min window. Engine-step-driven (one push per
/// forward step from the colocation loop). Higher = more loaded;
/// the policy minimizes.
#[cfg(feature = "preble-bs-q")]
pub(crate) fn preble_bs_sum(sctx: &Observation) -> f64 {
    sctx.preble_bs_sum
}

/// Preble-TPS load-balancing branch: count of forward steps in the
/// per-replica 3-min window. Higher = more throughput, less loaded;
/// the policy maximizes.
#[cfg(feature = "preble-tps-q")]
pub(crate) fn preble_tps_count(sctx: &Observation) -> usize {
    sctx.preble_tps_count
}

/// `after_extra` hook for PrebleQ: pushes a per-request cost
/// contribution into the chosen replica's sliding window. Tree state
/// is independently maintained by the colocation event loop's
/// `BlockHash::insert` (engine-driven), so this helper does NOT touch
/// the tree; it only updates Preble's per-replica aggregates.
#[cfg(feature = "preble-q")]
pub(crate) async fn preble_update_after(
    entry: &Entry,
    chosen: &Observation,
    all_sctx: &[Arc<Mutex<ScheduleContext>>],
) {
    let mut sctx = all_sctx[chosen.idx].lock().await;
    let prefix = entry.block_hash_state.get_hashes();
    let block_size = chosen.block_size;
    let context_length = entry.request.input_tokens.len();
    let cached_tokens = chosen.hit_blocks.saturating_mul(block_size).min(context_length);
    let num_tokens = context_length.saturating_sub(cached_tokens);
    let decoding_length = sctx.block_hash.default_decoding_length();
    sctx.block_hash.update_with_cost(
        prefix,
        num_tokens,
        context_length,
        decoding_length,
        std::time::Instant::now(),
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
