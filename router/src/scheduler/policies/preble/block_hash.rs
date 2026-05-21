//! `PrebleBlockHash` — Preble's `BlockHash` flavour.
//!
//! Composition of three pieces:
//!
//! 1. `inner: RadixTreeBlockHash` — the existing engine-driven KV-cache
//!    prefix matcher. The colocation event loop's
//!    `sctx.block_hash.{insert,remove}` calls keep this faithful to
//!    what's actually cached on the replica. Preble does not maintain
//!    a parallel stale-state guess (Go's 5-min LRU is unnecessary
//!    when engine SSE feedback is available).
//!
//! 2. `window: SlidingWindow<f64, Sum>` — per-request cost
//!    contributions over a 3-min sliding window. One window covers
//!    both `pod_load` (= `window.len()`, free from the deque) and
//!    `pod_cost` (= incrementally-maintained `Sum`).
//!
//! 3. `cost_model: TargetGpu` — hardcoded prefill polynomial
//!    coefficients (currently A800 by default). Same tables as Go;
//!    see [`super::cost_model`].
//!
//! Under `--features preble-q`, this type is aliased as
//! `PrefixBlockHash` (see `router::scheduler::kvcache`), so every
//! replica's `ScheduleContext.block_hash` is a `PrebleBlockHash`.
//! `BlockHash` trait methods delegate to `inner` (so the colocation
//! event loop is unchanged); inherent methods
//! [`PrebleBlockHash::update_with_cost`], [`PrebleBlockHash::load`],
//! [`PrebleBlockHash::cost`] expose Preble-specific aggregates and
//! resolve only because the alias points to the concrete type.
//!
//! ### Paper alignment
//!
//! - `pod_load(P) = window.len() = |W_P|` matches Preble paper §3.2's
//!   per-GPU formulation (no over-counting).
//! - `pod_cost(P) = Σ_{r ∈ W_P}(PT_r + DT_r)` matches the paper's
//!   `L_i = Σ_{r∈W}(PT_r + DT_r)` (per-request attribution to the
//!   routed replica, no `|owners|` division).
//! - Tree state follows engine `evicted_block_ids` SSE updates, which
//!   is closer to ground truth than the Go reference's 5-min stale
//!   guess.

use std::time::{Duration, Instant};

use radixtree::{BlockHash, RadixTreeBlockHash, SlidingWindow, Sum};
#[cfg(feature = "preble-tps-q")]
use radixtree::Count;

#[cfg(feature = "preble-q")]
use super::cost_model::{self, TargetGpu};

/// 3 min, matching Go's `slidingWindowPeriod`.
const WINDOW_DURATION: Duration = Duration::from_secs(3 * 60);

/// Hardcoded per-token decode time (seconds), matching Go's constant
/// at `prefix_cache_preble.go:345`. The `avgTimePerTokenPerPod` Go map
/// is dead code in production (only tests assign), so we do not
/// reproduce it.
#[cfg(feature = "preble-q")]
const TIME_PER_TOKEN_S: f64 = 0.15;

/// Default expected output length when the actual decoding length is
/// unknown at insert time. Go: `decodingLength = 45`
/// (`prefix_cache_preble.go`'s env default).
#[cfg(feature = "preble-q")]
const DEFAULT_DECODING_LENGTH: usize = 45;

/// Preble-flavoured `BlockHash`. See module docs.
#[cfg(feature = "preble-q")]
pub struct PrebleBlockHash {
    inner: RadixTreeBlockHash,
    window: SlidingWindow<f64, Sum>,
    cost_model: TargetGpu,
}

#[cfg(feature = "preble-q")]
unsafe impl Send for PrebleBlockHash {}
#[cfg(feature = "preble-q")]
unsafe impl Sync for PrebleBlockHash {}

#[cfg(feature = "preble-q")]
impl BlockHash for PrebleBlockHash {
    fn new(num_blocks: usize) -> Self {
        Self {
            inner: RadixTreeBlockHash::new(num_blocks),
            window: SlidingWindow::new(WINDOW_DURATION, Sum::default()),
            cost_model: TargetGpu::default(),
        }
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn insert(&mut self, block_hashes: &[u64], block_indices: Vec<u64>) -> usize {
        self.inner.insert(block_hashes, block_indices)
    }

    fn get(&self, block_hashes: &[u64]) -> usize {
        self.inner.get(block_hashes)
    }

    fn remove(&mut self, block_indices: Vec<u64>) {
        self.inner.remove(block_indices);
    }

    fn epoch(&self) -> u64 {
        self.inner.epoch()
    }
}

#[cfg(feature = "preble-q")]
impl PrebleBlockHash {
    /// Push a per-request cost contribution to the sliding window.
    ///
    /// Called from `PrebleQ`'s `after_extra`. Tree state is already
    /// up-to-date via the colocation event loop's `insert` (Preble does
    /// not separately materialize the path).
    ///
    /// `num_tokens` is the count of NEW (uncached) tokens — i.e. the
    /// actual prefill work this request incurs. `context_length` is
    /// the FULL input token count. `decoding_length` is the expected
    /// output length (caller passes
    /// [`PrebleBlockHash::default_decoding_length`] when unknown).
    ///
    /// `prefill_time(num_tokens, context_length)` already prices the
    /// missed-token prefill cost (linear MLP scales with `num_tokens`,
    /// attention with `context_length`), so the contribution is
    /// `prefill_t + decode_t` with no separate `miss_rate ×` factor.
    /// The `num_tokens == 0` gate preserves the "full cache hit ⇒
    /// zero prefill contribution" semantic — the cost model has a
    /// nonzero baseline (~22ms linear) at zero tokens, which is
    /// noise-floor artifact, not real prefill work.
    ///
    /// The Go reference's `missRate × count × prefillTime(segment_len,
    /// context_len)` is a node-level aggregate where `segment_len` is
    /// the radix-tree edge length, NOT the missed count; that
    /// per-node formula does not translate to per-request semantics
    /// without double-counting the miss ratio. See
    /// `docs/preble-design.md` §COST.
    pub fn update_with_cost(
        &mut self,
        _prefix: &[u64],
        num_tokens: usize,
        context_length: usize,
        decoding_length: usize,
        now: Instant,
    ) {
        let prefill_contrib = if num_tokens > 0 {
            cost_model::prefill_time(self.cost_model, num_tokens, context_length)
        } else {
            0.0
        };
        let decode_contrib = (decoding_length as f64) * TIME_PER_TOKEN_S;
        self.window.push(now, prefill_contrib + decode_contrib);
    }

    /// KV$-aware branch tie-break — count of non-expired entries in
    /// the 3-min window. Equals `|W_P|` from the Preble paper formula.
    pub fn load(&mut self, now: Instant) -> usize {
        self.window.len_at(now)
    }

    /// Load-balancing branch selection — sum of `(PT_r + DT_r)` for
    /// non-expired entries. Equals `L_i` from the Preble paper formula.
    pub fn cost(&mut self, now: Instant) -> f64 {
        self.window.aggregate_at(now).0
    }

    /// Default expected output length when the request's decode size
    /// is unknown at insert time. Go: `45`.
    pub fn default_decoding_length(&self) -> usize {
        DEFAULT_DECODING_LENGTH
    }
}

// =========================================================================
// Single-metric ablations: PrebleBsBlockHash / PrebleTpsBlockHash
// =========================================================================
//
// Same prefix-match tree as `PrebleBlockHash` (delegated to
// `inner: RadixTreeBlockHash`), but the per-pod aggregate is a single
// engine-step-driven sliding window — no cost model, no per-request
// hook. Pushed at every forward step from the colocation event loop
// after `sctx.lmetric -= metric_delta`. See
// `engine/colocation.rs`.
//
// Each type is gated behind its own feature flag so dead-code lints
// stay quiet under non-matching builds. Mutually exclusive with each
// other and with `PrebleBlockHash`; the alias chain in
// `scheduler/kvcache.rs` enforces a single active flavour.

/// Preble-BS flavour. Per-step window of batch-size samples; load is
/// the sum of BS samples in the past 3 min (higher = more loaded).
#[cfg(feature = "preble-bs-q")]
pub struct PrebleBsBlockHash {
    inner: RadixTreeBlockHash,
    bs_window: SlidingWindow<f64, Sum>,
}

#[cfg(feature = "preble-bs-q")]
unsafe impl Send for PrebleBsBlockHash {}
#[cfg(feature = "preble-bs-q")]
unsafe impl Sync for PrebleBsBlockHash {}

#[cfg(feature = "preble-bs-q")]
impl BlockHash for PrebleBsBlockHash {
    fn new(num_blocks: usize) -> Self {
        Self {
            inner: RadixTreeBlockHash::new(num_blocks),
            bs_window: SlidingWindow::new(WINDOW_DURATION, Sum::default()),
        }
    }
    fn len(&self) -> usize { self.inner.len() }
    fn is_empty(&self) -> bool { self.inner.is_empty() }
    fn insert(&mut self, h: &[u64], i: Vec<u64>) -> usize { self.inner.insert(h, i) }
    fn get(&self, h: &[u64]) -> usize { self.inner.get(h) }
    fn remove(&mut self, i: Vec<u64>) { self.inner.remove(i); }
    fn epoch(&self) -> u64 { self.inner.epoch() }
}

#[cfg(feature = "preble-bs-q")]
impl PrebleBsBlockHash {
    /// Push the current batch-size sample. Called once per forward
    /// step from the colocation event loop, after `LMetric` has been
    /// updated for that step (so `bs` reflects the post-step value).
    pub fn update_with_step(&mut self, bs: usize, now: Instant) {
        self.bs_window.push(now, bs as f64);
    }

    /// Sum of BS samples in the 3-min window. Higher = more loaded.
    /// Load-balancing-branch selector for `PrebleBsQ` is
    /// `select_min_by` over this.
    pub fn bs_sum(&mut self, now: Instant) -> f64 {
        self.bs_window.aggregate_at(now).0
    }
}

/// Preble-TPS flavour. Per-step window of forward-step counts; load
/// is *inversely* proportional to the count of steps in the past 3
/// min (more steps = more throughput headroom = preferred).
#[cfg(feature = "preble-tps-q")]
pub struct PrebleTpsBlockHash {
    inner: RadixTreeBlockHash,
    tps_window: SlidingWindow<(), Count>,
}

#[cfg(feature = "preble-tps-q")]
unsafe impl Send for PrebleTpsBlockHash {}
#[cfg(feature = "preble-tps-q")]
unsafe impl Sync for PrebleTpsBlockHash {}

#[cfg(feature = "preble-tps-q")]
impl BlockHash for PrebleTpsBlockHash {
    fn new(num_blocks: usize) -> Self {
        Self {
            inner: RadixTreeBlockHash::new(num_blocks),
            tps_window: SlidingWindow::new(WINDOW_DURATION, Count),
        }
    }
    fn len(&self) -> usize { self.inner.len() }
    fn is_empty(&self) -> bool { self.inner.is_empty() }
    fn insert(&mut self, h: &[u64], i: Vec<u64>) -> usize { self.inner.insert(h, i) }
    fn get(&self, h: &[u64]) -> usize { self.inner.get(h) }
    fn remove(&mut self, i: Vec<u64>) { self.inner.remove(i); }
    fn epoch(&self) -> u64 { self.inner.epoch() }
}

#[cfg(feature = "preble-tps-q")]
impl PrebleTpsBlockHash {
    /// Bump the forward-step count. `_bs` is accepted for call-site
    /// uniformity with `PrebleBsBlockHash::update_with_step` (the
    /// colocation loop does not know which flavour is aliased in).
    pub fn update_with_step(&mut self, _bs: usize, now: Instant) {
        self.tps_window.push(now, ());
    }

    /// Count of forward steps in the 3-min window. Higher = more
    /// throughput. Load-balancing-branch selector for `PrebleTpsQ`
    /// is `select_max_by` over this.
    pub fn tps_count(&mut self, now: Instant) -> usize {
        self.tps_window.len_at(now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "preble-q")]
    fn fresh() -> PrebleBlockHash {
        PrebleBlockHash::new(1024)
    }

    #[cfg(feature = "preble-q")]
    #[test]
    fn empty_state_zero_load_and_cost() {
        let mut p = fresh();
        let now = Instant::now();
        assert_eq!(p.load(now), 0);
        assert_eq!(p.cost(now), 0.0);
    }

    #[cfg(feature = "preble-q")]
    #[test]
    fn update_pushes_window_entry() {
        let mut p = fresh();
        let t = Instant::now();
        // 200 token input with 100 new tokens → miss_rate = 0.5
        p.update_with_cost(&[1, 2, 3], 100, 200, 45, t);
        assert_eq!(p.load(t), 1);
        // cost > 0 because miss_rate > 0 and decode contribution > 0
        let cost = p.cost(t);
        assert!(cost > 0.0, "cost should be positive after one update");
    }

    #[cfg(feature = "preble-q")]
    #[test]
    fn three_updates_three_load() {
        let mut p = fresh();
        let t = Instant::now();
        p.update_with_cost(&[1], 100, 200, 45, t);
        p.update_with_cost(&[1, 2], 50, 200, 45, t);
        p.update_with_cost(&[1, 2, 3], 0, 200, 45, t);
        assert_eq!(p.load(t), 3);
    }

    #[cfg(feature = "preble-q")]
    #[test]
    fn window_expires_after_3min() {
        let mut p = fresh();
        let t = Instant::now();
        p.update_with_cost(&[1], 100, 200, 45, t);
        // Just before expiry — entry still present.
        assert_eq!(p.load(t + Duration::from_secs(179)), 1);
        // Past 3 min — entry expired.
        assert_eq!(p.load(t + Duration::from_secs(181)), 0);
        assert_eq!(p.cost(t + Duration::from_secs(181)), 0.0);
    }

    #[cfg(feature = "preble-q")]
    #[test]
    fn full_cache_hit_zero_prefill_contribution() {
        // num_tokens = 0 → miss_rate = 0 → prefill_contrib = 0; only
        // the decode term contributes.
        let mut p = fresh();
        let t = Instant::now();
        p.update_with_cost(&[1, 2], 0, 200, 45, t);
        let expected_decode = 45.0 * TIME_PER_TOKEN_S;
        assert!((p.cost(t) - expected_decode).abs() < 1e-9);
    }

    #[cfg(feature = "preble-q")]
    #[test]
    fn delegates_blockhash_trait_to_inner() {
        let mut p = fresh();
        // Insert path through trait method.
        let n = p.insert(&[10, 20, 30], vec![0, 1, 2]);
        assert!(n > 0);
        assert_eq!(p.get(&[10, 20]), 2);
        assert_eq!(p.get(&[10, 20, 30]), 3);
        assert_eq!(p.get(&[10, 20, 30, 40]), 3);
        // Remove via trait method.
        p.remove(vec![0, 1, 2]);
        // After removing all blocks, the get matches drop to 0.
        assert_eq!(p.get(&[10, 20, 30]), 0);
    }

    #[cfg(feature = "preble-q")]
    #[test]
    fn engine_driven_state_does_not_touch_window() {
        // BlockHash::insert / remove must NOT push into the sliding
        // window — only update_with_cost does.
        let mut p = fresh();
        let t = Instant::now();
        p.insert(&[10, 20, 30], vec![0, 1, 2]);
        assert_eq!(p.load(t), 0);
        assert_eq!(p.cost(t), 0.0);
        p.remove(vec![0, 1, 2]);
        assert_eq!(p.load(t), 0);
    }

    // ----- Sanity tests for PrebleBsBlockHash / PrebleTpsBlockHash -----

    #[cfg(feature = "preble-bs-q")]
    #[test]
    fn bs_empty_window_zero() {
        let mut p = PrebleBsBlockHash::new(64);
        assert_eq!(p.bs_sum(Instant::now()), 0.0);
    }

    #[cfg(feature = "preble-bs-q")]
    #[test]
    fn bs_step_pushes_and_sums() {
        let mut p = PrebleBsBlockHash::new(64);
        let t = Instant::now();
        p.update_with_step(8, t);
        p.update_with_step(12, t);
        p.update_with_step(4, t);
        assert!((p.bs_sum(t) - 24.0).abs() < 1e-9);
    }

    #[cfg(feature = "preble-bs-q")]
    #[test]
    fn bs_window_expires_after_3min() {
        let mut p = PrebleBsBlockHash::new(64);
        let t = Instant::now();
        p.update_with_step(10, t);
        assert!((p.bs_sum(t + Duration::from_secs(179)) - 10.0).abs() < 1e-9);
        assert_eq!(p.bs_sum(t + Duration::from_secs(181)), 0.0);
    }

    #[cfg(feature = "preble-bs-q")]
    #[test]
    fn bs_blockhash_trait_delegates() {
        let mut p = PrebleBsBlockHash::new(64);
        p.insert(&[1, 2, 3], vec![0, 1, 2]);
        assert_eq!(p.get(&[1, 2]), 2);
        assert_eq!(p.bs_sum(Instant::now()), 0.0); // tree ops do NOT touch window
    }

    #[cfg(feature = "preble-tps-q")]
    #[test]
    fn tps_empty_window_zero() {
        let mut p = PrebleTpsBlockHash::new(64);
        assert_eq!(p.tps_count(Instant::now()), 0);
    }

    #[cfg(feature = "preble-tps-q")]
    #[test]
    fn tps_step_increments_count() {
        let mut p = PrebleTpsBlockHash::new(64);
        let t = Instant::now();
        for _ in 0..5 {
            p.update_with_step(8, t);
        }
        assert_eq!(p.tps_count(t), 5);
    }

    #[cfg(feature = "preble-tps-q")]
    #[test]
    fn tps_window_expires_after_3min() {
        let mut p = PrebleTpsBlockHash::new(64);
        let t = Instant::now();
        p.update_with_step(8, t);
        p.update_with_step(8, t);
        assert_eq!(p.tps_count(t + Duration::from_secs(179)), 2);
        assert_eq!(p.tps_count(t + Duration::from_secs(181)), 0);
    }
}
