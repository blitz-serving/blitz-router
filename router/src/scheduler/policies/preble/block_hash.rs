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

use super::cost_model::{self, TargetGpu};

/// 3 min, matching Go's `slidingWindowPeriod`.
const WINDOW_DURATION: Duration = Duration::from_secs(3 * 60);

/// Hardcoded per-token decode time (seconds), matching Go's constant
/// at `prefix_cache_preble.go:345`. The `avgTimePerTokenPerPod` Go map
/// is dead code in production (only tests assign), so we do not
/// reproduce it.
const TIME_PER_TOKEN_S: f64 = 0.15;

/// Default expected output length when the actual decoding length is
/// unknown at insert time. Go: `decodingLength = 45`
/// (`prefix_cache_preble.go`'s env default).
const DEFAULT_DECODING_LENGTH: usize = 45;

/// Preble-flavoured `BlockHash`. See module docs.
pub struct PrebleBlockHash {
    inner: RadixTreeBlockHash,
    window: SlidingWindow<f64, Sum>,
    cost_model: TargetGpu,
}

unsafe impl Send for PrebleBlockHash {}
unsafe impl Sync for PrebleBlockHash {}

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

impl PrebleBlockHash {
    /// Push a per-request cost contribution to the sliding window.
    ///
    /// Called from `PrebleQ`'s `after_extra`. Tree state is already
    /// up-to-date via the colocation event loop's `insert` (Preble does
    /// not separately materialize the path).
    ///
    /// `num_tokens` is the count of NEW (uncached) tokens — drives
    /// `miss_rate`. `context_length` is the FULL input token count.
    /// `decoding_length` is the expected output length (caller passes
    /// [`PrebleBlockHash::default_decoding_length`] when unknown).
    pub fn update_with_cost(
        &mut self,
        _prefix: &[u64],
        num_tokens: usize,
        context_length: usize,
        decoding_length: usize,
        now: Instant,
    ) {
        let miss_rate = if context_length > 0 {
            (num_tokens as f64) / (context_length as f64)
        } else {
            1.0
        };
        let prefill_t = cost_model::prefill_time(self.cost_model, num_tokens, context_length);
        let prefill_contrib = miss_rate * prefill_t;
        let decode_contrib = (decoding_length as f64) * TIME_PER_TOKEN_S;
        self.window.push(now, prefill_contrib + decode_contrib);
    }

    /// Stage 1 tie-break — count of non-expired entries in the
    /// 3-min window. Equals `|W_P|` from the Preble paper formula.
    pub fn load(&mut self, now: Instant) -> usize {
        self.window.len_at(now)
    }

    /// Stage 2 selection — sum of `(PT_r + DT_r)` for non-expired
    /// entries. Equals `L_i` from the Preble paper formula.
    pub fn cost(&mut self, now: Instant) -> f64 {
        self.window.aggregate_at(now).0
    }

    /// Default expected output length when the request's decode size
    /// is unknown at insert time. Go: `45`.
    pub fn default_decoding_length(&self) -> usize {
        DEFAULT_DECODING_LENGTH
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> PrebleBlockHash {
        PrebleBlockHash::new(1024)
    }

    #[test]
    fn empty_state_zero_load_and_cost() {
        let mut p = fresh();
        let now = Instant::now();
        assert_eq!(p.load(now), 0);
        assert_eq!(p.cost(now), 0.0);
    }

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

    #[test]
    fn three_updates_three_load() {
        let mut p = fresh();
        let t = Instant::now();
        p.update_with_cost(&[1], 100, 200, 45, t);
        p.update_with_cost(&[1, 2], 50, 200, 45, t);
        p.update_with_cost(&[1, 2, 3], 0, 200, 45, t);
        assert_eq!(p.load(t), 3);
    }

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
}
