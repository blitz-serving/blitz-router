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

/// Read the CLI-tunable Preble sliding-window duration, in seconds.
/// Default 180s (3 min, paper-faithful). Shared by all three Preble
/// flavours.
fn preble_window_duration() -> Duration {
    Duration::from_secs(
        crate::scheduler::state::PREBLE_WINDOW_SECS
            .get()
            .copied()
            .unwrap_or(180),
    )
}

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
            window: SlidingWindow::new(preble_window_duration(), Sum::default()),
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
// sliding window — no cost model. Push sites differ per flavour:
//
// - `PrebleBsBlockHash`: **push-both** — admission events (from
//   `PrebleBsQ::after_extra`) AND SSE forward-step events (from the
//   colocation loop). Both push the engine's current BS into the
//   window; the combined samples approximate ∫ BS(t) dt × event_rate.
//   Admission pushes give responsiveness before the next decision;
//   SSE pushes capture runtime occupancy.
// - `PrebleTpsBlockHash`: **SSE-only** — forward-step count window,
//   with idle-engine sentinel + idle-period compensation hooks
//   from `PrebleTpsQ`.
//
// Each type is gated behind its own feature flag so dead-code lints
// stay quiet under non-matching builds. Mutually exclusive with each
// other and with `PrebleBlockHash`; the alias chain in
// `scheduler/kvcache.rs` enforces a single active flavour.

/// Preble-BS flavour. **Push-both** window of BS snapshots:
///
/// - Admission events: `PrebleBsQ::after_extra` pushes the
///   post-`LMetricInc` BS once per admission. Makes the metric
///   responsive *before the next routing decision* — the gap that
///   broke the earlier engine-step-only design (ali-h20 `_1p` campaign
///   measured max-burst 1094 under step-only vs 3 under preble-q's
///   admission-driven cost).
/// - SSE forward-step events: the colocation loop pushes pre-subassign
///   BS once per engine tick. Captures runtime occupancy and prevents
///   semantic drift toward "arrival-weighted queue pressure only" —
///   a short burst that completes quickly is balanced by subsequent
///   low-BS SSE samples.
///
/// Both push the engine's instantaneous BS at the event moment. The
/// SlidingWindow `Sum` aggregate accumulates them; the combined
/// samples approximate ∫ BS(t) dt over the window, scaled by event
/// rate. Two regimes self-select:
///
/// - **Cold / idle replicas**: SSE doesn't tick (`bs=0` ⇒ no forward
///   steps in vLLM/yaullm). Admission pushes dominate the window →
///   strong admission-responsiveness drives cold-burst fan-out.
/// - **Warm / busy replicas**: SSE (~120fps at peak decode) outnumbers
///   admission rate by ~50×. SSE samples carry the load signal:
///   `Σ BS_SSE` for a BS=10 engine is ~10× that of a BS=1 engine, so
///   `min_by` strongly prefers the lighter engine.
///
/// Why the same `update_with_step` method for both call sites: the
/// operation is identical (push current BS at the event timestamp).
/// The method name follows the colocation convention; the after_extra
/// caller is documented at `preble_bs_update_after`.
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
            bs_window: SlidingWindow::new(preble_window_duration(), Sum::default()),
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
    /// Push a BS snapshot into the 3-min window. Called from TWO
    /// sites in the push-both design:
    ///
    /// 1. **Colocation SSE loop** (`engine/colocation.rs`): once per
    ///    forward step, with the pre-subassign BS — captures runtime
    ///    occupancy.
    /// 2. **`PrebleBsQ::after_extra`** (via `preble_bs_update_after`):
    ///    once per admission, with the post-`LMetricInc` BS — keeps
    ///    the metric responsive to routing decisions before the next
    ///    decision is taken.
    ///
    /// Both call sites push the engine's instantaneous BS at the
    /// event moment; the SlidingWindow Sum aggregate accumulates
    /// them. Combined samples approximate ∫ BS(t) dt × event_rate
    /// over the window.
    ///
    /// Tree state is independently maintained by the colocation event
    /// loop's `BlockHash::insert`.
    pub fn update_with_step(&mut self, bs: usize, now: Instant) {
        self.bs_window.push(now, bs as f64);
    }

    /// Sum of BS snapshots (admission + SSE events) in the 3-min
    /// window. Higher = more loaded. Load-balancing-branch selector
    /// for `PrebleBsQ` is `select_min_by` over this.
    pub fn bs_sum(&mut self, now: Instant) -> f64 {
        self.bs_window.aggregate_at(now).0
    }
}

/// Preble-TPS flavour. Per-step window of forward-step counts; load
/// is *inversely* proportional to the count of steps in the past
/// window (more steps = more throughput headroom = preferred).
///
/// The `last_busy_at` scalar is **deliberately separate** from the
/// sliding window: when an engine has been idle past the window
/// duration, the deque has popped all real samples, but we still
/// need the boundary timestamp to compute the idle gap for
/// compensation at the next idle→busy transition. Storing
/// `last_busy_at` inside the deque would lose this information.
#[cfg(feature = "preble-tps-q")]
pub struct PrebleTpsBlockHash {
    inner: RadixTreeBlockHash,
    tps_window: SlidingWindow<(), Count>,
    /// Timestamp of the most recent forward step. `None` only on a
    /// cold-start engine that has never ticked. Updated on every
    /// `update_with_step` call. Used by `compensate_idle_gap` to size
    /// the idle interval to backfill.
    last_busy_at: Option<Instant>,
    /// Window duration captured at construction time from
    /// `PREBLE_WINDOW_SECS` (CLI-tunable; default 180s).
    window_duration: Duration,
    /// Idle-period compensation rate captured at construction time
    /// from `PREBLE_IDLE_TPS` (CLI-tunable; default 120 fps).
    idle_tps: f32,
}

#[cfg(feature = "preble-tps-q")]
unsafe impl Send for PrebleTpsBlockHash {}
#[cfg(feature = "preble-tps-q")]
unsafe impl Sync for PrebleTpsBlockHash {}

#[cfg(feature = "preble-tps-q")]
impl BlockHash for PrebleTpsBlockHash {
    fn new(num_blocks: usize) -> Self {
        let window_duration = preble_window_duration();
        let idle_tps = crate::scheduler::state::PREBLE_IDLE_TPS
            .get()
            .copied()
            .unwrap_or(120.0);
        Self {
            inner: RadixTreeBlockHash::new(num_blocks),
            tps_window: SlidingWindow::new(window_duration, Count),
            last_busy_at: None,
            window_duration,
            idle_tps,
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
    /// Push one real forward-step sample and refresh `last_busy_at`.
    /// `_bs` is accepted for call-site uniformity with
    /// `PrebleBsBlockHash::update_with_step` (the colocation loop
    /// does not know which flavour is aliased in).
    ///
    /// Forward steps fire only when `bs > 0` on the engine side, so
    /// this method is never called on an idle engine — meaning
    /// `last_busy_at == Some(now)` after the call holds the
    /// "most recent moment the engine ticked" invariant.
    pub fn update_with_step(&mut self, _bs: usize, now: Instant) {
        self.tps_window.push(now, ());
        self.last_busy_at = Some(now);
    }

    /// Retroactively credit a now-elapsed idle interval as if the
    /// engine had been ticking at `idle_tps`. Called exactly when
    /// an admission lifts the engine from `bs=0` to `bs>0` — the
    /// edge of the `None ⇔ compensation` state-chain.
    ///
    /// The idle interval is `now - last_busy_at`, clamped to
    /// `window_duration` (we never compensate beyond the window's
    /// look-back). Synthetic samples are dated uniformly across the
    /// RECENT window `[now - idle_dur, now]`, not from
    /// `last_busy_at` forward — for long idle periods past the
    /// window the latter would write samples already older than the
    /// window and they'd be evicted on the next read. The intent is
    /// "this engine has been productive for the past N seconds" with
    /// N = min(real_idle, window).
    ///
    /// If `last_busy_at` is `None` (cold-start engine, never been
    /// busy), there is no idle interval to compensate — the window
    /// stays empty; Design 1's `None` sentinel still ensures idle
    /// engines win admissions.
    pub fn compensate_idle_gap(&mut self, now: Instant) {
        if let Some(last) = self.last_busy_at {
            let idle_dur = now.saturating_duration_since(last).min(self.window_duration);
            let count = (self.idle_tps * idle_dur.as_secs_f32()) as usize;
            if count > 0 {
                let step = idle_dur / (count as u32);
                let start = now - idle_dur;
                for i in 1..=count {
                    self.tps_window.push(start + step * (i as u32), ());
                }
            }
        }
    }

    /// Effective tps_count at decision time. Returns `None` for an
    /// idle engine (`bs == 0`) — Design 1's sentinel: idle engines
    /// dominate any busy engine in `select_max_by` via the
    /// `None > Some(_)` collapsing in the helper.
    ///
    /// For a busy engine, returns the current sliding-window count
    /// (real busy samples + any synthetic compensation previously
    /// pushed by `compensate_idle_gap`).
    pub fn tps_count(&mut self, bs: usize, now: Instant) -> Option<usize> {
        if bs == 0 { None } else { Some(self.tps_window.len_at(now)) }
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
    fn bs_push_both_admission_and_sse_samples() {
        // Push-both design: admission and SSE both call update_with_step
        // with the engine's current BS at that moment. The Sum
        // aggregate accumulates both.
        let mut p = PrebleBsBlockHash::new(64);
        let t = Instant::now();
        // Three admissions to an initially-idle engine — bs becomes
        // 1, 2, 3 post-LMetricInc. After-extra pushes those.
        p.update_with_step(1, t);
        p.update_with_step(2, t);
        p.update_with_step(3, t);
        // Two SSE forward steps at the same moment, recording pre-
        // subassign BS = 3 (no completion this step).
        p.update_with_step(3, t);
        p.update_with_step(3, t);
        // Sum = 1 + 2 + 3 + 3 + 3 = 12.
        assert!((p.bs_sum(t) - 12.0).abs() < 1e-9);
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
    fn tps_idle_returns_none_busy_returns_count() {
        // bs == 0 ⇒ None (sentinel for "idle, wins via select_max_by");
        // bs > 0  ⇒ Some(window.len_at(now)).
        let mut p = PrebleTpsBlockHash::new(64);
        let t = Instant::now();
        assert_eq!(p.tps_count(0, t), None);
        p.update_with_step(1, t);
        assert_eq!(p.tps_count(1, t), Some(1));
        // Becoming idle again immediately re-arms the None sentinel.
        assert_eq!(p.tps_count(0, t), None);
    }

    #[cfg(feature = "preble-tps-q")]
    #[test]
    fn tps_step_increments_count() {
        let mut p = PrebleTpsBlockHash::new(64);
        let t = Instant::now();
        for _ in 0..5 {
            p.update_with_step(8, t);
        }
        assert_eq!(p.tps_count(8, t), Some(5));
    }

    #[cfg(feature = "preble-tps-q")]
    #[test]
    fn tps_window_expires_after_default_window() {
        let mut p = PrebleTpsBlockHash::new(64);
        let t = Instant::now();
        p.update_with_step(8, t);
        p.update_with_step(8, t);
        assert_eq!(p.tps_count(8, t + Duration::from_secs(179)), Some(2));
        assert_eq!(p.tps_count(8, t + Duration::from_secs(181)), Some(0));
    }

    #[cfg(feature = "preble-tps-q")]
    #[test]
    fn tps_compensate_cold_start_is_noop() {
        // No prior busy state ⇒ last_busy_at == None ⇒ no synthetic
        // samples pushed.
        let mut p = PrebleTpsBlockHash::new(64);
        let t = Instant::now();
        p.compensate_idle_gap(t);
        assert_eq!(p.tps_count(8, t), Some(0));
    }

    #[cfg(feature = "preble-tps-q")]
    #[test]
    fn tps_compensate_after_idle_backfills_at_idle_tps() {
        // Engine ticks once at t, goes idle, comes back at t+10s.
        // Default idle_tps = 120 ⇒ expect ~120*10 = 1200 synthetic
        // samples added, plus the original 1 real sample, all within
        // the window when queried at t+10s.
        let mut p = PrebleTpsBlockHash::new(64);
        let t = Instant::now();
        p.update_with_step(1, t);                   // 1 real
        let returning = t + Duration::from_secs(10);
        p.compensate_idle_gap(returning);            // ~1200 synthetic
        let count = p.tps_count(1, returning).unwrap();
        // Allow ±5 for integer truncation of idle_tps × dur.
        assert!(
            (1200..=1205).contains(&(count - 1)),
            "expected ≈ 1201 samples (1 real + ~1200 synthetic), got {}",
            count
        );
    }

    #[cfg(feature = "preble-tps-q")]
    #[test]
    fn tps_compensate_clamps_to_window_duration() {
        // Idle 1 hour, window = 180s, idle_tps = 120 (default).
        // Pre-fix: synthetic samples were dated [last_busy_at,
        // last_busy_at + 180s] = [now-1h, now-57min], all expired at
        // query time → 0 credit (clamped to nothing).
        // Post-fix: dated [now - 180s, now] → all in-window →
        // 21600 synthetic samples credit. Verifies (a) we don't
        // overshoot 120*3600=432000 samples, AND (b) the recent
        // window is actually populated.
        let mut p = PrebleTpsBlockHash::new(64);
        let t = Instant::now();
        p.update_with_step(1, t);                       // 1 real sample at t
        let returning = t + Duration::from_secs(3600);
        p.compensate_idle_gap(returning);
        let count = p.tps_count(1, returning).unwrap();
        // Real sample at t = returning - 3600s is well out of window.
        // Synthetic samples are now dated in [returning-180s, returning].
        // Expect ~120 * 180 = 21600, ±5 for integer truncation.
        assert!(
            (21_595..=21_605).contains(&count),
            "expected ≈ 21600 synthetic samples in recent window; got {}",
            count
        );
    }

    #[cfg(feature = "preble-tps-q")]
    #[test]
    fn tps_blockhash_trait_delegates() {
        let mut p = PrebleTpsBlockHash::new(64);
        p.insert(&[1, 2, 3], vec![0, 1, 2]);
        assert_eq!(p.get(&[1, 2]), 2);
        // Tree ops do NOT touch the tps window.
        assert_eq!(p.tps_count(8, Instant::now()), Some(0));
    }
}
