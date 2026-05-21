//! Time-windowed aggregate. Generic over `(T, A: Aggregate<Item = T>)`.
//!
//! Entries are time-stamped `T`s. Old entries (older than `duration`)
//! are evicted lazily on [`SlidingWindow::expire`] — no background
//! thread, no 1 Hz timer. The aggregate `A` is maintained incrementally
//! on `push` (`A::add`) and `expire` (`A::sub`); reads are O(1) after a
//! cheap amortized expire pass.
//!
//! Used by the Preble policy
//! (`router::scheduler::policies::preble::PrebleBlockHash`) for the
//! per-replica `pod_load` (= deque length after expire) and `pod_cost`
//! (= incrementally-maintained `Sum`). One window covers both — count
//! falls out of `len()` for free; only the sum needs an aggregate.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Incrementally-maintained aggregate over a stream of `Item`s.
///
/// `add` is called when an item enters the window; `sub` when it
/// exits. `add` followed by `sub` of the same item must be a no-op
/// for the aggregate to be stable across long-running sequences.
pub trait Aggregate {
    type Item;
    fn add(&mut self, x: &Self::Item);
    fn sub(&mut self, x: &Self::Item);
}

/// Running sum over `f64` items.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Sum(pub f64);

impl Aggregate for Sum {
    type Item = f64;
    fn add(&mut self, x: &f64) {
        self.0 += *x;
    }
    fn sub(&mut self, x: &f64) {
        self.0 -= *x;
    }
}

pub struct SlidingWindow<T, A: Aggregate<Item = T>> {
    entries: VecDeque<(Instant, T)>,
    agg: A,
    duration: Duration,
}

impl<T, A: Aggregate<Item = T>> SlidingWindow<T, A> {
    pub fn new(duration: Duration, agg_init: A) -> Self {
        Self { entries: VecDeque::new(), agg: agg_init, duration }
    }

    /// Append `(t, item)` to the back of the window.
    ///
    /// Caller is responsible for monotonic `t` — pushing an item with
    /// `t` earlier than the current back is accepted but disturbs the
    /// time ordering and may confuse subsequent `expire` calls.
    pub fn push(&mut self, t: Instant, item: T) {
        self.agg.add(&item);
        self.entries.push_back((t, item));
    }

    /// Drop entries whose timestamp is older than `now − duration`,
    /// calling `agg.sub` for each. Amortized O(1) per push.
    pub fn expire(&mut self, now: Instant) {
        while let Some(&(t, _)) = self.entries.front() {
            if now.saturating_duration_since(t) > self.duration {
                let (_, item) = self.entries.pop_front().expect("front just peeked");
                self.agg.sub(&item);
            } else {
                break;
            }
        }
    }

    /// Aggregate without expiring first. Reflects the state as of the
    /// last `push` / `expire`. Use [`SlidingWindow::aggregate_at`] if
    /// you want fresh-as-of-`now`.
    pub fn aggregate(&self) -> &A {
        &self.agg
    }

    /// Number of entries currently in the window. Reflects state as
    /// of the last `push` / `expire`. Use [`SlidingWindow::len_at`]
    /// for fresh-as-of-`now`.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Convenience: `expire(now)` then aggregate.
    pub fn aggregate_at(&mut self, now: Instant) -> &A {
        self.expire(now);
        &self.agg
    }

    /// Convenience: `expire(now)` then length.
    pub fn len_at(&mut self, now: Instant) -> usize {
        self.expire(now);
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0() -> Instant {
        Instant::now()
    }

    #[test]
    fn empty_window_is_zero() {
        let w: SlidingWindow<f64, Sum> =
            SlidingWindow::new(Duration::from_secs(60), Sum::default());
        assert_eq!(w.aggregate().0, 0.0);
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
    }

    #[test]
    fn push_increments_aggregate_and_len() {
        let mut w: SlidingWindow<f64, Sum> =
            SlidingWindow::new(Duration::from_secs(60), Sum::default());
        let t = t0();
        w.push(t, 1.5);
        w.push(t, 2.5);
        assert_eq!(w.aggregate().0, 4.0);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn expire_removes_old_entries_and_decrements_aggregate() {
        let mut w: SlidingWindow<f64, Sum> =
            SlidingWindow::new(Duration::from_secs(10), Sum::default());
        let t = t0();
        w.push(t, 1.0);
        w.push(t + Duration::from_secs(2), 2.0);
        w.push(t + Duration::from_secs(8), 4.0);
        // 9s elapsed since t: nothing > 10s old, all three remain.
        w.expire(t + Duration::from_secs(9));
        assert_eq!(w.aggregate().0, 7.0);
        assert_eq!(w.len(), 3);
        // 13s elapsed since t: first (t) is now > 10s old; second (t+2s)
        // is 11s old, also expired; third (t+8s) is 5s old, kept.
        w.expire(t + Duration::from_secs(13));
        assert_eq!(w.aggregate().0, 4.0);
        assert_eq!(w.len(), 1);
        // 30s elapsed: all expired.
        w.expire(t + Duration::from_secs(30));
        assert_eq!(w.aggregate().0, 0.0);
        assert_eq!(w.len(), 0);
    }

    #[test]
    fn aggregate_at_expires_then_reads() {
        let mut w: SlidingWindow<f64, Sum> =
            SlidingWindow::new(Duration::from_secs(5), Sum::default());
        let t = t0();
        w.push(t, 10.0);
        w.push(t + Duration::from_secs(1), 20.0);
        // 10s elapsed: both > 5s old, both expire.
        assert_eq!(w.aggregate_at(t + Duration::from_secs(10)).0, 0.0);
        assert_eq!(w.len_at(t + Duration::from_secs(10)), 0);
    }

    #[test]
    fn aggregate_matches_replay() {
        // Push N items at staggered times; verify aggregate equals the
        // sum over non-expired entries computed by direct iteration.
        let mut w: SlidingWindow<f64, Sum> =
            SlidingWindow::new(Duration::from_secs(30), Sum::default());
        let t = t0();
        let items: Vec<(u64, f64)> = (0..50)
            .map(|i| (i as u64 * 1, (i as f64) * 0.7 - 5.0))
            .collect();
        for &(dt_s, v) in &items {
            w.push(t + Duration::from_secs(dt_s), v);
        }
        // Now scan at several time points and cross-check.
        for now_dt_s in [0, 10, 25, 30, 40, 60, 100] {
            let now = t + Duration::from_secs(now_dt_s);
            w.expire(now);
            let expected: f64 = items
                .iter()
                .filter(|&&(dt_s, _)| {
                    now.saturating_duration_since(t + Duration::from_secs(dt_s))
                        <= Duration::from_secs(30)
                })
                .map(|&(_, v)| v)
                .sum();
            assert!(
                (w.aggregate().0 - expected).abs() < 1e-9,
                "at +{}s: aggregate={} expected={}",
                now_dt_s,
                w.aggregate().0,
                expected,
            );
        }
    }

    #[test]
    fn boundary_at_exact_duration_is_kept() {
        // An entry exactly `duration` old should remain (we use `>`).
        let mut w: SlidingWindow<f64, Sum> =
            SlidingWindow::new(Duration::from_secs(5), Sum::default());
        let t = t0();
        w.push(t, 7.0);
        w.expire(t + Duration::from_secs(5));
        assert_eq!(w.aggregate().0, 7.0);
        assert_eq!(w.len(), 1);
        w.expire(t + Duration::from_secs(5) + Duration::from_nanos(1));
        assert_eq!(w.aggregate().0, 0.0);
        assert_eq!(w.len(), 0);
    }

    #[test]
    fn negative_values_and_signed_sum() {
        // Sum aggregate handles negatives correctly.
        let mut w: SlidingWindow<f64, Sum> =
            SlidingWindow::new(Duration::from_secs(60), Sum::default());
        let t = t0();
        w.push(t, 10.0);
        w.push(t, -3.5);
        w.push(t, -6.5);
        assert_eq!(w.aggregate().0, 0.0);
    }
}
