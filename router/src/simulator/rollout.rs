// Layer-3 ephemeral rollout — the output of one `PCtx::query(candidate)`
// call. One slot ↔ one engine forward step rolled forward by the outer
// discrete-event simulator. Composed of a `BatchForPredictor` (the
// regressor's input feature vector) and the predicted per-step latency
// in milliseconds.
//
// The buffer's three index markers (`prefill_begin_step`,
// `prefill_end_step`, `in_decode_step`) project the candidate's
// life-cycle onto the slot timeline so that `RolloutGist` can be
// extracted in O(1):
//
//   * `prefill_begin_step` = first slot where the candidate is in
//     PREFILL state (its first chunk is scheduled).
//   * `prefill_end_step`   = last slot where the candidate is in
//     PREFILL state (its final chunk completes; ttft_ms = sum of slot
//     latencies through this index).
//   * `in_decode_step`     = first slot where the batch composition
//     reflects steady-state decode after `prefill_end_step`. Used as
//     the in-decode TBT estimate.
//
// `candidate_id == None` marks a *baseline* rollout (no admission;
// invariant-keeping rollout for a PCtx whose engine has prefill work
// but no associated candidate query).

#[derive(Debug, Clone)]
pub struct RolloutSlot {
    pub batch: super::BatchForPredictor,
    pub predicted_lat_ms: f32,
}

#[derive(Debug, Clone, Default)]
pub struct RolloutBuffer {
    /// `Some(req_id)` for a candidate-bound rollout (produced by
    /// `query(candidate)`). `None` for the candidate-free baseline
    /// rebuilt on the SSE path to satisfy the non-empty invariant.
    pub candidate_id: Option<u64>,
    pub slots: Vec<RolloutSlot>,
    pub prefill_begin_step: Option<usize>,
    pub prefill_end_step: Option<usize>,
    pub in_decode_step: Option<usize>,
}

impl RolloutBuffer {
    /// Project the buffer down to the per-replica scoring fields the
    /// scheduling policy actually consumes. All three fields are
    /// `Option` because either the candidate's prefill never started
    /// (`prefill_begin_step` is None) or in-decode steady-state was
    /// not reached within the rollout horizon.
    pub fn gist(&self) -> RolloutGist {
        let ttft_ms = self.prefill_end_step.map(|i| {
            self.slots
                .iter()
                .take(i + 1)
                .map(|s| s.predicted_lat_ms)
                .sum()
        });
        let chunked_prefill_steps = match (self.prefill_begin_step, self.prefill_end_step) {
            (Some(begin), Some(end)) if end >= begin => Some(end - begin + 1),
            _ => None,
        };
        let in_decode_tbt_ms =
            self.in_decode_step.and_then(|i| self.slots.get(i)).map(|s| s.predicted_lat_ms);
        RolloutGist { ttft_ms, chunked_prefill_steps, in_decode_tbt_ms }
    }
}

/// Per-replica scoring summary derived from the latest `RolloutBuffer`.
/// All three fields are `Option` to communicate "not yet known" rather
/// than fabricating a default value.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RolloutGist {
    pub ttft_ms: Option<f32>,
    pub chunked_prefill_steps: Option<usize>,
    pub in_decode_tbt_ms: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::BatchForPredictor;

    fn slot(latency: f32) -> RolloutSlot {
        RolloutSlot { batch: BatchForPredictor::default(), predicted_lat_ms: latency }
    }

    #[test]
    fn gist_extracts_three_fields() {
        let buf = RolloutBuffer {
            candidate_id: Some(7),
            slots: vec![slot(2.0), slot(3.0), slot(4.0), slot(5.0)],
            prefill_begin_step: Some(0),
            prefill_end_step: Some(2),
            in_decode_step: Some(3),
        };
        let g = buf.gist();
        assert_eq!(g.ttft_ms, Some(9.0));
        assert_eq!(g.chunked_prefill_steps, Some(3));
        assert_eq!(g.in_decode_tbt_ms, Some(5.0));
    }

    #[test]
    fn gist_baseline_with_no_candidate() {
        let buf = RolloutBuffer {
            candidate_id: None,
            slots: vec![slot(2.0), slot(3.0)],
            prefill_begin_step: None,
            prefill_end_step: None,
            in_decode_step: None,
        };
        let g = buf.gist();
        assert_eq!(g, RolloutGist::default());
    }

    #[test]
    fn gist_partial_only_in_decode() {
        let buf = RolloutBuffer {
            candidate_id: Some(1),
            slots: vec![slot(2.0)],
            prefill_begin_step: None,
            prefill_end_step: None,
            in_decode_step: Some(0),
        };
        let g = buf.gist();
        assert_eq!(g.ttft_ms, None);
        assert_eq!(g.chunked_prefill_steps, None);
        assert_eq!(g.in_decode_tbt_ms, Some(2.0));
    }
}
