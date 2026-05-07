// RolloutBuffer — the outer discrete-event simulator's output for one
// `query_sim()` call. One slot ↔ one engine forward step. All slots are
// implicitly chunked-prefill steps (the rollout's stop condition is
// `waiting==∅ && chunked_prefill_in_progress==∅`, so the rollout never
// enters a decode-only step). Steady-state TBT after prefill terminates is
// reported via `post_prefill_tbt_ms`, computed by one extra inner-regressor
// call on the post-prefill batch composition.

use smallvec::SmallVec;

pub struct RolloutBuffer {
    /// The candidate request being scored. `None` when the query was a
    /// decode-only forecast (no admission).
    pub candidate_request_id: Option<u64>,
    pub slots: Vec<RolloutSlot>,
    /// Steady-state TBT estimate for the batch composition that remains after
    /// the last chunked-prefill slot completes. One extra inner-regressor call.
    pub post_prefill_tbt_ms: f32,
}

pub struct RolloutSlot {
    pub step_latency_ms: f32,
    /// Requests whose prefill completes in this slot. Each entry: (request_id,
    /// cumulative_ms_at_event), where cumulative_ms is the sum of
    /// `step_latency_ms` over slots `0..=i` for slot index `i`.
    pub ttft_reached: SmallVec<[(u64, f32); 2]>,
    /// Requests whose decode completes in this slot (e.g. short-output
    /// in-flight requests that finish during the chunked-prefill window).
    pub finished: SmallVec<[(u64, f32); 2]>,
}
