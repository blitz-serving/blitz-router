// Input features for the inner regressor at one engine forward step.
//
// Field set adapted from everparadise (`tmp/blitz-infer-pack-sim/router_v2/
// src/simulator/batch.rs`). `doing_prefill_req` and `prefill_done_request`
// (outer-DES bookkeeping) are dropped. `num_tokens_rounded` is kept because
// it's a scheduler-level concept (rounded to block_size, not to the
// regressor's granularity) and is computed by the outer DES at construction.

#[derive(Debug, Clone, Default)]
pub struct BatchForPredictor {
    /// Total tokens in this forward step (prefill + decode).
    pub num_tokens: usize,
    /// `num_tokens` rounded up to the scheduler's `block_size` (typically 16).
    /// Many per-op lookups key on this rather than `num_tokens`.
    pub num_tokens_rounded: usize,
    /// Per prefill request: tokens scheduled this step.
    pub num_prefill_tokens: Vec<usize>,
    /// Per prefill request: tokens already computed before this step (KV size).
    pub num_prefill_computed_tokens: Vec<usize>,
    /// Per decode request: current KV cache size in tokens.
    pub num_decode_computed_tokens: Vec<usize>,
    /// Total batch size (prefill request count + decode request count).
    pub size: usize,
}
