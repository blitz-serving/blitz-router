//! `most-hit-q` — pick the replica with the most cached prefix blocks.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2):
//!
//! ```text
//! Select max by hit_blocks(req, sctx)
//! ```
//!
//! Counterpart to `bounded-most-hit-q` (in `simple.rs`) without the
//! `queued_tokens < BOUND` gate and without the attention-black-hole
//! fallback — purely "go where the cache hit is biggest".
//!
//! ## llm-d origin
//!
//! Our port of llm-d's production baseline `sim-epp-kvcache-config.yaml`,
//! which enables exactly two scheduler plugins:
//!   - scorer: `precise-prefix-cache-scorer` (weight=10)
//!   - picker:  `max-score-picker`
//!
//! llm-d's `precise-prefix-cache-scorer` min-max-normalizes per-pod KV-block
//! hit counts to `[0, 1]`; the picker shuffles then stable-sorts by score
//! descending. With a SINGLE scorer the min-max + constant weight (10) are
//! monotonic transforms — argmax invariant — so the policy reduces to
//! `argmax hit_blocks`, exactly our DSL form. We name it after what it
//! COMPUTES (`most-hit`) rather than llm-d's plugin labels
//! ("precise-prefix-cache" tells the reader nothing about the algorithm).
//!
//! Two divergences from upstream, worth flagging if the camera-ready paper
//! compares routing decisions block-by-block:
//!   1. **Tiebreak**: llm-d does true random tiebreak (shuffle, then stable
//!      sort). Our `select_max_by` is `Iterator::max_by` (deterministic
//!      last-maximum). Matters in cold-cluster cases (many replicas tied at
//!      `hit_blocks=0`).
//!   2. **Hit-count source**: llm-d's count comes from a per-pod ZMQ-event-
//!      fed KV-block index; ours from the RadixTree (`block_hash.get`).
//!      Same semantic quantity, different machinery.

use policy_dsl::policy;

policy! {
    name: MostHitQ,
    gctx: (),
    body: {
        select_max_by(&root_target(&observations), |o| hit_blocks(req, o))
    },
}
