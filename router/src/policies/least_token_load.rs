//! `least-token-load-q` — pick the replica with the least total token load
//! (queued prefill + currently active).
//!
//! Paper-form DSL (`docs/dsl-schema.md` §8):
//!
//! ```text
//! Select min by queued_tokens(sctx) + sctx.all_tokens
//! ```
//!
//! ## llm-d origin
//!
//! Our port of llm-d's `token-load-scorer` single-scorer ablation. Upstream
//! inversely normalizes (queued + active tokens); argmax of
//! `1 − norm(queued + active)` is argmin `(queued + active)`.
//!
//! Native name: "least-token-load" — measures total token load (queued
//! prefill awaiting compute PLUS active tokens already in flight).
//! Captures a combined prefill-and-decode pressure signal. Distinct from:
//!   - `least-active-q` (`all_tokens` only, no queued prefill)
//!   - `least-wait-token-q` in `simple.rs` (per-request prefill only:
//!     `prefill_tokens(req, sctx) = queued + new_for_this_req`; depends on
//!     the candidate request's uncached prefix length, this one does not)

use policy_dsl::policy;

policy! {
    name: LeastTokenLoadQ,
    gctx: (),
    body: {
        select_min_by(&root_target(&observations), |o| queued_tokens(o) + o.all_tokens)
    },
}
