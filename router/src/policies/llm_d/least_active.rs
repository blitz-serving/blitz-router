//! `least-active-q` — pick the replica with the fewest active tokens.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2):
//!
//! ```text
//! Select min by sctx.all_tokens
//! ```
//!
//! ## llm-d origin
//!
//! Our port of llm-d's `kv-cache-utilization-scorer` single-scorer
//! ablation. Upstream computes `1 − KVCacheUsagePercent`; we don't track a
//! per-engine KV capacity at the metric layer, so we use `sctx.all_tokens`
//! (total active tokens, prefill + decode) directly. For any fixed engine
//! capacity `CAP`, `1 − all_tokens/CAP` is monotonic with `-all_tokens`,
//! so the argmax is argmin `sctx.all_tokens` regardless of CAP. Dropping
//! the cap makes the policy cap-free without changing rankings.
//!
//! Native name: "least-active" — what's being minimized is total active
//! tokens (prefill load + decode load combined into one usage signal).
//! Distinct from `least-token-load-q` which also adds queued prefill
//! tokens on top.

use policy_dsl::policy;

policy! {
    name: LeastActiveQ,
    gctx: (),
    body: {
        select_min_by(&root_target(&observations), |o| o.all_tokens)
    },
}
