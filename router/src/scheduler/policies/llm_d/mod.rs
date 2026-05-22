//! Scheduling policies derived from llm-d's gateway scheduler plugins.
//!
//! Each file ports one llm-d scorer (single-scorer ablation) or
//! multi-scorer combination, named after what it COMPUTES rather than
//! after llm-d's plugin labels. The mapping rationale lives in each
//! file's header.
//!
//! - `most-hit-q`              → `precise-prefix-cache-scorer`
//! - `least-waiting-q`         → `load-aware-scorer` / `queue-depth-scorer`
//! - `least-bs-q`              → `running-requests-scorer`
//! - `least-active-q`          → `kv-cache-utilization-scorer`
//! - `least-token-load-q`      → `token-load-scorer`
//! - `most-hit-load-q`         → precise-prefix-cache + load-aware
//! - `most-hit-load-active-q`  → precise-prefix-cache + load-aware + kv-util
//!
//! Reference: <https://github.com/llm-d/llm-d-scheduler> — the
//! `scheduler_profile.go::runScorerPlugins` pipeline. llm-d policies that
//! cannot be expressed in the DSL (e.g. session-aware) are NOT ported.

#[cfg(feature = "least-active-q")]
pub(crate) mod least_active;
#[cfg(feature = "least-bs-q")]
pub(crate) mod least_bs;
#[cfg(feature = "least-token-load-q")]
pub(crate) mod least_token_load;
#[cfg(feature = "least-waiting-q")]
pub(crate) mod least_waiting;
#[cfg(feature = "most-hit-q")]
pub(crate) mod most_hit;
#[cfg(feature = "most-hit-load-q")]
pub(crate) mod most_hit_load;
#[cfg(feature = "most-hit-load-active-q")]
pub(crate) mod most_hit_load_active;

#[cfg(feature = "least-active-q")]
pub(crate) use least_active::LeastActiveQ;
#[cfg(feature = "least-bs-q")]
pub(crate) use least_bs::LeastBsQ;
#[cfg(feature = "least-token-load-q")]
pub(crate) use least_token_load::LeastTokenLoadQ;
#[cfg(feature = "least-waiting-q")]
pub(crate) use least_waiting::LeastWaitingQ;
#[cfg(feature = "most-hit-q")]
pub(crate) use most_hit::MostHitQ;
#[cfg(feature = "most-hit-load-q")]
pub(crate) use most_hit_load::MostHitLoadQ;
#[cfg(feature = "most-hit-load-active-q")]
pub(crate) use most_hit_load_active::MostHitLoadActiveQ;
