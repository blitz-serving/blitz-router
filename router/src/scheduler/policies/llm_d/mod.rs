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

pub(crate) mod least_active;
pub(crate) mod least_bs;
pub(crate) mod least_token_load;
pub(crate) mod least_waiting;
pub(crate) mod most_hit;
pub(crate) mod most_hit_load;
pub(crate) mod most_hit_load_active;

#[allow(unused_imports)]
pub(crate) use least_active::LeastActiveQ;
#[allow(unused_imports)]
pub(crate) use least_bs::LeastBsQ;
#[allow(unused_imports)]
pub(crate) use least_token_load::LeastTokenLoadQ;
#[allow(unused_imports)]
pub(crate) use least_waiting::LeastWaitingQ;
#[allow(unused_imports)]
pub(crate) use most_hit::MostHitQ;
#[allow(unused_imports)]
pub(crate) use most_hit_load::MostHitLoadQ;
#[allow(unused_imports)]
pub(crate) use most_hit_load_active::MostHitLoadActiveQ;
