// Middle layer — request scheduling and admission.
//
// Houses the policy plug-ins, the per-replica state metrics, the
// prefix/KV cache trackers, the work-queue assigner, and (when the
// `simulator` feature is on) the latency simulator. Items here are
// shared across the front gateway and the back engine driver via the
// crate-root `pub use scheduler::*;` blanket in `lib.rs`.

pub(crate) mod infer;
pub(crate) mod kvcache;
pub(crate) mod policies;
pub(crate) mod queue;
pub(crate) mod state; // formerly metrics.rs (renamed to dodge crates.io `metrics` collision)
pub(crate) mod statistic;

#[cfg(feature = "simulator")]
pub mod simulator; // pub: `simulator::query` is the future service-sidecar API;
                   // `on_admit` / `on_sse` are pub(crate) hooks called by
                   // `policies/policy_runner.rs` and `engine/colocation.rs`.

#[allow(unused_imports)]
pub(crate) use infer::Infer;
#[allow(unused_imports)]
pub(crate) use kvcache::{
    BackendBlockHash, BlockHash, BlockHashState, DEFAULT_BLOCK_HASH, PrefixBlockHash,
};
#[allow(unused_imports)]
pub(crate) use queue::TaskAssigner;
#[allow(unused_imports)]
pub(crate) use state::{
    BAILIAN_ALPHA, BAILIAN_BETA, BAILIAN_GAMMA, LMetric, LMetricDec, LMetricInc,
    LOAD_AWARE_QUEUE_T, MOST_HIT_LOAD_ACTIVE_W_HIT, MOST_HIT_LOAD_ACTIVE_W_KV,
    MOST_HIT_LOAD_ACTIVE_W_LOAD, MOST_HIT_LOAD_W_HIT, MOST_HIT_LOAD_W_LOAD, ScheduleContext,
    WAITINGT_PREFILL_TOKEN_BOUND,
};
#[cfg(feature = "bailian-impl-q")]
pub use state::init_bailian_params;
#[cfg(feature = "polyserve-q")]
pub use state::init_polyserve_params;
