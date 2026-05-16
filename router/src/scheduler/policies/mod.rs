// Scheduling policies for replica selection.
//
// Every policy is emitted by the `policy!` proc macro from `policy-dsl/`,
// which lowers a spec-form DSL expression (`docs/dsl/policies.md` §2) into
// an `impl Policy for X` block. The generic `PolicyRunner<P: Policy>` in
// `policy_runner.rs` drives any such P with the queue management /
// commit-buffer / batch-dispatch boilerplate.
//
// Module layout — one module per upstream baseline system, plus `simple`
// for trivial policies with no upstream origin:
//
//   simple.rs   — random / round-robin / least-wait-token / bounded-most-hit
//   vllm.rs     — vLLM 4·waiting + bs                                   (1)
//   bailian.rs  — Bailian                                               (1)
//   aibrix.rs   — AIBrix                                                (1)
//   dynamo.rs   — AI-Dynamo Decode-node + Prefill-node logits           (2)
//   lmetric.rs  — our system                                            (1)
//   preble/     — Preble + cost-model + sliding-window histogram        (1)
//   llm_d/      — llm-d single-scorer ablations + multi-scorer combos   (7)
//
// Total: 18 policies.
//
// Phase 4 cleanup landed: legacy `QueuePlusPlus`, `AssignScore`,
// `NaiiveLattice`, `Deterministic`/`StochasticPolicy`, `ScheduleStep`,
// `SamplerFn`, `QueueRunner`, `step_deterministic`, `step_stochastic`,
// `apply_schedule_decision`, `EmptyContext`, `RRContext`, `NumHitKvBlock`
// have been deleted from this module. What remains is the bare minimum:
// the `Entry` queue type, the `QueuePro` public interface, and the
// per-feature `TaskAssigner` aliases.

pub(crate) mod aibrix;
pub(crate) mod bailian;
pub(crate) mod dsl_runtime;
pub(crate) mod dynamo;
#[cfg(feature = "least-ttft-q")]
pub(crate) mod least_ttft;
pub(crate) mod llm_d;
pub(crate) mod lmetric;
pub(crate) mod policy_runner;
pub(crate) mod policy_trait;
#[cfg(feature = "polyserve-q")]
pub(crate) mod polyserve;
#[cfg(feature = "polyserve2-q")]
pub(crate) mod polyserve2;
pub(crate) mod preble;
#[cfg(any(feature = "polyserve-q", feature = "polyserve2-q"))]
mod recent_exclusion;
pub(crate) mod simple;
pub(crate) mod vllm;

// Re-export concrete policy structs (each emitted by `policy!`).
// `#[allow(unused_imports)]` because exactly one is referenced per
// build via the cargo-feature-gated `TaskAssigner` alias below.
#[allow(unused_imports)]
pub(crate) use aibrix::AibrixQ;
#[allow(unused_imports)]
pub(crate) use bailian::BailianImplQ;
#[allow(unused_imports)]
pub(crate) use dynamo::{DynamoPoQ, DynamoQ};
#[cfg(feature = "least-ttft-q")]
#[allow(unused_imports)]
pub(crate) use least_ttft::LeastTtftQ;
#[allow(unused_imports)]
pub(crate) use llm_d::{
    LeastActiveQ, LeastBsQ, LeastTokenLoadQ, LeastWaitingQ, MostHitLoadActiveQ, MostHitLoadQ,
    MostHitQ,
};
#[allow(unused_imports)]
pub(crate) use lmetric::LmetricQ;
#[cfg(feature = "polyserve-q")]
#[allow(unused_imports)]
pub(crate) use polyserve::PolyserveQ;
#[cfg(feature = "polyserve2-q")]
#[allow(unused_imports)]
pub(crate) use polyserve2::Polyserve2Q;
#[allow(unused_imports)]
pub(crate) use preble::PrebleQ;
#[allow(unused_imports)]
pub(crate) use simple::{JBoundMostHitQ2, JLeastWaitTokenQ, RandomQ, RoundRobinQ};
#[allow(unused_imports)]
pub(crate) use vllm::JShortestWeightQ;

use super::infer::{InferError, InferStreamResponse};
use super::kvcache::BlockHashState;
use crate::gateway::validation::ValidGenerateRequest;

use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::Span;

// ---------------------------------------------------------------------------
// Common type aliases
// ---------------------------------------------------------------------------

pub(crate) type NextRequest = (u64, Entry);

// ---------------------------------------------------------------------------
// Entry -- a single queued request
// ---------------------------------------------------------------------------

/// Queue entry
#[derive(Debug)]
#[allow(dead_code)] // span/temp_span carried for tracing context, accessed via Debug
pub(crate) struct Entry {
    /// Request
    pub request: ValidGenerateRequest,
    /// BlockHash
    pub block_hash_state: BlockHashState,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: mpsc::UnboundedSender<Result<InferStreamResponse, InferError>>,
    /// Span that will live as long as entry
    pub span: Span,
    /// Temporary span used as a guard when logging inference, wait times...
    pub temp_span: Option<Span>,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
    /// Number of current generated tokens
    pub generated_token_cnt: usize,
    /// Instant when generated previous token, set after prefilling
    pub prev_token_time: Option<Instant>,
    /// Averaged TPOT, set after first decoding
    pub time_of_per_token: Option<Duration>,
    /// TBT of the first pure-decode step this request appears in.
    pub first_tbt_time: Option<Duration>,
    /// Max TBT
    pub max_time_between_tokens: Duration,
}

impl Entry {
    pub fn append_state(&mut self, new_tokens: &Vec<u32>, tbt: &Duration) {
        self.append_tbt_to_tpot(tbt);
        self.block_hash_state.append_tokens(new_tokens);
    }

    fn append_tbt_to_tpot(&mut self, tbt: &Duration) {
        if let Some(tpot) = self.time_of_per_token {
            let s = self.generated_token_cnt;
            let old_tpot = tpot.as_secs_f32();
            let new_tpot = (old_tpot * (s as f32) + tbt.as_secs_f32()) / {
                self.generated_token_cnt += 1;
                self.generated_token_cnt as f32
            };
            self.time_of_per_token.replace(Duration::from_secs_f32(new_tpot));
        } else {
            self.generated_token_cnt += 1;
            self.time_of_per_token = Some(tbt.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// QueuePro -- public interface for a scheduling queue (used by infer.rs,
// colocation.rs); implemented by PolicyRunner<P>.
// ---------------------------------------------------------------------------

pub(crate) trait QueuePro {
    fn append(&self, entry: Entry);
    async fn next_request(&self, replica_id: usize) -> Option<NextRequest>;
}

// ---------------------------------------------------------------------------
// Feature-gated TaskAssigner alias.
// Each policy is wired through PolicyRunner<P: Policy>.
// `join-shortest-weight-q` is the catch-all default when no policy feature
// is selected.
// ---------------------------------------------------------------------------

use policy_runner::PolicyRunner;

#[cfg(feature = "bailian-impl-q")]
pub(crate) type TaskAssigner = PolicyRunner<BailianImplQ>;
#[cfg(feature = "bounded-most-hit-q")]
pub(crate) type TaskAssigner = PolicyRunner<JBoundMostHitQ2>;
#[cfg(feature = "least-wait-token-q")]
pub(crate) type TaskAssigner = PolicyRunner<JLeastWaitTokenQ>;
#[cfg(feature = "aibrix-q")]
pub(crate) type TaskAssigner = PolicyRunner<AibrixQ>;
#[cfg(feature = "dynamo-q")]
pub(crate) type TaskAssigner = PolicyRunner<DynamoQ>;
#[cfg(feature = "dynamo-po-q")]
pub(crate) type TaskAssigner = PolicyRunner<DynamoPoQ>;
#[cfg(feature = "lmetric-q")]
pub(crate) type TaskAssigner = PolicyRunner<LmetricQ>;
#[cfg(feature = "preble-q")]
pub(crate) type TaskAssigner = PolicyRunner<PrebleQ>;
#[cfg(feature = "most-hit-q")]
pub(crate) type TaskAssigner = PolicyRunner<MostHitQ>;
#[cfg(feature = "most-hit-load-q")]
pub(crate) type TaskAssigner = PolicyRunner<MostHitLoadQ>;
#[cfg(feature = "most-hit-load-active-q")]
pub(crate) type TaskAssigner = PolicyRunner<MostHitLoadActiveQ>;
#[cfg(feature = "least-waiting-q")]
pub(crate) type TaskAssigner = PolicyRunner<LeastWaitingQ>;
#[cfg(feature = "least-bs-q")]
pub(crate) type TaskAssigner = PolicyRunner<LeastBsQ>;
#[cfg(feature = "least-active-q")]
pub(crate) type TaskAssigner = PolicyRunner<LeastActiveQ>;
#[cfg(feature = "least-token-load-q")]
pub(crate) type TaskAssigner = PolicyRunner<LeastTokenLoadQ>;
#[cfg(feature = "round-robin-q")]
pub(crate) type TaskAssigner = PolicyRunner<RoundRobinQ>;
#[cfg(feature = "random-q")]
pub(crate) type TaskAssigner = PolicyRunner<RandomQ>;
#[cfg(feature = "least-ttft-q")]
pub(crate) type TaskAssigner = PolicyRunner<LeastTtftQ>;
#[cfg(feature = "polyserve-q")]
pub(crate) type TaskAssigner = PolicyRunner<PolyserveQ>;
#[cfg(feature = "polyserve2-q")]
pub(crate) type TaskAssigner = PolicyRunner<Polyserve2Q>;
// `join-shortest-weight-q` is also the catch-all default — vLLM's
// `4·sctx.waiting + sctx.bs` formula.
#[cfg(any(
    feature = "join-shortest-weight-q",
    not(any(
        feature = "bailian-impl-q",
        feature = "bounded-most-hit-q",
        feature = "least-wait-token-q",
        feature = "aibrix-q",
        feature = "dynamo-q",
        feature = "dynamo-po-q",
        feature = "lmetric-q",
        feature = "preble-q",
        feature = "most-hit-q",
        feature = "most-hit-load-q",
        feature = "most-hit-load-active-q",
        feature = "least-waiting-q",
        feature = "least-bs-q",
        feature = "least-active-q",
        feature = "least-token-load-q",
        feature = "round-robin-q",
        feature = "random-q",
        feature = "least-ttft-q",
        feature = "polyserve-q",
        feature = "polyserve2-q",
    ))
))]
pub(crate) type TaskAssigner = PolicyRunner<JShortestWeightQ>;
