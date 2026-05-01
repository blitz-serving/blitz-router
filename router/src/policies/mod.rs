// Scheduling policies for replica selection.
//
// Every policy is emitted by the `policy!` proc macro from `policy-dsl/`,
// which lowers a paper-form DSL expression (`docs/dsl-schema.md` §8) into
// an `impl Policy for X` block. The generic `PolicyRunner<P: Policy>` in
// `policy_runner.rs` drives any such P with the queue management /
// commit-buffer / batch-dispatch boilerplate.
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
pub(crate) mod lmetric;
pub(crate) mod most_hit;
pub(crate) mod policy_runner;
pub(crate) mod policy_trait;
pub(crate) mod preble;
pub(crate) mod simple;

// Re-export concrete policy structs (each emitted by `policy!`).
// `#[allow(unused_imports)]` because exactly one is referenced per
// build via the cargo-feature-gated `TaskAssigner` alias below.
#[allow(unused_imports)]
pub(crate) use aibrix::AibrixQ;
#[allow(unused_imports)]
pub(crate) use bailian::BailianImplQ;
#[allow(unused_imports)]
pub(crate) use dynamo::{DynamoPoQ, DynamoQ};
#[allow(unused_imports)]
pub(crate) use lmetric::LmetricQ;
#[allow(unused_imports)]
pub(crate) use most_hit::MostHitQ;
#[allow(unused_imports)]
pub(crate) use preble::PrebleQ;
#[allow(unused_imports)]
pub(crate) use simple::{
    JBoundMostHitQ2, JLeastWaitTokenQ, JShortestQ, JShortestQWeight, RandomQ, RoundRobinQ,
};

use crate::kvcache::BlockHashState;
use crate::infer::{InferError, InferStreamResponse};
use crate::validation::ValidGenerateRequest;

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
#[cfg(feature = "join-shortest-q-weight")]
pub(crate) type TaskAssigner = PolicyRunner<JShortestQWeight>;
#[cfg(feature = "round-robin-q")]
pub(crate) type TaskAssigner = PolicyRunner<RoundRobinQ>;
#[cfg(feature = "random-q")]
pub(crate) type TaskAssigner = PolicyRunner<RandomQ>;
// `join-shortest-q` (default catch-all) → JShortestQ (vLLM 4·waiting + bs).
#[cfg(all(feature = "join-shortest-q", not(any(
    feature = "bailian-impl-q",
    feature = "bounded-most-hit-q",
    feature = "least-wait-token-q",
    feature = "aibrix-q",
    feature = "dynamo-q",
    feature = "join-shortest-q-weight",
    feature = "round-robin-q",
    feature = "random-q",
    feature = "most-hit-q",
))))]
pub(crate) type TaskAssigner = PolicyRunner<JShortestQ>;
