// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// Unified EngineClient trait abstracting over different backend transports
// (HTTP+SSE via VllmClient, ZMQ via ZmqEngineClient).
//
// This module defines a common interface that the event loop can program
// against, regardless of the underlying engine communication protocol.

use crate::kvcache::BackendBlockHash;
use crate::validation::ValidGenerateRequest;
use nohash_hasher::IntMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Unified error type for all engine client implementations.
#[derive(Debug, Error)]
pub enum EngineClientError {
    #[error("HTTP request failed: {0}")]
    Http(String),

    #[error("SSE stream error: {0}")]
    Sse(String),

    #[error("JSON serialization/deserialization failed: {0}")]
    Json(String),

    #[error("ZMQ socket error: {0}")]
    Zmq(String),

    #[error("Msgpack encode/decode error: {0}")]
    Msgpack(String),

    #[error("Connection not established")]
    NotConnected,

    #[error("Engine API error: {0}")]
    ApiError(String),

    #[error("Stream ended unexpectedly")]
    StreamEnded,
}

// ---------------------------------------------------------------------------
// Unified output types
// ---------------------------------------------------------------------------

/// Per-request status from a single engine step.
/// Unified representation that both HTTP+SSE (VllmRequestStatus) and ZMQ
/// (EngineCoreOutput) map into.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestStepOutput {
    /// Unique request identifier.
    pub request_id: u64,
    /// Newly generated token IDs in this step.
    pub new_token_ids: Vec<u32>,
    /// Engine-reported state string (e.g., "PREFILL", "DECODE", "RUNNING").
    pub state: String,
    /// Whether the request has finished generating.
    pub is_finished: bool,
    /// Number of prompt tokens that hit the KV cache (prefix cache hit).
    pub hit_token_cnt: u64,
}

/// Aggregated output from a single engine step.
/// This is the unified type that replaces both `VllmMetric` (HTTP+SSE) and
/// `EngineCoreOutputs` (ZMQ) for the event loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStepOutput {
    /// Number of prefill tokens processed in this step.
    pub prefill_tokens: usize,
    /// Remaining prefill token budget after this step.
    pub prefill_token_budget: usize,
    /// Step latency in milliseconds.
    pub latency: u64,
    /// Per-request outputs from this step.
    pub outputs: Vec<RequestStepOutput>,
    /// New block hashes discovered in this step (for cache tracking).
    pub new_block_hashes: Vec<BackendBlockHash>,
    /// Block hashes that were evicted from the engine's cache.
    pub evicted_block_hashes: Vec<BackendBlockHash>,
    /// Block IDs that were evicted from the engine's cache.
    pub evicted_block_ids: Vec<u64>,
    /// Mapping of request_id -> currently used block IDs.
    pub cur_used_block_ids: IntMap<u64, Vec<u64>>,
    /// Mapping of request_id -> new block hash IDs.
    pub new_block_hashes_ids: IntMap<u64, Vec<u64>>,
    /// Optional execution log from the engine's scheduler.
    pub op_exec_log: Option<String>,
    /// Request IDs that were preempted by the engine scheduler.
    pub preempted_ids: Vec<u64>,
    /// Request IDs that were aborted by the engine.
    pub aborted_requests: Vec<u64>,
}

// ---------------------------------------------------------------------------
// EngineClient trait
// ---------------------------------------------------------------------------

/// Unified trait for communicating with an LLM inference engine.
///
/// Implementations exist for:
/// - `VllmClient` (HTTP+SSE, feature `vllm-backend`): sends requests via HTTP
///   POST and receives per-step metrics via SSE stream.
/// - `ZmqEngineClient` (ZMQ, feature `zmq-backend`): sends requests via ZMQ
///   DEALER socket and receives outputs via ZMQ PULL socket.
#[async_trait::async_trait]
pub trait EngineClient: Send {
    /// Send an inference request to the engine.
    ///
    /// The implementation is responsible for translating the
    /// `ValidGenerateRequest` into the engine's wire format (HTTP JSON body,
    /// ZMQ msgpack, etc.) and dispatching it.
    ///
    /// Takes `&mut self` because some backends (ZMQ) require mutable access
    /// to their sockets. The event loop owns a single client instance per
    /// replica, so exclusive access is guaranteed.
    async fn add_request(
        &mut self,
        id: u64,
        request: &ValidGenerateRequest,
    ) -> Result<(), EngineClientError>;

    /// Receive the next engine step output (metrics + per-request status).
    ///
    /// This replaces SSE stream polling (HTTP) and `recv_outputs` (ZMQ).
    /// Blocks asynchronously until the engine produces a step output.
    async fn recv_step(&mut self) -> Result<EngineStepOutput, EngineClientError>;

    /// Abort a running request on the engine.
    async fn abort_request(&mut self, id: u64) -> Result<(), EngineClientError>;

    /// Take the error notification receiver.
    ///
    /// Returns an unbounded receiver that yields request IDs for which the
    /// backend reported an error (e.g., HTTP POST failure). Can only be
    /// called once per client instance.
    fn get_error_rx(&mut self) -> mpsc::UnboundedReceiver<u64>;
}

// ===========================================================================
// VllmClient implementation (HTTP+SSE backend)
// ===========================================================================

#[cfg(feature = "vllm-backend")]
mod vllm_impl {
    use super::*;
    use crate::vllmlet::{VllmClient, VllmMetric};
    use eventsource_client as es;
    use futures::StreamExt;

    /// Internal state for the SSE stream, lazily initialized on the first
    /// call to `recv_step`.
    struct SseState {
        stream: Box<dyn futures::Stream<Item = Result<es::SSE, es::Error>> + Unpin + Send>,
        connected: bool,
    }

    /// Wrapper around `VllmClient` that implements `EngineClient`.
    ///
    /// The SSE connection is established lazily on the first `recv_step` call,
    /// so construction is cheap and non-blocking.
    pub struct VllmEngineClient {
        inner: VllmClient,
        sse_state: Option<SseState>,
    }

    impl VllmEngineClient {
        /// Create a new `VllmEngineClient` wrapping an existing `VllmClient`.
        pub fn new(client: VllmClient) -> Self {
            Self {
                inner: client,
                sse_state: None,
            }
        }

        /// Ensure the SSE stream is initialized and connected.
        async fn ensure_sse_connected(&mut self) -> Result<(), EngineClientError> {
            if self.sse_state.is_some() {
                return Ok(());
            }

            let sse_client = self
                .inner
                .init_sse_client()
                .await
                .map_err(|e| EngineClientError::Sse(e.to_string()))?;

            let mut stream = sse_client.stream();

            // Wait for the initial SSE connection event
            match stream.next().await {
                Some(Ok(es::SSE::Connected(conn))) => {
                    let status = conn.response().status();
                    if status != 200 {
                        return Err(EngineClientError::Sse(format!(
                            "SSE connection returned status {}",
                            status
                        )));
                    }
                }
                Some(Ok(_)) => {
                    // Unexpected first event, but tolerate it
                }
                Some(Err(e)) => {
                    return Err(EngineClientError::Sse(format!(
                        "SSE connection error: {}",
                        e
                    )));
                }
                None => {
                    return Err(EngineClientError::StreamEnded);
                }
            }

            self.sse_state = Some(SseState {
                stream: Box::new(stream),
                connected: true,
            });

            Ok(())
        }
    }

    /// Convert `VllmMetric` (from SSE JSON) into the unified `EngineStepOutput`.
    fn vllm_metric_to_step_output(m: VllmMetric) -> EngineStepOutput {
        let outputs = m
            .outputs
            .into_iter()
            .map(|s| RequestStepOutput {
                request_id: s.request_id,
                new_token_ids: s.new_token_ids,
                state: s.state,
                is_finished: s.is_finished,
                hit_token_cnt: s.hit_token_cnt,
            })
            .collect();

        EngineStepOutput {
            prefill_tokens: m.prefill_tokens,
            prefill_token_budget: m.prefill_token_budget,
            latency: m.latency,
            outputs,
            new_block_hashes: m.new_block_hashes,
            evicted_block_hashes: m.evicted_block_hashes,
            evicted_block_ids: m.evicted_block_ids,
            cur_used_block_ids: m.cur_used_block_ids,
            new_block_hashes_ids: m.new_block_hashes_ids,
            op_exec_log: m.op_exec_log,
            preempted_ids: m.preempted_ids,
            aborted_requests: m.aborted_requests,
        }
    }

    #[async_trait::async_trait]
    impl EngineClient for VllmEngineClient {
        async fn add_request(
            &mut self,
            id: u64,
            request: &ValidGenerateRequest,
        ) -> Result<(), EngineClientError> {
            // The existing VllmClient::add_request spawns a tokio task and
            // returns a JoinHandle. We fire-and-forget here since the event
            // loop tracks completion via SSE, not via the HTTP response.
            let _handle = self.inner.add_request(id, request).await;
            Ok(())
        }

        async fn recv_step(&mut self) -> Result<EngineStepOutput, EngineClientError> {
            self.ensure_sse_connected().await?;

            let sse_state = self.sse_state.as_mut().unwrap();

            loop {
                match sse_state.stream.next().await {
                    Some(Ok(es::SSE::Event(e))) => {
                        let m: VllmMetric = serde_json::from_str(&e.data).map_err(|err| {
                            EngineClientError::Json(format!(
                                "Failed to parse SSE event data: {} (raw: {:?})",
                                err, e.data
                            ))
                        })?;
                        return Ok(vllm_metric_to_step_output(m));
                    }
                    Some(Ok(es::SSE::Comment(_))) => {
                        // SSE comment/keepalive, skip
                        continue;
                    }
                    Some(Ok(es::SSE::Connected(_))) => {
                        // Reconnection event, skip
                        tracing::debug!("SSE reconnected");
                        continue;
                    }
                    Some(Err(e)) => {
                        return Err(EngineClientError::Sse(format!("SSE stream error: {}", e)));
                    }
                    None => {
                        return Err(EngineClientError::StreamEnded);
                    }
                }
            }
        }

        async fn abort_request(&mut self, _id: u64) -> Result<(), EngineClientError> {
            // HTTP+SSE backend does not support explicit abort via a separate
            // endpoint. The engine handles cleanup when the HTTP connection is
            // dropped. This is a no-op.
            tracing::debug!("abort_request is a no-op for VllmClient (HTTP+SSE backend)");
            Ok(())
        }

        fn get_error_rx(&mut self) -> mpsc::UnboundedReceiver<u64> {
            self.inner.get_error_rx()
        }
    }
}

#[cfg(feature = "vllm-backend")]
pub use vllm_impl::VllmEngineClient;

// ===========================================================================
// ZmqEngineClient implementation (ZMQ backend)
// ===========================================================================

#[cfg(feature = "zmq-backend")]
mod zmq_impl {
    use super::*;
    use crate::zmq_engine::{
        EngineCoreRequest, SamplingParams, ZmqEngineClient, ZmqEngineError,
    };

    impl From<ZmqEngineError> for EngineClientError {
        fn from(e: ZmqEngineError) -> Self {
            match e {
                ZmqEngineError::Zmq(e) => EngineClientError::Zmq(e.to_string()),
                ZmqEngineError::MsgpackEncode(e) => EngineClientError::Msgpack(e.to_string()),
                ZmqEngineError::MsgpackDecode(e) => EngineClientError::Msgpack(e.to_string()),
                ZmqEngineError::NotConnected => EngineClientError::NotConnected,
            }
        }
    }

    /// Wrapper around `ZmqEngineClient` that implements `EngineClient`.
    pub struct ZmqEngineClientAdapter {
        inner: ZmqEngineClient,
        /// Error notification channel (mirrors VllmClient's pattern).
        error_tx: mpsc::UnboundedSender<u64>,
        error_rx: Option<mpsc::UnboundedReceiver<u64>>,
    }

    impl ZmqEngineClientAdapter {
        /// Create a new adapter wrapping a `ZmqEngineClient`.
        ///
        /// The ZMQ client should already be connected before wrapping.
        pub fn new(client: ZmqEngineClient) -> Self {
            let (tx, rx) = mpsc::unbounded_channel();
            Self {
                inner: client,
                error_tx: tx,
                error_rx: Some(rx),
            }
        }
    }

    /// Convert `ValidGenerateRequest` into `EngineCoreRequest` for the ZMQ
    /// wire format.
    fn valid_request_to_engine_core(
        id: u64,
        request: &ValidGenerateRequest,
    ) -> EngineCoreRequest {
        EngineCoreRequest {
            request_id: id.to_string(),
            prompt_token_ids: request.input_tokens.clone(),
            mm_inputs: vec![],
            mm_hashes: vec![],
            mm_placeholders: vec![],
            sampling_params: SamplingParams {
                max_tokens: request.stopping_parameters.max_new_tokens,
                temperature: 0.0,
                min_tokens: request.stopping_parameters.max_new_tokens,
                ..Default::default()
            },
            eos_token_id: None,
            arrival_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            lora_request: None,
            prompt_adapter_request: None,
        }
    }

    #[async_trait::async_trait]
    impl EngineClient for ZmqEngineClientAdapter {
        async fn add_request(
            &mut self,
            id: u64,
            request: &ValidGenerateRequest,
        ) -> Result<(), EngineClientError> {
            let core_request = valid_request_to_engine_core(id, request);
            self.inner.add_request(core_request).await?;
            Ok(())
        }

        async fn recv_step(&mut self) -> Result<EngineStepOutput, EngineClientError> {
            let outputs = self.inner.recv_outputs().await?;

            // Convert EngineCoreOutputs to EngineStepOutput
            let step_outputs: Vec<RequestStepOutput> = outputs
                .outputs
                .into_iter()
                .map(|o| {
                    // Parse request_id from string back to u64
                    let request_id = o.request_id.parse::<u64>().unwrap_or_else(|_| {
                        tracing::warn!(
                            "Failed to parse request_id '{}' as u64, using 0",
                            o.request_id
                        );
                        0
                    });

                    let is_finished = o.finish_reason.is_some();
                    let state = match &o.state {
                        Some(s) => s.clone(),
                        None => {
                            if is_finished {
                                "DECODE".to_string()
                            } else {
                                "DECODE".to_string()
                            }
                        }
                    };

                    RequestStepOutput {
                        request_id,
                        new_token_ids: o.new_token_ids,
                        state,
                        is_finished,
                        hit_token_cnt: o.num_cached_tokens as u64,
                    }
                })
                .collect();

            // Convert latency from seconds (f64) to milliseconds (u64)
            let latency_ms = outputs
                .latency
                .map(|l| (l * 1000.0) as u64)
                .unwrap_or(0);

            Ok(EngineStepOutput {
                prefill_tokens: outputs.prefill_tokens.unwrap_or(0) as usize,
                prefill_token_budget: outputs.prefill_token_budget.unwrap_or(0) as usize,
                latency: latency_ms,
                outputs: step_outputs,
                // ZMQ protocol does not currently carry block hash / eviction
                // info. These fields are left empty; cache-aware routing is
                // handled differently in ZMQ mode.
                new_block_hashes: Vec::new(),
                evicted_block_hashes: Vec::new(),
                evicted_block_ids: Vec::new(),
                cur_used_block_ids: IntMap::default(),
                new_block_hashes_ids: IntMap::default(),
                op_exec_log: None,
                preempted_ids: Vec::new(),
                aborted_requests: Vec::new(),
            })
        }

        async fn abort_request(&mut self, id: u64) -> Result<(), EngineClientError> {
            let id_str = id.to_string();
            self.inner.abort_request(&id_str).await?;
            Ok(())
        }

        fn get_error_rx(&mut self) -> mpsc::UnboundedReceiver<u64> {
            self.error_rx
                .take()
                .expect("get_error_rx called more than once on ZmqEngineClientAdapter")
        }
    }
}

#[cfg(feature = "zmq-backend")]
pub use zmq_impl::ZmqEngineClientAdapter;
