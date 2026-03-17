// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// ZMQ Engine Client for direct communication with yaullm's EngineCore.
//
// This module bypasses the Python master process and speaks directly to
// EngineCore via ZMQ sockets using msgpack-encoded messages.
//
// Protocol:
//   Input  (Router -> Engine): ZMQ DEALER socket, msgpack-encoded requests
//   Output (Engine -> Router): ZMQ PULL socket, msgpack-encoded outputs
//
// Socket addresses are typically:
//   ipc:///tmp/vllm-engine-{port}-input
//   ipc:///tmp/vllm-engine-{port}-output

use serde::{Deserialize, Serialize};
use thiserror::Error;
use zeromq::{DealerSocket, PullSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ZmqEngineError {
    #[error("ZMQ socket error: {0}")]
    Zmq(#[from] zeromq::ZmqError),

    #[error("Msgpack encode error: {0}")]
    MsgpackEncode(#[from] rmp_serde::encode::Error),

    #[error("Msgpack decode error: {0}")]
    MsgpackDecode(#[from] rmp_serde::decode::Error),

    #[error("Connection not established: call connect() first")]
    NotConnected,
}

// ---------------------------------------------------------------------------
// EngineCore request/response types (msgpack wire format)
// ---------------------------------------------------------------------------

/// Request type discriminator, sent as the first byte of the msgpack payload.
/// Must match yaullm's `EngineCoreRequestType` enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum EngineCoreRequestType {
    AddRequest = 0,
    AbortRequest = 1,
    Profile = 2,
}

/// Sampling parameters sent alongside an inference request.
/// Fields mirror yaullm's `SamplingParams` (only the commonly used subset).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature. 0.0 means greedy.
    #[serde(default)]
    pub temperature: f64,
    /// Top-p (nucleus) sampling threshold.
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    /// Top-k sampling. -1 means disabled.
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    /// Repetition penalty. 1.0 means no penalty.
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f64,
    /// Minimum number of tokens to generate before allowing EOS.
    #[serde(default)]
    pub min_tokens: u32,
    /// Whether to skip special tokens during decoding.
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
    /// Whether to include stop strings in the output.
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    /// Stop token IDs.
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
    /// Whether to detokenize output on the engine side.
    /// When true, `EngineCoreOutput.new_text` will be populated.
    #[serde(default = "default_true")]
    pub detokenize: bool,
}

fn default_top_p() -> f64 {
    1.0
}
fn default_top_k() -> i32 {
    -1
}
fn default_repetition_penalty() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            top_k: -1,
            repetition_penalty: 1.0,
            min_tokens: 0,
            skip_special_tokens: true,
            include_stop_str_in_output: false,
            stop_token_ids: Vec::new(),
            detokenize: true,
        }
    }
}

/// A request to add a new inference job to the engine.
/// Serialized to msgpack and sent over the DEALER socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreRequest {
    /// Unique request identifier (string, not u64, to match yaullm's protocol).
    pub request_id: String,
    /// Pre-tokenized prompt token IDs (router handles tokenization).
    pub prompt_token_ids: Vec<u32>,
    /// Multimodal inputs (usually empty for text-only models).
    #[serde(default)]
    pub mm_inputs: Vec<serde_json::Value>,
    /// Multimodal content hashes (usually empty).
    #[serde(default)]
    pub mm_hashes: Vec<String>,
    /// Multimodal placeholder info (usually empty).
    #[serde(default)]
    pub mm_placeholders: Vec<serde_json::Value>,
    /// Sampling parameters for this request.
    pub sampling_params: SamplingParams,
    /// End-of-sequence token ID. `None` means use model default.
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    /// Request arrival time (Unix timestamp in seconds, f64).
    pub arrival_time: f64,
    /// LoRA adapter request (None for base model).
    #[serde(default)]
    pub lora_request: Option<serde_json::Value>,
    /// Prompt adapter request (None for default).
    #[serde(default)]
    pub prompt_adapter_request: Option<serde_json::Value>,
}

/// Finish reason for a completed request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Generation stopped because max_tokens was reached.
    Length,
    /// Generation stopped because an EOS token was generated.
    Stop,
    /// Generation was aborted.
    Abort,
}

/// Per-request output from a single engine step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreOutput {
    /// The request ID this output belongs to.
    pub request_id: String,
    /// Newly generated token IDs in this step.
    pub new_token_ids: Vec<u32>,
    /// Number of prompt tokens that were found in the KV cache (prefix cache hit).
    #[serde(default)]
    pub num_cached_tokens: u32,
    /// Current request state (e.g., "RUNNING", "WAITING").
    #[serde(default)]
    pub state: Option<String>,
    /// If the request is finished, the reason why.
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
    /// Detokenized text for the new tokens (populated when detokenize=true).
    #[serde(default)]
    pub new_text: Option<String>,
}

/// Scheduler statistics reported alongside outputs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Number of running requests.
    #[serde(default)]
    pub num_running_reqs: u32,
    /// Number of waiting requests.
    #[serde(default)]
    pub num_waiting_reqs: u32,
    /// GPU KV cache usage ratio (0.0 - 1.0).
    #[serde(default)]
    pub gpu_cache_usage: f64,
}

/// Aggregated outputs from a single engine step.
/// This is the top-level message received on the PULL socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreOutputs {
    /// Per-request outputs from this step.
    pub outputs: Vec<EngineCoreOutput>,
    /// Step latency in seconds (time spent in EngineCore.step()).
    #[serde(default)]
    pub latency: Option<f64>,
    /// Scheduler statistics for this step.
    #[serde(default)]
    pub scheduler_stats: Option<SchedulerStats>,
    /// Timestamp of when this output was produced (Unix seconds).
    #[serde(default)]
    pub timestamp: f64,
    /// Remaining prefill token budget after this step.
    #[serde(default)]
    pub prefill_token_budget: Option<u32>,
    /// Number of prefill tokens processed in this step.
    #[serde(default)]
    pub prefill_tokens: Option<u32>,
}

// ---------------------------------------------------------------------------
// ZMQ Engine Client
// ---------------------------------------------------------------------------

/// Client for communicating with yaullm's EngineCore via ZMQ sockets.
///
/// Uses a DEALER socket to send requests (ADD_REQUEST, ABORT_REQUEST) and
/// a PULL socket to receive step outputs (tokens, metrics, finish signals).
///
/// # Example
/// ```ignore
/// let mut client = ZmqEngineClient::new();
/// client.connect(
///     "ipc:///tmp/vllm-engine-8000-input",
///     "ipc:///tmp/vllm-engine-8000-output",
/// ).await?;
///
/// client.add_request(request).await?;
///
/// loop {
///     let outputs = client.recv_outputs().await?;
///     for output in &outputs.outputs {
///         if output.finish_reason.is_some() {
///             // Request completed
///         }
///     }
/// }
/// ```
pub struct ZmqEngineClient {
    /// DEALER socket for sending requests to EngineCore.
    input_socket: Option<DealerSocket>,
    /// PULL socket for receiving outputs from EngineCore.
    output_socket: Option<PullSocket>,
    /// Input socket address (for logging/diagnostics).
    input_addr: Option<String>,
    /// Output socket address (for logging/diagnostics).
    output_addr: Option<String>,
}

impl ZmqEngineClient {
    /// Create a new unconnected ZMQ engine client.
    pub fn new() -> Self {
        Self {
            input_socket: None,
            output_socket: None,
            input_addr: None,
            output_addr: None,
        }
    }

    /// Connect to EngineCore's ZMQ sockets.
    ///
    /// * `input_addr` - Address of the engine's input socket (ROUTER side).
    ///   Typically `ipc:///tmp/vllm-engine-{port}-input` or `tcp://host:port`.
    /// * `output_addr` - Address of the engine's output socket (PUSH side).
    ///   Typically `ipc:///tmp/vllm-engine-{port}-output` or `tcp://host:port`.
    pub async fn connect(
        &mut self,
        input_addr: &str,
        output_addr: &str,
    ) -> Result<(), ZmqEngineError> {
        // DEALER socket for sending requests.
        // The engine side binds a ROUTER socket; we connect a DEALER to it.
        let mut dealer = DealerSocket::new();
        dealer.connect(input_addr).await?;

        // PULL socket for receiving outputs.
        // The engine side binds a PUSH socket; we connect a PULL to it.
        let mut pull = PullSocket::new();
        pull.connect(output_addr).await?;

        self.input_socket = Some(dealer);
        self.output_socket = Some(pull);
        self.input_addr = Some(input_addr.to_string());
        self.output_addr = Some(output_addr.to_string());

        tracing::info!(
            "ZmqEngineClient connected: input={}, output={}",
            input_addr,
            output_addr
        );

        Ok(())
    }

    /// Check whether the client has been connected.
    pub fn is_connected(&self) -> bool {
        self.input_socket.is_some() && self.output_socket.is_some()
    }

    /// Send an ADD_REQUEST message to the engine.
    ///
    /// The wire format is a two-frame ZMQ message:
    ///   Frame 0: msgpack-encoded request type (u8 = 0)
    ///   Frame 1: msgpack-encoded EngineCoreRequest
    pub async fn add_request(
        &mut self,
        request: EngineCoreRequest,
    ) -> Result<(), ZmqEngineError> {
        let socket = self
            .input_socket
            .as_mut()
            .ok_or(ZmqEngineError::NotConnected)?;

        // Frame 0: request type as a single msgpack-encoded byte
        let type_bytes = rmp_serde::to_vec(&(EngineCoreRequestType::AddRequest as u8))?;
        // Frame 1: the full request payload
        let request_bytes = rmp_serde::to_vec(&request)?;

        // Construct a multi-frame ZMQ message
        let mut msg = ZmqMessage::from(type_bytes);
        msg.push_back(request_bytes.into());

        socket.send(msg).await?;

        tracing::debug!(
            "Sent ADD_REQUEST for request_id={}, prompt_len={}",
            request.request_id,
            request.prompt_token_ids.len()
        );

        Ok(())
    }

    /// Send an ABORT_REQUEST message to the engine.
    ///
    /// The wire format is a two-frame ZMQ message:
    ///   Frame 0: msgpack-encoded request type (u8 = 1)
    ///   Frame 1: msgpack-encoded request_id (string)
    pub async fn abort_request(
        &mut self,
        request_id: &str,
    ) -> Result<(), ZmqEngineError> {
        let socket = self
            .input_socket
            .as_mut()
            .ok_or(ZmqEngineError::NotConnected)?;

        let type_bytes = rmp_serde::to_vec(&(EngineCoreRequestType::AbortRequest as u8))?;
        let id_bytes = rmp_serde::to_vec(&request_id)?;

        let mut msg = ZmqMessage::from(type_bytes);
        msg.push_back(id_bytes.into());

        socket.send(msg).await?;

        tracing::debug!("Sent ABORT_REQUEST for request_id={}", request_id);

        Ok(())
    }

    /// Receive and deserialize the next `EngineCoreOutputs` from the engine.
    ///
    /// This call blocks (asynchronously) until the engine produces a step output.
    /// The message is a single-frame msgpack-encoded `EngineCoreOutputs`.
    pub async fn recv_outputs(&mut self) -> Result<EngineCoreOutputs, ZmqEngineError> {
        let socket = self
            .output_socket
            .as_mut()
            .ok_or(ZmqEngineError::NotConnected)?;

        let msg = socket.recv().await?;

        // The engine sends a single-frame message with msgpack-encoded outputs.
        // Get the first (and typically only) frame.
        let data = msg
            .into_vec()
            .into_iter()
            .next()
            .ok_or_else(|| {
                ZmqEngineError::MsgpackDecode(rmp_serde::decode::Error::Uncategorized(
                    "Empty ZMQ message received".to_string(),
                ))
            })?;

        let outputs: EngineCoreOutputs = rmp_serde::from_slice(&data)?;

        tracing::trace!(
            "Received EngineCoreOutputs: {} outputs, latency={:?}",
            outputs.outputs.len(),
            outputs.latency
        );

        Ok(outputs)
    }

    /// Return the input socket address, if connected.
    pub fn input_addr(&self) -> Option<&str> {
        self.input_addr.as_deref()
    }

    /// Return the output socket address, if connected.
    pub fn output_addr(&self) -> Option<&str> {
        self.output_addr.as_deref()
    }
}

impl Default for ZmqEngineClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper: construct socket addresses from a port number
// ---------------------------------------------------------------------------

/// Generate the default IPC socket addresses for a given engine port.
///
/// Returns `(input_addr, output_addr)` following yaullm's naming convention.
pub fn default_ipc_addrs(port: u16) -> (String, String) {
    let input = format!("ipc:///tmp/vllm-engine-{}-input", port);
    let output = format!("ipc:///tmp/vllm-engine-{}-output", port);
    (input, output)
}

/// Generate TCP socket addresses for a given host and base port.
///
/// The input socket uses `base_port` and the output socket uses `base_port + 1`.
pub fn default_tcp_addrs(host: &str, base_port: u16) -> (String, String) {
    let input = format!("tcp://{}:{}", host, base_port);
    let output = format!("tcp://{}:{}", host, base_port + 1);
    (input, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_ipc_addrs() {
        let (input, output) = default_ipc_addrs(8000);
        assert_eq!(input, "ipc:///tmp/vllm-engine-8000-input");
        assert_eq!(output, "ipc:///tmp/vllm-engine-8000-output");
    }

    #[test]
    fn test_default_tcp_addrs() {
        let (input, output) = default_tcp_addrs("127.0.0.1", 5570);
        assert_eq!(input, "tcp://127.0.0.1:5570");
        assert_eq!(output, "tcp://127.0.0.1:5571");
    }

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert_eq!(params.max_tokens, 256);
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.top_k, -1);
        assert_eq!(params.repetition_penalty, 1.0);
        assert!(params.skip_special_tokens);
        assert!(params.detokenize);
    }

    #[test]
    fn test_engine_core_request_serialization_roundtrip() {
        let request = EngineCoreRequest {
            request_id: "req-001".to_string(),
            prompt_token_ids: vec![1, 2, 3, 4, 5],
            mm_inputs: vec![],
            mm_hashes: vec![],
            mm_placeholders: vec![],
            sampling_params: SamplingParams {
                max_tokens: 128,
                temperature: 0.7,
                ..Default::default()
            },
            eos_token_id: Some(2),
            arrival_time: 1700000000.0,
            lora_request: None,
            prompt_adapter_request: None,
        };

        let encoded = rmp_serde::to_vec(&request).expect("Failed to encode request");
        let decoded: EngineCoreRequest =
            rmp_serde::from_slice(&encoded).expect("Failed to decode request");

        assert_eq!(decoded.request_id, "req-001");
        assert_eq!(decoded.prompt_token_ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(decoded.sampling_params.max_tokens, 128);
        assert_eq!(decoded.sampling_params.temperature, 0.7);
        assert_eq!(decoded.eos_token_id, Some(2));
    }

    #[test]
    fn test_engine_core_outputs_deserialization() {
        let outputs = EngineCoreOutputs {
            outputs: vec![
                EngineCoreOutput {
                    request_id: "req-001".to_string(),
                    new_token_ids: vec![42, 43],
                    num_cached_tokens: 10,
                    state: Some("RUNNING".to_string()),
                    finish_reason: None,
                    new_text: Some("hello".to_string()),
                },
                EngineCoreOutput {
                    request_id: "req-002".to_string(),
                    new_token_ids: vec![99],
                    num_cached_tokens: 0,
                    state: None,
                    finish_reason: Some(FinishReason::Stop),
                    new_text: Some(" world".to_string()),
                },
            ],
            latency: Some(0.045),
            scheduler_stats: Some(SchedulerStats {
                num_running_reqs: 2,
                num_waiting_reqs: 0,
                gpu_cache_usage: 0.35,
            }),
            timestamp: 1700000001.0,
            prefill_token_budget: Some(2048),
            prefill_tokens: Some(64),
        };

        let encoded = rmp_serde::to_vec(&outputs).expect("Failed to encode outputs");
        let decoded: EngineCoreOutputs =
            rmp_serde::from_slice(&encoded).expect("Failed to decode outputs");

        assert_eq!(decoded.outputs.len(), 2);
        assert_eq!(decoded.outputs[0].request_id, "req-001");
        assert_eq!(decoded.outputs[0].new_token_ids, vec![42, 43]);
        assert_eq!(decoded.outputs[0].num_cached_tokens, 10);
        assert_eq!(decoded.outputs[0].new_text, Some("hello".to_string()));
        assert!(decoded.outputs[0].finish_reason.is_none());

        assert_eq!(decoded.outputs[1].request_id, "req-002");
        assert_eq!(decoded.outputs[1].finish_reason, Some(FinishReason::Stop));

        assert_eq!(decoded.latency, Some(0.045));
        assert_eq!(decoded.prefill_token_budget, Some(2048));

        let stats = decoded.scheduler_stats.unwrap();
        assert_eq!(stats.num_running_reqs, 2);
        assert_eq!(stats.gpu_cache_usage, 0.35);
    }

    #[test]
    fn test_request_type_values() {
        assert_eq!(EngineCoreRequestType::AddRequest as u8, 0);
        assert_eq!(EngineCoreRequestType::AbortRequest as u8, 1);
        assert_eq!(EngineCoreRequestType::Profile as u8, 2);
    }

    #[test]
    fn test_finish_reason_serialization() {
        let length = FinishReason::Length;
        let stop = FinishReason::Stop;
        let abort = FinishReason::Abort;

        // Verify msgpack roundtrip
        for reason in [length, stop, abort] {
            let encoded = rmp_serde::to_vec(&reason).unwrap();
            let decoded: FinishReason = rmp_serde::from_slice(&encoded).unwrap();
            assert_eq!(decoded, reason);
        }
    }

    #[test]
    fn test_zmq_engine_client_not_connected() {
        let client = ZmqEngineClient::new();
        assert!(!client.is_connected());
        assert!(client.input_addr().is_none());
        assert!(client.output_addr().is_none());
    }
}
