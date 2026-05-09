// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// Hand-written replacement for the legacy `pb::generate::v2` types that
// previously came from `proto/generate.proto` via the `rust-proto/` crate.
//
// These types were never used as a wire protocol in this codebase — they
// only ever served as internal Rust data structures. The `proto/` and
// `rust-proto/` directories were deleted; their relevant message and enum
// shapes are reproduced here verbatim, preserving every field name,
// field type, enum variant, default value, and trait impl that the
// rest of the crate depends on. See
// `docs/architecture/workspace-and-features.md` for the workspace
// layout following the migration.
//
// NOTE: This module is intentionally a 1:1 preservation of the
// legacy prost-generated shapes; design changes (renames, dead-field
// removal, enum simplification) are deliberately deferred to a
// follow-up step.

// ==== Generation result ====

/// Final generation result for a finished request.
///
/// Mirrors `message GeneratedText { required string text = 1; required
/// uint32 generated_tokens = 2; required FinishReason finish_reason = 3;
/// optional uint64 seed = 4; }`.
///
/// Note: `finish_reason` is stored as `i32` to match the legacy proto
/// enum-field convention (callers parse it via
/// `FinishReason::try_from(i)`). Preserved verbatim.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeneratedText {
    /// Output text.
    pub text: String,
    /// Number of generated tokens.
    pub generated_tokens: u32,
    /// Finish reason (decode with `FinishReason::try_from`).
    pub finish_reason: i32,
    /// Sampling seed, if any.
    pub seed: Option<u64>,
}

// ==== Finish reason ====

/// Reason the engine stopped generating.
///
/// Mirrors `enum FinishReason { FINISH_REASON_LENGTH = 0;
/// FINISH_REASON_EOS_TOKEN = 1; FINISH_REASON_STOP_SEQUENCE = 2; }`.
///
/// Variant names use the auto-converted CamelCase form (`Length`,
/// `EosToken`, `StopSequence`) inherited from the legacy prost output,
/// so existing match arms compile unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum FinishReason {
    Length = 0,
    EosToken = 1,
    StopSequence = 2,
}

impl Default for FinishReason {
    fn default() -> Self {
        FinishReason::Length
    }
}

/// Error returned when an `i32` does not match any `FinishReason`
/// discriminant. Mirrors the legacy `try_from` semantics; the original
/// error type was an opaque `DecodeError` whose details no caller in
/// this crate inspects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownFinishReason(pub i32);

impl std::fmt::Display for UnknownFinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown FinishReason discriminant: {}", self.0)
    }
}

impl std::error::Error for UnknownFinishReason {}

impl TryFrom<i32> for FinishReason {
    type Error = UnknownFinishReason;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(FinishReason::Length),
            1 => Ok(FinishReason::EosToken),
            2 => Ok(FinishReason::StopSequence),
            other => Err(UnknownFinishReason(other)),
        }
    }
}

// ==== Service info ====

/// Backend shard info reported at startup.
///
/// Pared to the fields actually consumed by the gateway: `dtype` and
/// `device_type` flow into `Info { model_dtype, model_device_type }`
/// for the `/info` debug endpoint, and `speculate` parameterises the
/// Prometheus latency-bucket setup. The legacy `requires_padding` and
/// `window_size` fields had no live reader and were dropped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InfoResponse {
    /// Compute dtype label (e.g. `"float16"`).
    pub dtype: String,
    /// Device type label (e.g. `"cuda"`).
    pub device_type: String,
    /// Number of speculatively generated tokens per step.
    pub speculate: u32,
}

// ==== Generation parameters ====

/// Per-request generation parameters: a merged sampling + stopping
/// configuration that mirrors the OpenAI ChatCompletion → vanilla vLLM
/// passthrough surface.
///
/// Field selection follows the rule: a field is kept iff it can flow
/// from an OpenAI-style request through to vanilla vLLM's
/// `CompletionRequest` (see `vllm/entrypoints/openai/protocol.py`).
/// Fields currently unwired but in the path are kept stubbed (`Default`
/// supplies sensible neutrals) so the wiring can be added later without
/// reshape. Fields not in the path were dropped:
///
///   - `typical_p` (no vLLM equivalent at the OpenAI layer)
///   - `do_sample` (vLLM uses temperature == 0 to mean greedy)
///   - `watermark` (TGI-only)
///
/// The legacy split between `NextTokenChooserParameters` and
/// `StoppingCriteriaParameters` was a proto-era artefact; the merged
/// shape matches how callers consume it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationParams {
    /// Exponential scaling of the output probability distribution.
    pub temperature: f32,
    /// Restrict to the `k` highest probability elements.
    pub top_k: u32,
    /// Restrict to top tokens summing to `<= top_p`.
    pub top_p: f32,
    /// Repetition penalty.
    pub repetition_penalty: f32,
    /// Random seed for sampling.
    pub seed: u64,
    /// Maximum number of generated tokens.
    pub max_new_tokens: u32,
    /// Stopping sequences (matched against generated text).
    pub stop_sequences: Vec<String>,
    /// Ignore end-of-sequence token (used for benchmarking).
    pub ignore_eos_token: bool,
}
