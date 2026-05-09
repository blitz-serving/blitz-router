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

// ==== Tokens ====

/// Per-step token output for a single request.
///
/// Mirrors `message Tokens { repeated uint32 ids = 1; repeated float
/// logprobs = 2; repeated string texts = 3; repeated bool is_special = 4; }`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tokens {
    /// Token IDs.
    pub ids: Vec<u32>,
    /// Logprobs.
    pub logprobs: Vec<f32>,
    /// Decoded token strings.
    pub texts: Vec<String>,
    /// Whether each token is a special token.
    pub is_special: Vec<bool>,
}

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

/// Per-request generation event emitted by the (legacy) backend.
///
/// Mirrors `message Generation { required uint64 request_id = 1;
/// optional Tokens prefill_tokens = 2; required Tokens tokens = 3;
/// optional GeneratedText generated_text = 4; repeated Tokens
/// top_tokens = 5; }`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Generation {
    /// Request ID.
    pub request_id: u64,
    /// Prefill tokens (optional).
    pub prefill_tokens: Option<Tokens>,
    /// Newly produced tokens for this step.
    pub tokens: Tokens,
    /// Final generated text, if the request just finished.
    pub generated_text: Option<GeneratedText>,
    /// Top-N alternative tokens (optional).
    pub top_tokens: Vec<Tokens>,
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
/// Mirrors `message InfoResponse { required bool requires_padding = 1;
/// required string dtype = 2; required string device_type = 3;
/// optional uint32 window_size = 4; required uint32 speculate = 5; }`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InfoResponse {
    /// Whether the backend requires inputs to be padded.
    pub requires_padding: bool,
    /// Compute dtype label (e.g. `"float16"`).
    pub dtype: String,
    /// Device type label (e.g. `"cuda"`).
    pub device_type: String,
    /// Optional sliding-window size.
    pub window_size: Option<u32>,
    /// Number of speculatively generated tokens per step.
    pub speculate: u32,
}

// ==== Sampling parameters ====

/// Per-request sampling configuration.
///
/// Mirrors `message NextTokenChooserParameters { required float
/// temperature = 1; required uint32 top_k = 2; required float top_p = 3;
/// required float typical_p = 4; required bool do_sample = 5; required
/// uint64 seed = 6; required float repetition_penalty = 7; required
/// bool watermark = 8; }`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NextTokenChooserParameters {
    /// Exponential scaling output probability distribution.
    pub temperature: f32,
    /// Restrict to the `k` highest probability elements.
    pub top_k: u32,
    /// Restrict to top tokens summing to `<= top_p`.
    pub top_p: f32,
    /// Restrict to top tokens summing to `<= typical_p`.
    pub typical_p: f32,
    /// Apply sampling on the logits.
    pub do_sample: bool,
    /// Random seed for sampling.
    pub seed: u64,
    /// Repetition penalty.
    pub repetition_penalty: f32,
    /// Token watermarking ("A Watermark for Large Language Models").
    pub watermark: bool,
}

/// Per-request stopping criteria.
///
/// Mirrors `message StoppingCriteriaParameters { required uint32
/// max_new_tokens = 1; repeated string stop_sequences = 2; required
/// bool ignore_eos_token = 3; }`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StoppingCriteriaParameters {
    /// Maximum number of generated tokens.
    pub max_new_tokens: u32,
    /// Optional stopping sequences.
    pub stop_sequences: Vec<String>,
    /// Ignore end-of-sequence token (used for benchmarking).
    pub ignore_eos_token: bool,
}
