// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// Derived from text-generation-inference by Hugging Face Inc. (Apache-2.0).

#[cfg(not(any(feature = "vllm-backend", feature = "zmq-backend")))]
compile_error!("You must enable either `vllm-backend` or `zmq-backend`!");

mod kvcache;
#[cfg(any(test, kani))]
mod radixtrie;
mod policies;
mod colocation;
mod metrics;
mod statistic;
mod vllmlet;
#[cfg(feature = "zmq-backend")]
pub mod zmq_engine;
pub mod engine_client;
pub mod chat_template;
pub mod model_config;

pub use colocation::*;
pub use metrics::*;
pub use vllmlet::*;
pub use chat_template::{ChatRenderer, load_chat_template};

pub mod error;
mod health;
pub(crate) mod infer;
mod queue;
pub mod server;
mod validation;

use infer::Infer;
use policies::Entry;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validation::{Validation, ValidationError};

// `TokenizerRender`
use std::path::Path;
use tokenizers::Tokenizer;

/// Hub type
#[derive(Clone, Debug, Deserialize)]
pub struct HubModelInfo {
    #[serde(rename(deserialize = "id"))]
    pub model_id: String,
    pub sha: Option<String>,
    pub pipeline_tag: Option<String>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct Info {
    /// Model info
    #[schema(example = "bigscience/blomm-560m")]
    pub model_id: String,
    #[schema(nullable = true, example = "e985a63cdc139290c5f700ff1929f0b5942cced2")]
    pub model_sha: Option<String>,
    #[schema(example = "torch.float16")]
    pub model_dtype: String,
    #[schema(example = "cuda")]
    pub model_device_type: String,
    #[schema(nullable = true, example = "text-generation")]
    pub model_pipeline_tag: Option<String>,
    /// Router Parameters
    #[schema(example = "128")]
    pub max_concurrent_requests: usize,
    #[schema(example = "2")]
    pub max_best_of: usize,
    #[schema(example = "4")]
    pub max_stop_sequences: usize,
    #[schema(example = "1024")]
    pub max_input_length: usize,
    #[schema(example = "2048")]
    pub max_total_tokens: usize,
    #[schema(example = "1.2")]
    pub waiting_served_ratio: f32,
    #[schema(example = "32000")]
    pub max_batch_total_tokens: u32,
    #[schema(example = "20")]
    pub max_waiting_tokens: usize,
    #[schema(example = "2")]
    pub validation_workers: usize,
    /// Router Info
    #[schema(example = "0.5.0")]
    pub version: &'static str,
    #[schema(nullable = true, example = "null")]
    pub sha: Option<&'static str>,
    #[schema(nullable = true, example = "null")]
    pub docker_label: Option<&'static str>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct GenerateParameters {
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 1)]
    pub best_of: Option<usize>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0.0, nullable = true, default = "null", example = 0.5)]
    pub temperature: Option<f32>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0.0, nullable = true, default = "null", example = 1.03)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 10)]
    pub top_k: Option<i32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        maximum = 1.0,
        nullable = true,
        default = "null",
        example = 0.95
    )]
    pub top_p: Option<f32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        maximum = 1.0,
        nullable = true,
        default = "null",
        example = 0.95
    )]
    pub typical_p: Option<f32>,
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub do_sample: bool,
    #[serde(default = "default_max_new_tokens")]
    #[schema(nullable = true, default = "100", example = "20")]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = false)]
    pub return_full_text: Option<bool>,
    #[serde(default)]
    #[schema(inline, max_items = 4, example = json ! (["photographer"]))]
    pub stop: Vec<String>,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "null")]
    pub truncate: Option<usize>,
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub watermark: bool,
    #[serde(default)]
    #[schema(default = "true")]
    pub details: bool,
    #[serde(default)]
    #[schema(default = "true")]
    pub decoder_input_details: bool,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = "null")]
    pub seed: Option<u64>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 5)]
    pub top_n_tokens: Option<u32>,
}

fn default_max_new_tokens() -> Option<u32> {
    Some(100)
}

pub(crate) fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        best_of: None,
        temperature: None,
        repetition_penalty: None,
        top_k: None,
        top_p: None,
        typical_p: None,
        do_sample: false,
        max_new_tokens: default_max_new_tokens(),
        return_full_text: None,
        stop: Vec::new(),
        truncate: None,
        watermark: false,
        details: false,
        decoder_input_details: false,
        seed: None,
        top_n_tokens: None,
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct GenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
    /// Pre-provided chat messages (set by /v1/chat/completions handler).
    /// When set, the tokenizer worker uses these for chat template rendering
    /// instead of wrapping `inputs` as a single user message.
    #[serde(skip)]
    pub(crate) chat_messages: Option<Vec<ChatMessage>>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct CompatGenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
    #[serde(default)]
    #[schema(default = "false")]
    pub stream: bool,
}

impl From<CompatGenerateRequest> for GenerateRequest {
    fn from(req: CompatGenerateRequest) -> Self {
        Self { inputs: req.inputs, parameters: req.parameters, chat_messages: None }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PrefillToken {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(nullable = true, example = "-0.34")]
    logprob: f32,
}

#[derive(Debug, Serialize, ToSchema, Default)]
pub struct Token {
    #[schema(example = 0)]
    pub(crate) id: u32,
    #[schema(example = "test")]
    pub(crate) text: String,
    #[schema(nullable = true, example = "-0.34")]
    pub(crate) logprob: f32,
    #[schema(example = "false")]
    pub(crate) special: bool,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all(serialize = "snake_case"))]
pub(crate) enum FinishReason {
    #[schema(rename = "length")]
    Length,
    #[serde(rename = "eos_token")]
    #[schema(rename = "eos_token")]
    EndOfSequenceToken,
    #[schema(rename = "stop_sequence")]
    StopSequence,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct BestOfSequence {
    #[schema(example = "test")]
    pub generated_text: String,
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
    pub prefill: Vec<PrefillToken>,
    pub tokens: Vec<Token>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_tokens: Vec<Vec<Token>>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Details {
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
    pub prefill: Vec<PrefillToken>,
    pub tokens: Vec<Token>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of_sequences: Option<Vec<BestOfSequence>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_tokens: Vec<Vec<Token>>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct GenerateResponse {
    #[schema(example = "test")]
    pub generated_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Details>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct StreamDetails {
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct StreamResponse {
    pub token: Token,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_tokens: Vec<Token>,
    #[schema(nullable = true, default = "null", example = "test")]
    pub generated_text: Option<String>,
    #[schema(nullable = true, default = "null")]
    pub details: Option<StreamDetails>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ErrorResponse {
    pub error: String,
    pub error_type: String,
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completions API types
// ---------------------------------------------------------------------------

/// OpenAI-compatible chat completion request.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<ChatCompletionStop>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

/// The `stop` field in OpenAI API can be a string or array of strings.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ChatCompletionStop {
    Single(String),
    Multiple(Vec<String>),
}

impl ChatCompletionStop {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            ChatCompletionStop::Single(s) => vec![s],
            ChatCompletionStop::Multiple(v) => v,
        }
    }
}

/// OpenAI-compatible chat completion response (non-streaming).
#[derive(Serialize)]
pub(crate) struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: ChatCompletionUsage,
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI-compatible chat completion chunk (streaming).
#[derive(Serialize)]
pub(crate) struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// OpenAI's format
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String, // "system", "user", "assistant"
    pub content: String,
}

/// Wrapper around HuggingFace `tokenizers::Tokenizer` for encoding/decoding.
///
/// Chat template rendering is NOT done here — see [`ChatRenderer`] instead.
/// This struct only handles text ↔ token ID conversion.
#[derive(Debug, Clone)]
pub struct TokenizerRender {
    pub tokenizer: Tokenizer,
}

impl TokenizerRender {
    pub fn new(local_path: &Path) -> Self {
        let tokenizer = Tokenizer::from_file(local_path.join("tokenizer.json"))
            .ok()
            .expect("Invalid tokenizer.json file path!");
        Self { tokenizer }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[allow(unused)]
    #[deprecated = "unintended to download models"]
    pub(crate) async fn get_tokenizer() -> Tokenizer {
        let filename = std::path::Path::new("tokenizer.json");
        if !filename.exists() {
            let content = reqwest::get("https://huggingface.co/gpt2/raw/main/tokenizer.json")
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            let tmp_filename = "tokenizer.json.temp";
            let mut file = std::fs::File::create(tmp_filename).unwrap();
            file.write_all(&content).unwrap();
            if !filename.exists() {
                std::fs::rename(tmp_filename, filename).unwrap()
            }
        }
        Tokenizer::from_file("tokenizer.json").unwrap()
    }
}
