// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// This file is a **modified** version of
// text-generation-inference/src/token_stream.rs
// © 2022-present Hugging Face Inc. – Apache-2.0.
//
// Modifications by Blitz-serving:
//   - Add more structure for blitzscale
#[cfg(not(any(
    feature = "impl_blitz",
    feature = "impl_sllm",
    feature = "manually_scale",
    feature = "vllm-backend",
    feature = "zmq-backend",
)))]
compile_error!("You must define how this system is running!");

#[cfg(all(feature = "colocation", feature = "disaggregation"))]
compile_error!("You must specify either colocation or disaggregation!");

#[cfg(all(feature = "impl_blitz", feature = "mock_transfer"))]
compile_error!("Feature mock_transfer is not supported with impl_blitz.");

#[cfg(all(feature = "impl_blitz", feature = "impl_sllm"))]
compile_error!("Feature impl_sllm is not supported with impl_blitz.");

#[cfg(all(feature = "cache_all_miss", feature = "cache_replace"))]
compile_error!("Feature mock_load is not supported with enable_cache.");

#[cfg(all(feature = "mock_transfer", feature = "cache_replace"))]
compile_error!("Feature mock_transfer is not supported with enable_cache.");

#[cfg(all(feature = "cache_all_miss", feature = "mock_transfer"))]
compile_error!("Features 'mock_load' and 'mock_transfer' cannot be enabled at the same time.");

#[cfg(all(feature = "mock_transfer", feature = "cache_all_hit"))]
compile_error!("Features 'mock_transfer' and 'mock_cache_hit' cannot be enabled at the same time.");

#[cfg(all(feature = "mock_transfer", feature = "cache_replace"))]
compile_error!("Features 'mock_transfer' and 'enable_cache' cannot be enabled at the same time.");

/// Compilation time features satisfiability checker
///
/// TODO: add detailed rules!
#[allow(dead_code)]
const fn feat_sat_checker() {
    let serverless_llm = cfg!(feature = "impl_sllm");
    let blitz_scale = cfg!(feature = "impl_blitz");
    if cfg!(feature = "vllm-backend") || cfg!(feature = "zmq-backend") {
        assert!(!serverless_llm && !blitz_scale);
        return;
    }
    assert!(serverless_llm != blitz_scale);
    let rdma = cfg!(feature = "impl_rdma");
    let nvlink = cfg!(feature = "impl_nvl");
    assert!((rdma || nvlink) ^ serverless_llm);
    // one of
    if serverless_llm {
        let (a, b, c) = (
            cfg!(feature = "cache_all_miss"),
            cfg!(feature = "cache_all_hit"),
            cfg!(feature = "cache_replace"),
        );
        assert!(a && !b && !c || !a && b && !c || !a && !b && c);
    }
}

#[allow(dead_code)]
const VALID_FEATURE: () = feat_sat_checker();

mod kvcache;
#[cfg(any(test, kani))]
mod radixtrie;
mod policies;
mod replica;
mod statistic;
mod stub;
mod vllmlet;
#[cfg(feature = "zmq-backend")]
pub mod zmq_engine;
pub mod engine_client;

pub use replica::*;
pub use stub::*;
pub use vllmlet::*;

pub use infer::{parse_deployment, Deployment};

pub mod error;
mod health;
pub(crate) mod infer;
mod queue;
pub mod server;
mod validation;

use infer::Infer;
use policies::Entry;
use queue::Queue;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validation::{Validation, ValidationError};

// `TokenizerRender`
use minijinja::{context, Environment};
use std::{fs, path::Path};
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

fn default_parameters() -> GenerateParameters {
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
}

#[allow(unused)]
#[derive(Clone, Debug, Deserialize, ToSchema)]
#[serde(tag = "type", content = "data")]
pub(crate) enum ModifyClusterStateRequest {
    /// The model_ready of new instance will be set to true automatically when the instance is ready.
    TriggerMutation {
        stub_indices: Vec<usize>,
    },

    TriggerPrefillUp {
        old_stub_indices: Vec<usize>,
        new_stub_indices: Vec<usize>,
    },

    TriggerScaleDown {
        stub_indices: Vec<usize>,
    },

    TriggerDecodeUp {
        old_stub_indices: Vec<usize>,
        new_stub_indices: Vec<usize>,
    },

    TriggerNormalUp {
        old_stub_indices: Vec<usize>,
        new_stub_indices: Vec<usize>,
    },

    TriggerNoramalDown {
        stub_indices: Vec<usize>,
    },
}

#[test]
pub fn test_print_modifyclusterstaterequest() {
    let req = ModifyClusterStateRequest::TriggerPrefillUp {
        old_stub_indices: vec![1, 2],
        new_stub_indices: vec![3, 4],
    };
    println!("{:?}", req);
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
        Self { inputs: req.inputs, parameters: req.parameters }
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
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(nullable = true, example = "-0.34")]
    logprob: f32,
    #[schema(example = "false")]
    special: bool,
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

use core::hint::spin_loop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::yield_now;

pub struct SpinLock {
    flag: AtomicBool, // false: unlocked, true: locked
}

unsafe impl Send for SpinLock {}
unsafe impl Sync for SpinLock {}

#[allow(unused)]
impl SpinLock {
    pub const fn new() -> Self {
        Self { flag: AtomicBool::new(false) }
    }

    /// Blocking spinlock
    pub fn lock(&self) {
        // test-and-set + backoff
        let mut spins = 0u32;
        loop {
            while self.flag.load(Ordering::Relaxed) {
                spins = backoff(spins);
            }

            match self.flag.compare_exchange(
                false,
                true,
                Ordering::Acquire, // 成功获取，建立 Acquire 栅栏
                Ordering::Relaxed, // 失败分支可用 Relaxed
            ) {
                Ok(_) => break,
                Err(_) => {
                    spins = backoff(spins);
                }
            }
        }
    }

    #[inline]
    fn unlock(&self) {
        self.flag.store(false, Ordering::Release);
    }
}

/// Exponential backoff: busy-spin at beginning, then yield time slice
#[inline]
fn backoff(spins: u32) -> u32 {
    // 前 64 次：CPU hint
    if spins < 64 {
        spin_loop();
        spins + 1
    } else {
        // 偶尔让出 CPU，避免长期霸占
        if spins & 0xF == 0 {
            // 每 16 次让出一次
            yield_now();
        } else {
            spin_loop();
        }
        spins.saturating_add(1)
    }
}

/// OpenAI's format
#[derive(Serialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String, // "system", "user", "assistant"
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct TokenizerRender {
    tokenizer: Tokenizer,
    tmpl_env: minijinja::Environment<'static>,
}

impl TokenizerRender {
    pub fn new(local_path: &Path) -> Self {
        let tokenizer = Tokenizer::from_file(local_path.join("tokenizer.json"))
            .ok()
            .expect("Invalid tokenizer.json file path!");
        let tkn_config_text = fs::read_to_string(local_path.join("tokenizer_config.json"))
            .expect("Invalid tokenizer_config.json file path!");
        let tkn_config_json: serde_json::Value = serde_json::from_str(&tkn_config_text)
            .expect("Invalid tokenzier_config.json file content!");
        // NOTE: ChatGPT says, `serde::Value::to_string` will serialize again, and produce Json string,
        //       however, a Json string contains unwelcome escape characters
        let chat_tmpl: String = tkn_config_json["chat_template"]
            .as_str()
            .expect("tokenzier_config.json file does not contain `chat_template`!")
            .to_string();
        let mut env = Environment::new();
        env.add_template_owned("ChatML", chat_tmpl).unwrap();
        Self { tokenizer, tmpl_env: env }
    }

    pub fn apply_chat_template(
        &self,
        messages: &Vec<ChatMessage>,
    ) -> Result<String, ValidationError> {
        // NOTE: ChatGPT says jinja2 is readily compitable with serde,
        //       and thus serialization is useless
        let chat_tmpl = self
            .tmpl_env
            .get_template("ChatML")
            .map_err(|e| ValidationError::Tokenizer(e.to_string()))?;
        let rendered_msg = chat_tmpl
            .render(context! { messages => messages, add_generation_prompt => true, })
            .map_err(|e| ValidationError::Tokenizer(e.to_string()))?;
        Ok(rendered_msg)
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
            // Re-check if another process has written this file maybe.
            if !filename.exists() {
                std::fs::rename(tmp_filename, filename).unwrap()
            }
        }
        Tokenizer::from_file("tokenizer.json").unwrap()
    }

    fn check_rendered_chat_message(model_name: &str, message: &str, content: &str) {
        match model_name {
            "Qwen2-7B-Instruct" => {
                assert!(message.contains("<|im_start|>system"));
                assert!(message.contains("You are a helpful assistant."));
                assert!(message.contains("<|im_start|>user"));
                assert!(message.contains(content));
                assert!(message.contains("<|im_start|>assistant"));
            }
            "Qwen2.5-7B-Instruct" => {
                assert!(message.contains("<|im_start|>system"));
                assert!(message.contains("You are Qwen, created by Alibaba Cloud."));
                assert!(message.contains("You are a helpful assistant."));
                assert!(message.contains("<|im_start|>user"));
                assert!(message.contains(content));
                assert!(message.contains("<|im_start|>assistant"));
            }
            "Meta-Llama-3-8B-Instruct" => {
                assert!(message.contains("<|im_start|>system"));
                assert!(message.contains("<|im_start|>user"));
                assert!(message.contains(content));
                assert!(message.contains("<|im_start|>assistant"));
            }
            "Qwen3-8B" => {
                assert!(message.contains("<|im_start|>user"));
                assert!(message.contains(content));
                assert!(message.contains("<|im_start|>assistant"));
            }
            &_ => unreachable!("Invalid model name!"),
        }
    }

    #[test]
    fn test_tokenizer_render_apply_chat_template() {
        let test_cases = [
            "Qwen2-7B-Instruct",
            "Qwen2.5-7B-Instruct",
            /*unsupported*/ "Meta-Llama-3-8B-Instruct",
            /*unsupported*/ "Qwen3-8B",
        ];
        let model_cache = Path::new("/nvme/models");
        let messages =
            vec![ChatMessage { role: "user".into(), content: "What is deep learning?".into() }];

        for model_name in test_cases {
            let local_path = model_cache.join(model_name);
            assert!(local_path.exists() && local_path.is_dir());
            let tokenizer_render = TokenizerRender::new(&local_path);

            let result = tokenizer_render.apply_chat_template(&messages);
            assert!(result.is_ok(), "Expected Ok, got {:?}", result);
            let rendered = result.unwrap();
            println!("Rendered:\n++++++++{}--------", rendered);

            check_rendered_chat_message(model_name, &rendered, &messages[0].content);
        }
    }
}
