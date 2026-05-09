// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// This file is a **modified** version of
// text-generation-inference/src/token_stream.rs
// © 2022-present Hugging Face Inc. – Apache-2.0.
#![allow(unused)]

use crate::engine::EngineClient;
use super::kvcache::{BlockHash, BlockHashState, PrefixBlockHash};
use super::queue::{QueuePro, TaskAssigner};
use super::statistic::statistic;
use crate::gateway::validation::{Validation, ValidationError};
use crate::{
    start_vllm_colocation_event_loop,
    ColocationController, Entry, LMetric,
    ScheduleContext, Token,
};

use crate::GenerateRequest;

use std::sync::{atomic::AtomicBool, Arc};
use std::time::Duration;

use futures::future::try_join_all;
use nohash_hasher::IntMap;
use crate::types::GeneratedText;
use thiserror::Error;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::{mpsc, Mutex, OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tracing::{info_span, instrument, Span};

/// Inference struct
#[derive(Clone)]
pub struct Infer {
    /// KVCache block size
    block_size: usize,
    /// Validation
    validation: Validation,
    /// Request queue
    queue: TaskAssigner,

    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,

    /// Colocation controller
    colocation_controller: Option<Arc<ColocationController>>,
}

impl Infer {
    pub(crate) fn create_vllm_colocation(
        all_engine_clients: Vec<Box<dyn EngineClient>>,
        block_size: usize,
        validation: Validation,
        max_concurrent_requests: usize,
        statistic_path: Option<String>,
        shared_tokenizer: Option<Arc<tokenizers::Tokenizer>>,
    ) -> Self {
        let num_replicas = all_engine_clients.len();
        let all_schedule_contexts: Vec<Arc<Mutex<ScheduleContext>>> = (0..num_replicas)
            .map(|_| {
                Arc::new(Mutex::new(ScheduleContext {
                    lmetric: LMetric::default(),
                    // FIXME(@Healthcliff-Ding): a hack for Qwen2.5-7B
                    block_hash: PrefixBlockHash::new(83000 + 1),
                }))
            })
            .collect();

        let queue = TaskAssigner::new(num_replicas, all_schedule_contexts.clone());

        let controller = start_vllm_colocation_event_loop(
            queue.clone(),
            all_engine_clients,
            all_schedule_contexts.clone(),
            shared_tokenizer,
        );

        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));

        tokio::spawn(statistic(all_schedule_contexts, statistic_path));

        Self {
            validation,
            block_size,
            queue,
            limit_concurrent_requests: semaphore,
            colocation_controller: Some(controller),
        }
    }

    /// Add a new request to the queue and return a stream of InferStreamResponse
    #[instrument(skip_all)]
    pub(crate) async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<
        (
            u64,
            OwnedSemaphorePermit,
            UnboundedReceiverStream<Result<InferStreamResponse, InferError>>,
        ),
        InferError,
    > {
        // Limit concurrent requests by acquiring a permit from the semaphore
        let permit = self.clone().limit_concurrent_requests.try_acquire_owned().map_err(|err| {
            metrics::increment_counter!("blitz_request_failure", "err" => "overloaded");
            tracing::error!("{err}");
            err
        })?;

        // Validate request
        let valid_request = self.validation.validate(request).await.map_err(|err| {
            metrics::increment_counter!("blitz_request_failure", "err" => "validation");
            tracing::error!("{err}");
            err
        })?;

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        let request_id = valid_request.request_id;
        let block_hash_state = BlockHashState::new(&valid_request.input_tokens, self.block_size);

        // Append the request to the queue
        self.queue.append(Entry {
            request: valid_request,
            block_hash_state,
            response_tx,
            span: Span::current(),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
            generated_token_cnt: 0,
            prev_token_time: None,
            time_of_per_token: None,
            max_time_between_tokens: Duration::from_micros(0),
        });

        // Return stream
        Ok((request_id, permit, UnboundedReceiverStream::new(response_rx)))
    }

    /// Add a new request to the queue and return a InferResponse
    #[instrument(skip_all)]
    pub(crate) async fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<InferResponse, InferError> {
        let use_top_tokens = request.parameters.top_n_tokens.is_some_and(|x| x > 0);

        // Create stream and keep semaphore permit as long as generate lives
        let (request_id, _permit, mut stream) = self.generate_stream(request).await?;

        // Return values
        let mut result_tokens = Vec::new();
        let mut result_top_tokens = Vec::new();
        let mut result_generated_text = None;
        let mut result_start = None;
        let mut result_queued = None;
        let mut result_max_time_between_tokens = None;
        let mut time_between_tokens: Vec<Duration> = Vec::new();
        let mut input_length = 0;
        let mut output_length = 0;

        let s = tokio::time::Instant::now();
        let mut interval = s.clone();
        let mut first_token_time = None;

        // Iterate on stream
        while let Some(response) = stream.next().await {
            match response? {
                // Add prefill tokens
                InferStreamResponse::Prefill => {
                    first_token_time.get_or_insert(s.elapsed());
                    interval = tokio::time::Instant::now();
                }
                // Push last token
                InferStreamResponse::Intermediate { token, top_tokens } => {
                    result_tokens.push(token);
                    result_top_tokens.push(top_tokens);
                    time_between_tokens.push(interval.elapsed());
                    interval = tokio::time::Instant::now();
                }
                // Final message
                // Set return values
                InferStreamResponse::End {
                    token,
                    generated_text,
                    start,
                    queued,
                    top_tokens,
                    max_time_between_tokens,
                } => {
                    result_generated_text = Some(generated_text);
                    result_start = Some(start);
                    result_queued = Some(queued);
                    result_max_time_between_tokens = Some(max_time_between_tokens);
                    // Removes a dummy startup duration.
                    //
                    // I guess the origin TGI designs a debug backdoor to visualize prefilled logits
                    // similar to the training process in `enum::Prefill`. However, we pack `enum::Prefill`
                    // and `enum::Intermediate` together.
                    time_between_tokens.remove(0);
                    output_length = result_tokens.len();
                    break;
                }
            }
        }

        // Check that we received a `InferStreamResponse::End` message
        if let (Some(mut generated_text), Some(queued), Some(start)) =
            (result_generated_text, result_queued, result_start)
        {
            let mut max_time_between_tokens_except_first = Duration::from_micros(0);
            let mut avg_time_between_tokens = Duration::from_micros(0);
            let mut p90_time_between_tokens = Duration::from_micros(0);
            let mut p95_time_between_tokens = Duration::from_micros(0);
            let mut p99_time_between_tokens = Duration::from_micros(0);
            let mut first_decode_token_time = Duration::from_micros(0);
            if !time_between_tokens.is_empty() {
                first_decode_token_time = time_between_tokens[0];
                time_between_tokens.sort();

                max_time_between_tokens_except_first = time_between_tokens
                    .iter()
                    .rev()
                    .nth(1)
                    .cloned()
                    .unwrap_or(Duration::from_micros(0));
                result_max_time_between_tokens = Some(time_between_tokens.last().cloned().unwrap());

                let len = time_between_tokens.len();
                avg_time_between_tokens = time_between_tokens.iter().sum::<Duration>() / len as u32;
                p90_time_between_tokens = time_between_tokens[(len as f32 * 0.9) as usize];
                p95_time_between_tokens = time_between_tokens[(len as f32 * 0.95) as usize];
                p99_time_between_tokens = time_between_tokens[(len as f32 * 0.99) as usize];
            }

            Ok(InferResponse {
                request_id,
                tokens: result_tokens,
                generated_text,
                queued,
                start,
                top_tokens: if use_top_tokens { result_top_tokens } else { Vec::new() },
                first_token_time: first_token_time.unwrap(),
                max_time_between_tokens: result_max_time_between_tokens.unwrap(),
                max_time_between_tokens_except_first,
                avg_time_between_tokens,
                p90_time_between_tokens,
                p95_time_between_tokens,
                p99_time_between_tokens,
                first_decode_token_time,
                input_length,
                output_length,
            })
        } else {
            let err = InferError::IncompleteGeneration;
            metrics::increment_counter!("blitz_request_failure", "err" => "incomplete");
            tracing::error!("{err}");
            Err(err)
        }
    }
    /// Add best_of new requests to the queue and return a InferResponse of the sequence with
    /// the highest log probability per token
    #[instrument(skip_all)]
    pub(crate) async fn generate_best_of(
        &self,
        request: GenerateRequest,
        best_of: usize,
    ) -> Result<(InferResponse, Vec<InferResponse>), InferError> {
        // validate  best_of parameter separately
        let best_of = self.validation.validate_best_of(best_of)?;

        // create multiple generate requests
        let mut infer_responses: Vec<InferResponse> =
            try_join_all((0..best_of).map(|_| self.generate(request.clone()))).await?;

        // get the sequence with the highest log probability per token
        let mut max_index = 0;
        let mut max_logprob: f32 = f32::MIN;

        for (i, response) in infer_responses.iter().enumerate() {
            // mean logprobs of the generated tokens
            let sequence_logprob = response.tokens.iter().map(|token| token.logprob).sum::<f32>()
                / response.tokens.len() as f32;

            // set best sequence
            if sequence_logprob > max_logprob {
                max_index = i;
                max_logprob = sequence_logprob;
            }
        }
        let best_response = infer_responses.remove(max_index);
        Ok((best_response, infer_responses))
    }
}

/// (BLITZ) :: send generation to http server
/// Send one or multiple `InferStreamResponse` to Infer for all `entries`
/// and filter entries
#[derive(Debug)]
pub(crate) enum InferStreamResponse {
    // Optional first message
    Prefill,
    // Intermediate messages
    Intermediate {
        token: Token,
        top_tokens: Vec<Token>,
    },
    // Last message
    End {
        token: Token,
        top_tokens: Vec<Token>,
        generated_text: GeneratedText,
        start: Instant,
        queued: Instant,
        max_time_between_tokens: Duration,
    },
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct InferResponse {
    pub(crate) request_id: u64,
    pub(crate) tokens: Vec<Token>,
    pub(crate) generated_text: GeneratedText,
    pub(crate) queued: Instant,
    pub(crate) start: Instant,
    pub(crate) top_tokens: Vec<Vec<Token>>,
    pub(crate) first_token_time: std::time::Duration,
    pub(crate) max_time_between_tokens: std::time::Duration,
    pub(crate) max_time_between_tokens_except_first: std::time::Duration,
    pub(crate) first_decode_token_time: std::time::Duration,
    pub(crate) avg_time_between_tokens: std::time::Duration,
    pub(crate) p90_time_between_tokens: std::time::Duration,
    pub(crate) p95_time_between_tokens: std::time::Duration,
    pub(crate) p99_time_between_tokens: std::time::Duration,
    pub(crate) input_length: usize,
    pub(crate) output_length: usize,
}

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
    #[error("Model is overloaded")]
    Overloaded(#[from] TryAcquireError),
    #[error("Input validation error: {0}")]
    ValidationError(#[from] ValidationError),
    #[error("Incomplete generation")]
    IncompleteGeneration,
}

impl InferError {
    pub(crate) fn error_type(&self) -> &str {
        match self {
            InferError::GenerationError(_) => "generation",
            InferError::Overloaded(_) => "overloaded",
            InferError::ValidationError(_) => "validation",
            InferError::IncompleteGeneration => "incomplete_generation",
        }
    }
}
