/// Payload validation logic
use super::chat_template::ChatRenderer;
use super::validation::ValidationError::{BestOfSampling, BestOfSeed, EmptyInput};
use crate::{ChatMessage, GenerateParameters, GenerateRequest, TokenizerRender};

use pb::generate::v2::{NextTokenChooserParameters, StoppingCriteriaParameters};
use rand::{thread_rng, Rng};
use thiserror::Error;
use tokenizers::TruncationDirection;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::{instrument, Span};

static RID_GENERATOR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    /// Validation parameters
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_length: usize,
    max_total_tokens: usize,
    /// Channel to communicate with the background tokenization task
    sender: Option<mpsc::UnboundedSender<TokenizerRequest>>,
}

impl Validation {
    pub(crate) fn new(
        workers: usize,
        tokenizer: Option<TokenizerRender>,
        chat_renderer: ChatRenderer,
        max_best_of: usize,
        max_stop_sequences: usize,
        max_top_n_tokens: u32,
        max_input_length: usize,
        max_total_tokens: usize,
    ) -> Self {
        // If we have a fast tokenizer
        let sender = if let Some(tokenizer) = tokenizer {
            // Create round robin channel
            let (validation_sender, validation_round_robin_receiver) = mpsc::unbounded_channel();
            let mut senders = Vec::with_capacity(workers);

            // Create workers
            for _ in 0..workers {
                let tokenizer_clone = tokenizer.clone();
                let chat_renderer_clone = chat_renderer.clone();
                let (tokenizer_sender, tokenizer_receiver) = mpsc::unbounded_channel();
                senders.push(tokenizer_sender);

                // Spawn worker
                tokio::task::spawn_blocking(move || {
                    tokenizer_worker(tokenizer_clone, chat_renderer_clone, tokenizer_receiver)
                });
            }

            // Create tokenization round robin task
            tokio::spawn(round_robin_task(validation_round_robin_receiver, senders));

            Some(validation_sender)
        } else {
            None
        };

        Self {
            max_best_of,
            sender,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
        }
    }

    #[instrument(skip_all)]
    async fn validate_input(
        &self,
        inputs: String,
        truncate: Option<usize>,
        max_new_tokens: Option<u32>,
        chat_messages: Option<Vec<ChatMessage>>,
    ) -> Result<(Vec<ChatMessage>, String, usize, u32, Vec<u32>), ValidationError> {
        // If we have a fast tokenizer
        if let Some(sender) = &self.sender {
            // Create response channel
            let (response_sender, response_receiver) = oneshot::channel();
            // Send request to the background validation task
            // Unwrap is safe here
            sender.send(((inputs, truncate, chat_messages), response_sender, Span::current())).unwrap();

            // Await on response channel
            // Unwrap is safe here
            let (messages, inputs, input_length, input_tokens) =
                response_receiver.await.unwrap()?;

            // Get total tokens
            let max_new_tokens: u32 = if let Some(max_new_tokens) = max_new_tokens {
                max_new_tokens
            } else {
                self.max_total_tokens.saturating_sub(input_length) as u32
            };
            let total_tokens = input_length + max_new_tokens as usize;

            // Validate MaxTotalTokens
            if total_tokens > self.max_total_tokens {
                return Err(ValidationError::MaxTotalTokens(
                    self.max_total_tokens,
                    input_length,
                    max_new_tokens,
                ));
            }

            // Validate InputLength
            if input_length > self.max_input_length {
                return Err(ValidationError::InputLength(self.max_input_length, input_length));
            }

            metrics::histogram!("blitz_request_input_length", input_length as f64);
            Ok((messages, inputs, input_length, max_new_tokens, input_tokens))
        }
        // Return inputs without validation
        else {
            // In this case, we don't know the real length in tokens of the inputs
            // However, the inputs will be truncated by the python servers
            // We make sure that truncate + max_new_tokens <= self.max_total_tokens
            let max_new_tokens: u32 = if let Some(max_new_tokens) = max_new_tokens {
                max_new_tokens
            } else if let Some(truncate) = truncate {
                self.max_total_tokens.saturating_sub(truncate) as u32
            } else {
                return Err(ValidationError::UnsetMaxNewTokens);
            };
            let input_length = truncate.unwrap_or(self.max_input_length);

            // Validate MaxNewTokens
            if (input_length as u32 + max_new_tokens) > self.max_total_tokens as u32 {
                return Err(ValidationError::MaxNewTokens(
                    self.max_total_tokens - self.max_input_length,
                    max_new_tokens,
                ));
            }

            // No tokenizer: use provided chat messages or wrap raw inputs
            let messages = chat_messages.unwrap_or_else(|| {
                vec![ChatMessage { role: "user".to_string(), content: inputs.clone() }]
            });
            Ok((messages, inputs, input_length, max_new_tokens, vec![]))
        }
    }

    /// Validate a payload and get the number of tokens in the input
    #[instrument(skip_all)]
    pub(crate) async fn validate(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, ValidationError> {
        let GenerateParameters {
            best_of,
            temperature,
            repetition_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            max_new_tokens,
            stop: stop_sequences,
            truncate,
            seed,
            watermark,
            decoder_input_details,
            top_n_tokens,
            ..
        } = request.parameters;

        // sampling must be true when best_of > 1
        let best_of = best_of.unwrap_or(1);
        let sampling = do_sample
            || temperature.is_some()
            || top_k.is_some()
            || top_p.is_some()
            || typical_p.is_some();

        if best_of > 1 && !sampling {
            return Err(BestOfSampling);
        }

        let temperature = temperature.unwrap_or(1.0);
        if temperature <= 0.0 {
            return Err(ValidationError::Temperature);
        }

        let repetition_penalty = repetition_penalty.unwrap_or(1.0);
        if repetition_penalty <= 0.0 {
            return Err(ValidationError::RepetitionPenalty);
        }

        // Different because the proto default value is not a valid value
        // for the user
        let top_p = top_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    return Err(ValidationError::TopP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let typical_p = typical_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    return Err(ValidationError::TypicalP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let top_k: u32 = top_k
            .map(|value| {
                if value <= 0 {
                    return Err(ValidationError::TopK);
                }
                Ok(value as u32)
            })
            .unwrap_or(Ok(0))?;

        if max_new_tokens == Some(0) {
            return Err(ValidationError::NegativeMaxNewTokens);
        }

        if stop_sequences.len() > self.max_stop_sequences {
            return Err(ValidationError::StopSequence(
                self.max_stop_sequences,
                stop_sequences.len(),
            ));
        }

        #[allow(unused_variables)]
        // If seed is None, assign a random one
        let seed: u64 = match seed {
            None => thread_rng().gen(),
            Some(seed) => {
                if best_of > 1 {
                    return Err(BestOfSeed);
                }
                seed
            }
        };

        let top_n_tokens = top_n_tokens
            .map(|value| {
                if value > self.max_top_n_tokens {
                    return Err(ValidationError::TopNTokens(self.max_top_n_tokens, value));
                }
                Ok(value)
            })
            .unwrap_or(Ok(0))?;

        // Check if inputs is empty (skip when chat messages are provided directly)
        if request.inputs.is_empty() && request.chat_messages.is_none() {
            return Err(EmptyInput);
        }

        // Check if truncate is strictly positive and less than max_input_length
        let truncate = truncate
            .map(|value| {
                if value == 0 || value > self.max_input_length {
                    return Err(ValidationError::Truncate(self.max_input_length, value));
                }
                Ok(Some(value))
            })
            .unwrap_or(Ok(None))?;

        // Validate inputs
        // NOTE: received `inputs` should be rendered according to chat template
        //       in file `tokenizer_config.json`
        let (messages, inputs, input_length, max_new_tokens, input_tokens) =
            self.validate_input(request.inputs, truncate, max_new_tokens, request.chat_messages).await?;

        let parameters = NextTokenChooserParameters {
            temperature,
            repetition_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            seed,
            watermark,
        };
        let stopping_parameters =
            StoppingCriteriaParameters { max_new_tokens, stop_sequences, ignore_eos_token: false };

        metrics::histogram!("blitz_request_max_new_tokens", max_new_tokens as f64);

        Ok(ValidGenerateRequest {
            request_id: RID_GENERATOR.fetch_add(1, std::sync::atomic::Ordering::AcqRel),
            messages,
            inputs,
            decoder_input_details,
            input_length: input_length as u32,
            truncate: truncate.unwrap_or(self.max_input_length) as u32,
            parameters,
            stopping_parameters,
            top_n_tokens,
            input_tokens,
        })
    }

    /// Validate the best_of parameter
    #[instrument(skip_all)]
    pub(crate) fn validate_best_of(&self, best_of: usize) -> Result<usize, ValidationError> {
        if self.max_best_of == 1 && best_of != 1 {
            return Err(ValidationError::BestOfDisabled);
        }

        if best_of > self.max_best_of {
            return Err(ValidationError::BestOf(self.max_best_of, best_of));
        }

        Ok(best_of)
    }
}

/// Round robin tokenization task
async fn round_robin_task(
    mut receiver: mpsc::UnboundedReceiver<TokenizerRequest>,
    senders: Vec<mpsc::UnboundedSender<TokenizerRequest>>,
) {
    loop {
        for sender in &senders {
            match receiver.recv().await {
                None => return,
                Some(request) => sender.send(request).unwrap(),
            };
        }
    }
}

/// Start tokenization workers
fn tokenizer_worker(
    tokenizer: TokenizerRender,
    mut chat_renderer: ChatRenderer,
    mut receiver: mpsc::UnboundedReceiver<TokenizerRequest>,
) {
    // Loop over requests
    while let Some(((inputs, truncate, chat_messages), response_tx, parent_span)) = receiver.blocking_recv() {
        parent_span.in_scope(|| {
            response_tx
                .send(prepare_input(inputs, truncate, chat_messages, &tokenizer, &mut chat_renderer))
                .unwrap_or(())
        })
    }
}

/// Optionally render input via chat template, tokenize, and truncate.
fn prepare_input(
    inputs: String,
    truncate: Option<usize>,
    chat_messages: Option<Vec<ChatMessage>>,
    tokenizer: &TokenizerRender,
    chat_renderer: &mut ChatRenderer,
) -> Result<(Vec<ChatMessage>, String, usize, Vec<u32>), ValidationError> {
    // Use provided chat messages (from /v1/chat/completions) or wrap raw input
    let messages = chat_messages.unwrap_or_else(|| {
        vec![ChatMessage { role: "user".to_string(), content: inputs }]
    });

    // Step 1: Render chat template (or pass through if ChatRenderer::None)
    let inputs = chat_renderer.render(&messages)?;

    // Step 2: Encode to token IDs (always Rust tokenizers crate)
    let mut encoding = tokenizer
        .tokenizer
        .encode(inputs.clone(), false)
        .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;

    // Optionally truncate
    let (inputs, input_length, input_tokens) = match truncate {
        // Truncate is some and < encoding length
        Some(truncate) if truncate < encoding.len() => {
            // truncate encoding and decode new inputs
            encoding.truncate(truncate, 0, TruncationDirection::Left);
            let inputs = tokenizer
                .tokenizer
                .decode(encoding.get_ids(), false)
                .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;
            (inputs, encoding.len(), encoding.get_ids().to_vec())
        }
        // Nothing to do
        _ => (inputs, encoding.len(), encoding.get_ids().to_vec()),
    };

    Ok((messages, inputs, input_length, input_tokens))
}

type TokenizerRequest = (
    (String, Option<usize>, Option<Vec<ChatMessage>>),
    oneshot::Sender<Result<(Vec<ChatMessage>, String, usize, Vec<u32>), ValidationError>>,
    Span,
);

#[derive(Debug)]
#[allow(dead_code)] // request fields carried for tracing/debug; Debug derive references them
pub(crate) struct ValidGenerateRequest {
    pub request_id: u64,
    /// Raw user input prompt
    pub messages: Vec<ChatMessage>,
    /// NOTE: all fields begin with `input` are rendered
    pub inputs: String,
    /// Immutable field
    pub input_length: u32,
    pub truncate: u32,
    pub decoder_input_details: bool,
    pub parameters: NextTokenChooserParameters,
    pub stopping_parameters: StoppingCriteriaParameters,
    pub top_n_tokens: u32,
    pub input_tokens: Vec<u32>,
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("`best_of` must be > 0 and <= {0}. Given: {1}")]
    BestOf(usize, usize),
    #[error("`best_of` != 1 is not allowed for this endpoint")]
    BestOfDisabled,
    #[error("you must use sampling when `best_of` is > 1")]
    BestOfSampling,
    #[error("`seed` must not be set when `best_of` > 1")]
    BestOfSeed,
    #[error("`best_of` != 1 is not supported when streaming tokens")]
    BestOfStream,
    #[error("`top_n_tokens` must be >= 0 and <= {0}. Given: {1}")]
    TopNTokens(u32, u32),
    #[error("`top_n_tokens` != 0 is not allowed for this endpoint")]
    TopNTokensDisabled,
    #[error("`decoder_input_details` == true is not supported when streaming tokens")]
    PrefillDetailsStream,
    #[error("`temperature` must be strictly positive")]
    Temperature,
    #[error("`repetition_penalty` must be strictly positive")]
    RepetitionPenalty,
    #[error("`top_p` must be > 0.0 and < 1.0")]
    TopP,
    #[error("`top_k` must be strictly positive")]
    TopK,
    #[error("`truncate` must be strictly positive and less than {0}. Given: {1}")]
    Truncate(usize, usize),
    #[error("`typical_p` must be > 0.0 and < 1.0")]
    TypicalP,
    #[error("one of `max_new_tokens` or `truncate` must be set if a fast tokenizer is not in use")]
    UnsetMaxNewTokens,
    #[error("`max_new_tokens` must be strictly positive")]
    NegativeMaxNewTokens,
    #[error("`max_new_tokens` must be <= {0}. Given: {1}")]
    MaxNewTokens(usize, u32),
    #[error("`inputs` tokens + `max_new_tokens` must be <= {0}. Given: {1} `inputs` tokens and {2} `max_new_tokens`")]
    MaxTotalTokens(usize, usize, u32),
    #[error("`inputs` must have less than {0} tokens. Given: {1}")]
    InputLength(usize, usize),
    #[error("`inputs` cannot be empty")]
    EmptyInput,
    #[error("`stop` supports up to {0} stop sequences. Given: {1}")]
    StopSequence(usize, usize),
    #[error("tokenizer error {0}")]
    Tokenizer(String),
}
