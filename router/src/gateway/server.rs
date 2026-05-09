// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// This file is a **modified** version of
// text-generation-inference/src/token_stream.rs
// © 2022-present Hugging Face Inc. – Apache-2.0.
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::health::Health;
use crate::scheduler::infer::{InferError, InferStreamResponse};
use crate::engine::EngineClient;
use crate::{
    ChatRenderer, ChatMessage, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionChoice, ChatCompletionUsage, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionDelta, ErrorResponse,
    GenerateParameters, GenerateRequest, HubModelInfo, Infer, Info,
    Token, TokenizerRender, Validation,
    default_parameters,
};

use axum::extract::Extension;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{http, Json, Router};
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;
use futures::stream::StreamExt;
use futures::Stream;
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use crate::types::InfoResponse;
use tokio::signal;
use tokio::time::Instant;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::instrument;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// Tombstone for the TGI-style legacy endpoints (`/`, `/generate`,
/// `/generate_stream`, `/invocations`). Returns HTTP 410 Gone with a JSON
/// body that points callers at the canonical OpenAI-compatible endpoint.
///
/// Rationale: these endpoints were inherited from upstream
/// text-generation-inference. The router now exclusively serves the
/// OpenAI chat-completions surface (`POST /v1/chat/completions`); the
/// legacy URLs are kept only so misdirected clients get a loud,
/// actionable error instead of a generic 404.
async fn tgi_deprecated() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::GONE,
        Json(ErrorResponse {
            error: concat!(
                "This endpoint is deprecated and no longer served. ",
                "Use `POST /v1/chat/completions` (OpenAI-compatible) instead. ",
                "See `docs/architecture/README.md` §1 for the supported API surface."
            )
            .to_string(),
            error_type: "deprecated_endpoint".to_string(),
        }),
    )
}




/// Blitz endpoint info
#[utoipa::path(
get,
tag = "Blitz",
path = "/info",
responses((status = 200, description = "Served model info", body = Info))
)]
#[instrument]
async fn get_model_info(info: Extension<Info>) -> Json<Info> {
    Json(info.0)
}

#[utoipa::path(
get,
tag = "Blitz",
path = "/health",
responses(
(status = 200, description = "Everything is working fine"),
(status = 503, description = "Blitz is down", body = ErrorResponse,
example = json ! ({"error": "unhealthy", "error_type": "healthcheck"})),
)
)]
#[instrument(skip_all)]
/// Health check method
async fn health(mut health: Extension<Health>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    match health.check().await {
        true => Ok(()),
        false => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "unhealthy".to_string(),
                error_type: "healthcheck".to_string(),
            }),
        )),
    }
}



/// Prometheus metrics scrape endpoint
#[utoipa::path(
get,
tag = "Blitz",
path = "/metrics",
responses((status = 200, description = "Prometheus Metrics", body = String))
)]
async fn metrics(prom_handle: Extension<PrometheusHandle>) -> String {
    prom_handle.render()
}

// ---------------------------------------------------------------------------
// OpenAI-compatible /v1/chat/completions endpoint
// ---------------------------------------------------------------------------

/// Convert a protobuf FinishReason (i32) to OpenAI-compatible string.
fn finish_reason_to_openai(reason: i32) -> String {
    match crate::types::FinishReason::try_from(reason) {
        Ok(crate::types::FinishReason::Length) => "length".to_string(),
        _ => "stop".to_string(),
    }
}

/// OpenAI-compatible /v1/chat/completions endpoint.
///
/// Accepts `{"messages": [...], "max_tokens": N, "stream": bool}` and
/// dispatches through the same inference pipeline as `/generate`.
/// Chat template rendering is handled by the tokenizer workers using
/// the messages array directly, ensuring correct multi-turn prompt formatting.
#[instrument(skip_all)]
async fn chat_completions(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let stream = req.stream.unwrap_or(false);
    let model_id = req.model.clone().unwrap_or_else(|| info.model_id.clone());

    // Build a GenerateRequest with the messages passed through to the tokenizer worker.
    // The `inputs` field is a placeholder (last message content) for logging/validation;
    // the actual prompt is rendered from `chat_messages` by the tokenizer worker's chat
    // template renderer.
    let inputs_placeholder = req
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let gen_req = GenerateRequest {
        inputs: inputs_placeholder,
        parameters: GenerateParameters {
            temperature: req.temperature,
            repetition_penalty: req.repetition_penalty,
            top_p: req.top_p,
            max_new_tokens: req.max_tokens,
            stop: req.stop.map(|s| s.into_vec()).unwrap_or_default(),
            seed: req.seed,
            ..default_parameters()
        },
        chat_messages: Some(req.messages),
    };

    if stream {
        Ok(chat_completions_stream(infer, model_id, gen_req).await.into_response())
    } else {
        chat_completions_non_stream(infer, model_id, gen_req).await
    }
}

/// Non-streaming chat completions: run inference and return a single JSON response.
#[instrument(skip_all)]
async fn chat_completions_non_stream(
    Extension(infer): Extension<Infer>,
    model_id: String,
    req: GenerateRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let start_time = Instant::now();
    metrics::increment_counter!("blitz_request_count");

    let response = infer.generate(req).await?;

    let total_time = start_time.elapsed();
    metrics::increment_counter!("blitz_request_success");
    metrics::histogram!("blitz_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "blitz_request_generated_tokens",
        response.generated_text.generated_tokens as f64
    );

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let finish_reason = finish_reason_to_openai(response.generated_text.finish_reason);
    let completion_tokens = response.generated_text.generated_tokens;

    let mut headers = HeaderMap::new();
    headers.insert("x-request-id", response.request_id.to_string().parse().unwrap());

    let chat_response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", response.request_id),
        object: "chat.completion".to_string(),
        created,
        model: model_id,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response.generated_text.text,
            },
            finish_reason: Some(finish_reason),
        }],
        usage: ChatCompletionUsage {
            prompt_tokens: response.input_length as u32,
            completion_tokens,
            total_tokens: response.input_length as u32 + completion_tokens,
        },
    };

    Ok((headers, Json(chat_response)).into_response())
}

/// Streaming chat completions: return SSE events in OpenAI chunk format.
#[instrument(skip_all)]
async fn chat_completions_stream(
    Extension(infer): Extension<Infer>,
    model_id: String,
    req: GenerateRequest,
) -> (HeaderMap, Sse<impl Stream<Item = Result<Event, Infallible>>>) {
    let start_time = Instant::now();
    metrics::increment_counter!("blitz_request_count");

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut headers = HeaderMap::new();
    headers.insert("X-Accel-Buffering", "no".parse().unwrap());
    headers.insert("content-type", "text/event-stream".parse().unwrap());

    let stream = async_stream::stream! {
        let mut end_reached = false;
        let mut error = false;

        match infer.generate_stream(req).await {
            Ok((request_id, _permit, mut response_stream)) => {
                let chat_id = format!("chatcmpl-{}", request_id);

                // First chunk: role announcement
                let first_chunk = ChatCompletionChunk {
                    id: chat_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionDelta {
                            role: Some("assistant".to_string()),
                            content: Some(String::new()),
                        },
                        finish_reason: None,
                    }],
                };
                yield Ok(Event::default().json_data(first_chunk).unwrap());

                while let Some(response) = response_stream.next().await {
                    match response {
                        Ok(response) => {
                            match response {
                                InferStreamResponse::Prefill => {}
                                InferStreamResponse::Intermediate { token, .. } => {
                                    let chunk = ChatCompletionChunk {
                                        id: chat_id.clone(),
                                        object: "chat.completion.chunk".to_string(),
                                        created,
                                        model: model_id.clone(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionDelta {
                                                role: None,
                                                content: Some(token.text),
                                            },
                                            finish_reason: None,
                                        }],
                                    };
                                    yield Ok(Event::default().json_data(chunk).unwrap());
                                }
                                InferStreamResponse::End {
                                    token,
                                    generated_text,
                                    ..
                                } => {
                                    end_reached = true;

                                    // Timings / metrics
                                    let total_time = start_time.elapsed();
                                    metrics::increment_counter!("blitz_request_success");
                                    metrics::histogram!("blitz_request_duration", total_time.as_secs_f64());
                                    metrics::histogram!(
                                        "blitz_request_generated_tokens",
                                        generated_text.generated_tokens as f64
                                    );

                                    // Emit last token content (if non-empty)
                                    if !token.text.is_empty() {
                                        let chunk = ChatCompletionChunk {
                                            id: chat_id.clone(),
                                            object: "chat.completion.chunk".to_string(),
                                            created,
                                            model: model_id.clone(),
                                            choices: vec![ChatCompletionChunkChoice {
                                                index: 0,
                                                delta: ChatCompletionDelta {
                                                    role: None,
                                                    content: Some(token.text),
                                                },
                                                finish_reason: None,
                                            }],
                                        };
                                        yield Ok(Event::default().json_data(chunk).unwrap());
                                    }

                                    // Emit finish chunk
                                    let finish_reason = finish_reason_to_openai(
                                        generated_text.finish_reason,
                                    );
                                    let finish_chunk = ChatCompletionChunk {
                                        id: chat_id.clone(),
                                        object: "chat.completion.chunk".to_string(),
                                        created,
                                        model: model_id.clone(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionDelta {
                                                role: None,
                                                content: None,
                                            },
                                            finish_reason: Some(finish_reason),
                                        }],
                                    };
                                    yield Ok(Event::default().json_data(finish_chunk).unwrap());

                                    // Emit [DONE] sentinel
                                    yield Ok(Event::default().data("[DONE]"));
                                    break;
                                }
                            }
                        }
                        Err(err) => {
                            error = true;
                            yield Ok(Event::from(err));
                            break;
                        }
                    }
                }
            }
            Err(err) => {
                error = true;
                yield Ok(Event::from(err));
            }
        }

        if !end_reached && !error {
            let err = InferError::IncompleteGeneration;
            metrics::increment_counter!("blitz_request_failure", "err" => "incomplete");
            tracing::error!("{err}");
            yield Ok(Event::from(err));
        }
    };

    (headers, Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Serving method
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_info: HubModelInfo,
    shard_info: InfoResponse,
    max_concurrent_requests: usize,
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_length: usize,
    max_total_tokens: usize,
    engine_clients: Vec<Box<dyn EngineClient>>,
    kvcache_block_size: usize,
    tokenizer: Option<TokenizerRender>,
    chat_renderer: ChatRenderer,
    shared_tokenizer: Option<Arc<tokenizers::Tokenizer>>,
    validation_workers: usize,
    addr: SocketAddr,
    allow_origin: Option<AllowOrigin>,
    ngrok: bool,
    ngrok_authtoken: Option<String>,
    ngrok_edge: Option<String>,
    statistic_path: Option<String>,
) -> Result<(), axum::BoxError> {
    // OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
    paths(
    health,
    get_model_info,
    metrics,
    ),
    components(
    schemas(
    Info,
    GenerateRequest,
    GenerateParameters,
    Token,
    ErrorResponse,
    )
    ),
    tags(
    (name = "Blitz", description = "Hugging Face Blitz API")
    ),
    info(
    title = "Blitz",
    license(
    name = "Apache 2.0",
    url = "https://www.apache.org/licenses/LICENSE-2.0"
    )
    )
    )]
    struct ApiDoc;

    // Create state
    let validation = Validation::new(
        validation_workers,
        tokenizer,
        chat_renderer,
        max_best_of,
        max_stop_sequences,
        max_top_n_tokens,
        max_input_length,
        max_total_tokens,
    );
    let health_ext = Health::new();

    let infer = Infer::create_vllm_colocation(
        engine_clients,
        kvcache_block_size,
        validation,
        max_concurrent_requests,
        statistic_path,
        shared_tokenizer,
    );

    println!("Blitz router is ready");

    // Duration buckets
    let duration_matcher = Matcher::Suffix(String::from("duration"));
    let n_duration_buckets = 35;
    let mut duration_buckets = Vec::with_capacity(n_duration_buckets);
    // Minimum duration in seconds
    let mut value = 0.0001;
    for _ in 0..n_duration_buckets {
        // geometric sequence
        value *= 1.5;
        duration_buckets.push(value);
    }
    // Input Length buckets
    let input_length_matcher = Matcher::Full(String::from("blitz_request_input_length"));
    let input_length_buckets: Vec<f64> =
        (0..100).map(|x| (max_input_length as f64 / 100.0) * (x + 1) as f64).collect();
    // Generated tokens buckets
    let generated_tokens_matcher = Matcher::Full(String::from("blitz_request_generated_tokens"));
    let generated_tokens_buckets: Vec<f64> =
        (0..100).map(|x| (max_total_tokens as f64 / 100.0) * (x + 1) as f64).collect();
    // Input Length buckets
    let max_new_tokens_matcher = Matcher::Full(String::from("blitz_request_max_new_tokens"));
    let max_new_tokens_buckets: Vec<f64> =
        (0..100).map(|x| (max_total_tokens as f64 / 100.0) * (x + 1) as f64).collect();
    // Batch size buckets
    let batch_size_matcher = Matcher::Full(String::from("blitz_batch_next_size"));
    let batch_size_buckets: Vec<f64> = (0..1024).map(|x| (x + 1) as f64).collect();
    // Speculated tokens buckets
    let skipped_matcher = Matcher::Full(String::from("blitz_request_skipped_tokens"));
    let skipped_buckets: Vec<f64> = (0..shard_info.speculate + 1).map(|x| x as f64).collect();

    // Prometheus handler
    let builder = PrometheusBuilder::new()
        .set_buckets_for_metric(duration_matcher, &duration_buckets)
        .unwrap()
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)
        .unwrap()
        .set_buckets_for_metric(generated_tokens_matcher, &generated_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(max_new_tokens_matcher, &max_new_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)
        .unwrap()
        .set_buckets_for_metric(skipped_matcher, &skipped_buckets)
        .unwrap();
    let prom_handle = builder.install_recorder().expect("failed to install metrics recorder");

    describe_metric();
    // CORS layer
    let allow_origin = allow_origin.unwrap_or(AllowOrigin::any());
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    // Endpoint info
    let info = Info {
        model_id: model_info.model_id,
        model_sha: model_info.sha,
        model_dtype: shard_info.dtype,
        model_device_type: shard_info.device_type,
        model_pipeline_tag: model_info.pipeline_tag,
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
        validation_workers,
        version: env!("CARGO_PKG_VERSION"),
        sha: option_env!("VERGEN_GIT_SHA"),
        docker_label: option_env!("DOCKER_LABEL"),
    };

    // Create router
    let app = Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()))
        // Base routes
        .route("/", post(tgi_deprecated))
        .route("/info", get(get_model_info))
        .route("/generate", post(tgi_deprecated))
        .route("/generate_stream", post(tgi_deprecated))
        // OpenAI-compatible chat completions
        .route("/v1/chat/completions", post(chat_completions))
        // AWS Sagemaker route
        .route("/invocations", post(tgi_deprecated))
        // Base Health route
        .route("/health", get(health))
        // Inference API health route
        .route("/", get(health))
        // AWS Sagemaker health route
        .route("/ping", get(health))
        // Prometheus metrics route
        .route("/metrics", get(metrics))
        .layer(Extension(info))
        .layer(Extension(health_ext.clone()))
        .layer(Extension(infer))
        .layer(Extension(prom_handle.clone()))
        .layer(OtelAxumLayer::default())
        .layer(cors_layer);

    if ngrok {
        #[cfg(feature = "ngrok")]
        {
            use ngrok::config::TunnelBuilder;

            let _ = addr;

            let authtoken =
                ngrok_authtoken.expect("`ngrok-authtoken` must be set when using ngrok tunneling");

            let edge = ngrok_edge.expect("`ngrok-edge` must be set when using ngrok tunneling");

            let tunnel = ngrok::Session::builder()
                .authtoken(authtoken)
                .connect()
                .await
                .unwrap()
                .labeled_tunnel()
                .label("edge", edge);

            let listener = tunnel.listen().await.unwrap();

            // Run prom metrics and health locally too
            tokio::spawn(
                axum::Server::bind(&addr)
                    .serve(
                        Router::new()
                            .route("/health", get(health))
                            .route("/metrics", get(metrics))
                            .layer(Extension(health_ext))
                            .layer(Extension(prom_handle))
                            .into_make_service(),
                    )
                    //Wait until all requests are finished to shut down
                    .with_graceful_shutdown(shutdown_signal()),
            );

            // Run server
            axum::Server::builder(listener)
                .serve(app.into_make_service())
                //Wait until all requests are finished to shut down
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
        #[cfg(not(feature = "ngrok"))]
        {
            let _ngrok_authtoken = ngrok_authtoken;
            let _ngrok_edge = ngrok_edge;
            panic!("`blitz-router` was compiled without the `ngrok` feature");
        }
    } else {
        // Run server
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            // Wait until all requests are finished to shut down
            .with_graceful_shutdown(shutdown_signal())
            .await?;
    }
    Ok(())
}

/// Shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("signal received, starting graceful shutdown");
    opentelemetry::global::shutdown_tracer_provider();
}

/// Convert to Axum supported formats
impl From<InferError> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: InferError) -> Self {
        let status_code = match err {
            InferError::GenerationError(_) => StatusCode::FAILED_DEPENDENCY,
            InferError::Overloaded(_) => StatusCode::TOO_MANY_REQUESTS,
            InferError::ValidationError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            InferError::IncompleteGeneration => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (
            status_code,
            Json(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            }),
        )
    }
}

impl From<InferError> for Event {
    fn from(err: InferError) -> Self {
        Event::default()
            .json_data(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            })
            .unwrap()
    }
}

fn describe_metric() {
    metrics::describe_counter!("blitz_request_count", "Total number of requests received");
    metrics::describe_counter!("blitz_request_success", "Total number of successful requests");
    metrics::describe_counter!(
        "blitz_request_failure",
        "Total number of failed requests, labeled by error type"
    );
    metrics::describe_gauge!("blitz_queue_size", "Current size of the request queue");
    metrics::describe_histogram!(
        "blitz_request_duration",
        "Total time taken to process a request (seconds)"
    );
    metrics::describe_histogram!(
        "blitz_request_validation_duration",
        "Time spent on request validation (seconds)"
    );
    metrics::describe_histogram!(
        "blitz_request_queue_duration",
        "Time spent in the request queue (seconds)"
    );
    metrics::describe_histogram!(
        "blitz_request_inference_duration",
        "Time spent on inference (seconds)"
    );
    metrics::describe_histogram!(
        "blitz_request_mean_time_per_token_duration",
        "Mean time per generated token (seconds)"
    );
    metrics::describe_histogram!(
        "blitz_request_generated_tokens",
        "Number of tokens generated per request"
    );
    metrics::describe_histogram!(
        "blitz_request_input_length",
        "Input length (number of tokens) per request"
    );
    metrics::describe_histogram!(
        "blitz_request_max_new_tokens",
        "Max new tokens requested per request"
    );
    metrics::describe_histogram!("blitz_batch_next_size", "Batch size for next batch");
    metrics::describe_histogram!(
        "blitz_request_skipped_tokens",
        "Number of speculated/skipped tokens per request"
    );
}
