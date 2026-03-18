// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// This file is a **modified** version of
// text-generation-inference/src/token_stream.rs
// © 2022-present Hugging Face Inc. – Apache-2.0.
use std::fs::File;
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::time::Duration;

use axum::http::HeaderValue;
use clap::Parser;
use opentelemetry::sdk::propagation::TraceContextPropagator;
use opentelemetry::sdk::trace;
use opentelemetry::sdk::trace::Sampler;
use opentelemetry::sdk::Resource;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use router::error::ClientError;
use router::{server, HubModelInfo, TokenizerRender};
#[cfg(feature = "vllm-backend")]
use router::VllmClient;
use router::engine_client::EngineClient;
use thiserror::Error;
#[allow(unused_imports)]
use tokenizers::{FromPretrainedParameters, Tokenizer};
use tower_http::cors::AllowOrigin;
use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = "2", long, env)]
    max_best_of: usize,
    #[clap(default_value = "4", long, env)]
    max_stop_sequences: usize,
    #[clap(default_value = "5", long, env)]
    max_top_n_tokens: u32,
    #[clap(default_value = "1024", long, env)]
    max_input_length: usize,
    #[clap(default_value = "2048", long, env)]
    max_total_tokens: usize,
    #[clap(default_value = "4096", long, env)]
    max_batch_prefill_tokens: u32,
    #[clap(long, env)]
    max_batch_total_tokens: Option<u32>,
    #[clap(default_value = "0.0.0.0", long, env)]
    hostname: String,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(long)]
    client_config: String,
    #[clap(default_value = "bigscience/bloom", long, env)]
    tokenizer_name: String,
    #[clap(default_value_t = false, long)]
    use_tokenizer: bool,
    #[clap(long, env)]
    revision: Option<String>,
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,
    #[clap(long, env)]
    json_output: bool,
    #[clap(long, env)]
    otlp_endpoint: Option<String>,
    #[clap(long, env)]
    cors_allow_origin: Option<Vec<String>>,
    #[clap(long, env)]
    ngrok: bool,
    #[clap(long, env)]
    ngrok_authtoken: Option<String>,
    #[clap(long, env)]
    ngrok_edge: Option<String>,
    #[clap(long, env)]
    log_path: Option<String>,
    #[clap(long, env)]
    statistic_path: Option<String>,

    #[clap(long, required = true)]
    model_name: String,

    #[clap(long, default_value_t = 16)]
    kvcache_block_size: usize,
}

fn main() -> Result<(), RouterError> {
    // Get args
    let args = Args::parse();
    // Pattern match configuration
    let Args {
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_top_n_tokens,
        max_input_length,
        max_total_tokens,
        max_batch_prefill_tokens,
        max_batch_total_tokens,
        kvcache_block_size,
        hostname,
        port,
        client_config,
        tokenizer_name,
        use_tokenizer,
        revision,
        validation_workers,
        json_output,
        otlp_endpoint,
        cors_allow_origin,
        ngrok,
        ngrok_authtoken,
        ngrok_edge,
        log_path,
        statistic_path,
        model_name,
    } = args;

    // Validate args
    if max_input_length >= max_total_tokens {
        return Err(RouterError::ArgumentValidation(
            "`max_input_length` must be < `max_total_tokens`".to_string(),
        ));
    }
    if max_input_length as u32 > max_batch_prefill_tokens {
        return Err(RouterError::ArgumentValidation(format!("`max_batch_prefill_tokens` must be >= `max_input_length`. Given: {max_batch_prefill_tokens} and {max_input_length}")));
    }

    if validation_workers == 0 {
        return Err(RouterError::ArgumentValidation(
            "`validation_workers` must be > 0".to_string(),
        ));
    }

    if let Some(ref max_batch_total_tokens) = max_batch_total_tokens {
        if max_batch_prefill_tokens > *max_batch_total_tokens {
            return Err(RouterError::ArgumentValidation(format!("`max_batch_prefill_tokens` must be <= `max_batch_total_tokens`. Given: {max_batch_prefill_tokens} and {max_batch_total_tokens}")));
        }
        if max_total_tokens as u32 > *max_batch_total_tokens {
            return Err(RouterError::ArgumentValidation(format!("`max_total_tokens` must be <= `max_batch_total_tokens`. Given: {max_total_tokens} and {max_batch_total_tokens}")));
        }
    }

    // CORS allowed origins
    let cors_allow_origin: Option<AllowOrigin> = cors_allow_origin.map(|cors_allow_origin| {
        AllowOrigin::list(
            cors_allow_origin.iter().map(|origin| origin.parse::<HeaderValue>().unwrap()),
        )
    });

    // Parse Huggingface hub token
    let authorization_token = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();

    // Tokenizer instance
    let local_path = Path::new(&tokenizer_name);
    let local_model = local_path.exists() && local_path.is_dir();
    let tokenizer = if use_tokenizer {
        if local_model {
            Some(TokenizerRender::new(local_path))
        } else {
            unreachable!("Unexisted path {} to tokenizer!", tokenizer_name);
            #[allow(unreachable_code)]
            {
                let _params = FromPretrainedParameters {
                    revision: revision.clone().unwrap_or("main".to_string()),
                    ..Default::default()
                };
                None
            }
        }
    } else {
        None
    };

    let server_future = async {
        let _guard = init_logging(otlp_endpoint, json_output, log_path);

        if tokenizer.is_none() {
            tracing::warn!("Could not find a fast tokenizer implementation for {tokenizer_name}");
            tracing::warn!("Rust input length validation and truncation is disabled");
        }

        // Get Model info
        let model_info = match local_model {
            true => {
                HubModelInfo { model_id: tokenizer_name.clone(), sha: None, pipeline_tag: None }
            }
            false => get_model_info(&tokenizer_name, revision, authorization_token)
                .await
                .unwrap_or_else(|| {
                    tracing::warn!("Could not retrieve model info from the Hugging Face hub.");
                    HubModelInfo {
                        model_id: tokenizer_name.to_string(),
                        sha: None,
                        pipeline_tag: None,
                    }
                }),
        };

        // if pipeline-tag == text-generation we default to return_full_text = true
        let compat_return_full_text = match &model_info.pipeline_tag {
            None => {
                tracing::warn!("no pipeline tag found for model {tokenizer_name}");
                false
            }
            Some(pipeline_tag) => pipeline_tag.as_str() == "text-generation",
        };

        // Read uris from client_config
        let mut buf = String::new();
        File::open(client_config).unwrap().read_to_string(&mut buf).unwrap();

        #[cfg(feature = "vllm-backend")]
        let engine_clients: Vec<Box<dyn EngineClient>> = {
            use router::engine_client::VllmEngineClient;
            serde_json::from_str::<Vec<String>>(buf.as_str())
                .unwrap()
                .into_iter()
                .map(|uri| {
                    let vllm_client = VllmClient::new(uri.as_str(), &model_name);
                    Box::new(VllmEngineClient::new(vllm_client)) as Box<dyn EngineClient>
                })
                .collect()
        };

        // ZMQ backend: create engine clients from IPC/TCP socket addresses.
        #[cfg(feature = "zmq-backend")]
        let engine_clients: Vec<Box<dyn EngineClient>> = {
            use router::engine_client::ZmqEngineClientAdapter;
            use router::zmq_engine::ZmqEngineClient;

            let addr_pairs: Vec<(String, String)> =
                serde_json::from_str::<Vec<Vec<String>>>(buf.as_str())
                    .expect("ZMQ backend expects JSON array of [input_addr, output_addr] pairs")
                    .into_iter()
                    .map(|pair| {
                        assert_eq!(pair.len(), 2, "Each ZMQ address entry must be [input, output]");
                        (pair[0].clone(), pair[1].clone())
                    })
                    .collect();

            let mut clients = Vec::with_capacity(addr_pairs.len());
            for (input_addr, output_addr) in addr_pairs {
                let mut zmq_client = ZmqEngineClient::new();
                zmq_client
                    .connect(&input_addr, &output_addr)
                    .await
                    .expect(&format!(
                        "Failed to connect ZMQ engine client to {} / {}",
                        input_addr, output_addr
                    ));
                clients.push(
                    Box::new(ZmqEngineClientAdapter::new(zmq_client)) as Box<dyn EngineClient>,
                );
            }
            clients
        };

        // Get info from the shard
        tracing::warn!("The shard info is not set properly by the st-server");
        let shard_info = Default::default();

        let addr = match hostname.parse() {
            Ok(ip) => SocketAddr::new(ip, port),
            Err(_) => {
                tracing::warn!("Invalid hostname, defaulting to 0.0.0.0");
                SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port)
            }
        };

        let max_supported_batch_total_tokens = 16000;

        // Run server
        server::run(
            model_info,
            shard_info,
            compat_return_full_text,
            max_concurrent_requests,
            max_best_of,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            max_batch_prefill_tokens,
            max_supported_batch_total_tokens,
            engine_clients,
            kvcache_block_size,
            tokenizer,
            validation_workers,
            addr,
            cors_allow_origin,
            ngrok,
            ngrok_authtoken,
            ngrok_edge,
            statistic_path,
        )
        .await?;
        Ok(())
    };

    // Launch Tokio runtime
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(server_future)
}

/// Init logging using env variables LOG_LEVEL and LOG_FORMAT:
///     - otlp_endpoint is an optional URL to an Open Telemetry collector
///     - LOG_LEVEL may be TRACE, DEBUG, INFO, WARN or ERROR (default to INFO)
///     - LOG_FORMAT may be TEXT or JSON (default to TEXT)
fn init_logging(
    otlp_endpoint: Option<String>,
    json_output: bool,
    log_path: Option<String>,
) -> Option<WorkerGuard> {
    let mut layers = Vec::new();
    // STDOUT/STDERR layer
    let fmt_layer = tracing_subscriber::fmt::layer().with_file(true).with_line_number(true);

    let guard = match log_path {
        Some(path) => {
            let (non_blocking, guard) = NonBlocking::new(std::fs::File::create(path).unwrap());
            let fmt_layer = fmt_layer.with_ansi(false).with_writer(non_blocking);
            let fmt_layer = match json_output {
                true => fmt_layer.json().flatten_event(true).boxed(),
                false => fmt_layer.boxed(),
            };
            layers.push(fmt_layer);
            Some(guard)
        }
        None => {
            let fmt_layer = match json_output {
                true => fmt_layer.json().flatten_event(true).boxed(),
                false => fmt_layer.boxed(),
            };
            layers.push(fmt_layer);
            Option::None
        }
    };

    // OpenTelemetry tracing layer
    if let Some(otlp_endpoint) = otlp_endpoint {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(opentelemetry_otlp::new_exporter().tonic().with_endpoint(otlp_endpoint))
            .with_trace_config(
                trace::config()
                    .with_resource(Resource::new(vec![KeyValue::new(
                        "service.name",
                        "blitz.router",
                    )]))
                    .with_sampler(Sampler::AlwaysOn),
            )
            .install_batch(opentelemetry::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            init_tracing_opentelemetry::init_propagator().unwrap();
        } else {
            panic!("Failed to install OTEL pipeline, error: {:?}", tracer);
        }
    }

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry().with(env_filter).with(layers).init();
    return guard;
}

/// get model info from the Huggingface Hub
pub async fn get_model_info(
    model_id: &str,
    revision: Option<String>,
    token: Option<String>,
) -> Option<HubModelInfo> {
    let revision = match revision {
        None => {
            tracing::warn!("`--revision` is not set");
            tracing::warn!("We strongly advise to set it to a known supported commit.");
            "main".to_string()
        }
        Some(revision) => revision,
    };

    let client = reqwest::Client::new();
    // Poor man's urlencode
    let revision = revision.replace('/', "%2F");
    let url = format!("https://huggingface.co/api/models/{model_id}/revision/{revision}");
    let mut builder = client.get(url).timeout(Duration::from_secs(5));
    if let Some(token) = token {
        builder = builder.bearer_auth(token);
    }

    let response = builder.send().await.ok()?;

    if response.status().is_success() {
        let hub_model_info: HubModelInfo =
            serde_json::from_str(&response.text().await.ok()?).ok()?;
        if let Some(sha) = &hub_model_info.sha {
            tracing::info!("Serving revision {sha} of model {}", hub_model_info.model_id);
        }
        Some(hub_model_info)
    } else {
        None
    }
}

#[allow(unused)]
#[derive(Debug, Error)]
enum RouterError {
    #[error("Argument validation error: {0}")]
    ArgumentValidation(String),
    #[error("Unable to connect to the Python model shards: {0}")]
    Connection(ClientError),
    #[error("Unable to clear the Python model shards cache: {0}")]
    Cache(ClientError),
    #[error("Unable to get the Python model shards info: {0}")]
    Info(ClientError),
    #[error("Unable to warmup the Python model shards: {0}")]
    Warmup(ClientError),
    #[error("Tokio runtime failed to start: {0}")]
    Tokio(#[from] std::io::Error),
    #[error("Axum webserver failed: {0}")]
    Axum(#[from] axum::BoxError),
}
