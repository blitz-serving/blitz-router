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
use router::{server, ChatRenderer, HubModelInfo, TokenizerRender};
use std::sync::Arc;
use router::model_config::load_model_config;
#[cfg(feature = "vllm-backend")]
use router::VllmClient;
use router::engine_client::EngineClient;
use thiserror::Error;
#[allow(unused_imports)]
use tokenizers::Tokenizer;
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

    /// Chat template rendering mode: "none" (inner cluster, pre-rendered prompts)
    /// or "python" (embed Python jinja2 via PyO3, requires feature python-chat-template).
    #[clap(long, env, default_value = "none")]
    chat_template_mode: String,

    // ---- Latency simulator (feature `simulator`, piggyback observer) ---- //
    /// Enable the latency simulator subsystem (requires --features simulator).
    /// When enabled, observes admissions made by the active <name>-q policy
    /// and emits predicted-vs-actual histograms via the metrics endpoint.
    /// Does not influence routing.
    #[clap(long, env, default_value_t = false)]
    enable_simulator: bool,
    /// Directory containing precomputed Vidur prediction grids
    /// ({op}_{model_hash}_predictions.csv).
    #[clap(long, env, default_value = "/nvme/zkx/Modified_vidur/cache")]
    simulator_cache_dir: String,
    /// Vidur model hash. Qwen2.5: 9f4b3b9a, Llama3: d29f0375.
    #[clap(long, env, default_value = "9f4b3b9a")]
    simulator_model_hash: String,
    /// Number of transformer layers in the served model.
    #[clap(long, env, default_value_t = 28)]
    simulator_num_layers: usize,
    /// Treat the served model as Mixture-of-Experts (uses moe_linear grid
    /// instead of the dense MLP grids).
    #[clap(long, env, default_value_t = false)]
    simulator_moe: bool,
    /// Online linreg correction learning rate.
    #[clap(long, env, default_value_t = 1e-6)]
    simulator_learning_rate: f32,
    /// Reject calibration samples whose |actual - corrected| exceeds this
    /// many milliseconds. Set high (e.g. 10000) when starting from a stopgap
    /// grid that's way off, low (e.g. 5) when the grid is well-calibrated.
    #[clap(long, env, default_value_t = 10000.0)]
    simulator_outlier_threshold_ms: f32,
}

fn main() -> Result<(), RouterError> {
    // Get args
    let args = Args::parse();
    // Pattern match configuration
    let Args {
        mut max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_top_n_tokens,
        mut max_input_length,
        mut max_total_tokens,
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
        chat_template_mode,
        enable_simulator,
        simulator_cache_dir,
        simulator_model_hash,
        simulator_num_layers,
        simulator_moe,
        simulator_learning_rate,
        simulator_outlier_threshold_ms,
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

    // Tokenizer instance (encoding only — no chat template)
    let local_path = Path::new(&tokenizer_name);
    let local_model = local_path.exists() && local_path.is_dir();
    let tokenizer = if use_tokenizer {
        if local_model {
            Some(TokenizerRender::new(local_path))
        } else {
            return Err(RouterError::ArgumentValidation(format!(
                "Tokenizer path does not exist: {tokenizer_name}"
            )));
        }
    } else {
        None
    };

    let shared_tokenizer: Option<Arc<tokenizers::Tokenizer>> = tokenizer
        .as_ref()
        .map(|tr| Arc::new(tr.tokenizer.clone()));

    // Chat template renderer (decoupled from tokenizer)
    let chat_renderer = match chat_template_mode.as_str() {
        "none" => ChatRenderer::None,
        #[cfg(feature = "python-chat-template")]
        "python" => {
            if !local_model {
                return Err(RouterError::ArgumentValidation(
                    "chat_template_mode=python requires a valid --tokenizer-name path".to_string(),
                ));
            }
            ChatRenderer::python(local_path).map_err(|e| {
                RouterError::ArgumentValidation(format!(
                    "Failed to initialize Python chat template renderer: {e}"
                ))
            })?
        }
        #[cfg(not(feature = "python-chat-template"))]
        "python" => {
            return Err(RouterError::ArgumentValidation(
                "chat_template_mode=python requires building with --features python-chat-template"
                    .to_string(),
            ));
        }
        other => {
            return Err(RouterError::ArgumentValidation(format!(
                "Invalid chat_template_mode: '{other}'. Must be 'none' or 'python'"
            )));
        }
    };

    let server_future = async {
        let _guard = init_logging(otlp_endpoint, json_output, log_path);

        if tokenizer.is_none() {
            tracing::warn!("Tokenizer not loaded for {tokenizer_name}");
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

        // (TGI-era `compat_return_full_text` calculation removed along with
        // the deprecated /generate handlers — see server::tgi_deprecated.)


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

        // Latency simulator initialisation (piggyback observer).
        // No-op unless built with --features simulator AND --enable-simulator.
        #[cfg(feature = "simulator")]
        if enable_simulator {
            use router::simulator::{init_vidur_rf, ModelKind, SimulatorConfig};
            let mut sim_cfg = SimulatorConfig::default();
            sim_cfg.cache_dir = std::path::PathBuf::from(&simulator_cache_dir);
            sim_cfg.model_hash = simulator_model_hash.clone();
            sim_cfg.num_layers = simulator_num_layers;
            sim_cfg.model_kind = if simulator_moe { ModelKind::Moe } else { ModelKind::Llama };
            sim_cfg.learning_rate = simulator_learning_rate;
            sim_cfg.linreg_outlier_threshold_ms = simulator_outlier_threshold_ms;
            sim_cfg.block_size = kvcache_block_size;
            let n = engine_clients.len();
            tracing::info!(
                target: "simulator",
                replicas = n,
                cache_dir = %simulator_cache_dir,
                model_hash = %simulator_model_hash,
                "INITIALISING_SIMULATOR"
            );
            if let Err(e) = init_vidur_rf(n, sim_cfg) {
                tracing::error!(target: "simulator", error = %e, "SIMULATOR_INIT_FAILED");
                return Err(RouterError::ArgumentValidation(format!(
                    "simulator init failed: {e}"
                )));
            }
            tracing::info!(target: "simulator", "SIMULATOR_READY");
        }
        #[cfg(not(feature = "simulator"))]
        if enable_simulator {
            return Err(RouterError::ArgumentValidation(
                "--enable-simulator requires --features simulator at build time".to_string(),
            ));
        }

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

        // --- Auto-discovery from model config.json ---
        // blitz-router is an inner cluster router (not a full inference system).
        // Model-dependent parameters should be auto-discovered, not manually specified.
        let num_engines = engine_clients.len();

        if let Some(model_config) = load_model_config(local_path) {
            tracing::info!(
                "Loaded model config: type={}, max_position_embeddings={}",
                model_config.model_type,
                model_config.max_position_embeddings
            );

            // Auto-discover max_total_tokens if user didn't override (default is 2048)
            if max_total_tokens == 2048 {
                max_total_tokens = model_config.max_position_embeddings;
                tracing::info!("Auto-set max_total_tokens={} from config.json", max_total_tokens);
            }

            // Auto-discover max_input_length if user didn't override (default is 1024)
            if max_input_length == 1024 {
                max_input_length = max_total_tokens - 1;
                tracing::info!("Auto-set max_input_length={}", max_input_length);
            }
        }

        // Scale max_concurrent_requests with engine count if user didn't override (default is 128)
        if max_concurrent_requests == 128 {
            max_concurrent_requests = num_engines * 64;
            tracing::info!(
                "Auto-scaled max_concurrent_requests={} ({} engines x 64)",
                max_concurrent_requests,
                num_engines
            );
        }

        // Run server
        server::run(
            model_info,
            shard_info,
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
            chat_renderer,
            shared_tokenizer,
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
    // Default: cache_tracking is off to avoid noise; enable with LOG_LEVEL="info,cache_tracking=info"
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info,cache_tracking=off"));

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
