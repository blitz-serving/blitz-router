use crate::simulator::predictor::TrainedPredictorType;
use clap::{Args, Parser};
use std::default;

#[allow(dead_code)]
#[derive(Default, Args, Debug)]
pub struct ReplicaConfig {
    // pub num_replicas: usize,
    // pub replica_id: usize,
    #[arg(long, default_value_t = 1)]
    pub num_pipeline_stages: usize,
    #[arg(long, default_value_t = 1)]
    pub tensor_parallel_size: usize,
}
#[derive(Default, Args, Debug)]
pub struct SchedulerConfig {
    #[arg(long, default_value_t = 1024)]
    pub token_budget: u32,
    // pub batch_size_cap: usize,
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,
    // pub watermark_block_fraction: f64,
    // hack for llama3-8B
    /// !!!!!!!!!!!!!!need exact match to vllm!!!!!!!!!!!!!!!!
    /// 80860 / 36393
    #[arg(long, default_value_t = 80860)]
    pub num_blocks: usize,
    // pub max_tokens_in_batch: usize,
}
#[derive(Default, Args, Debug)]
pub struct ModelConfig {
    // #[arg(long, default_value = "llama")] llama: 32, qwen: 28
    // pub model_name: String,
    #[arg(long, default_value_t = 28)]
    pub num_layers: usize,
    // #[arg(long, default_value_t = 4096)]
    // pub hidden_size: usize,
    // #[arg(long, default_value_t = 11008)]
    // pub intermediate_size: usize,
    #[arg(long, default_value_t = 28)]
    pub num_attention_heads: usize,
    #[arg(long, default_value_t = true)]
    pub post_attn_norm: bool,

    // For GQA models:
    // self._attention_prefill_batching_overhead_fraction = (
    //     (self._config.attention_prefill_batching_overhead_fraction)
    //     if self._model_config.num_q_heads > self._model_config.num_kv_heads
    //     else 0
    // )
    // self._attention_decode_batching_overhead_fraction = (
    //     (self._config.attention_decode_batching_overhead_fraction)
    //     if self._model_config.num_q_heads > self._model_config.num_kv_heads
    //     else 0
    // )
    #[arg(long, default_value_t = 0.1)]
    pub attention_prefill_batching_overhead_fraction: f32,
    #[arg(long, default_value_t = 0.4)]
    pub attention_decode_batching_overhead_fraction: f32,
}
#[derive(Args, Debug, Default)]
pub struct PredictorConfig {
    // Qwen2.5: 9f4b3b9a
    // Llama3: d29f0375
    #[arg(long, default_value = "9f4b3b9a")]
    pub model_hash: String,
    #[arg(long, default_value = "/nvme/zkx/Modified_vidur/cache")]
    pub predict_cache_path_prefix: String,
    #[arg(long, required = true)]
    pub trained_type: TrainedPredictorType,

    #[arg(long)]
    pub learning_rate: Option<f32>,

    #[arg(long, default_value_t = 50.0)]
    pub refine_threshold: f32,

    #[arg(long, default_value_t = 64)]
    pub kv_cache_prediction_granularity: usize,
    #[arg(long, default_value_t = 1024)]
    pub flops_prediction_granularity: usize,
    // #[arg(long, default_value_t = 1024)]
    // pub prediction_max_prefill_chunk_size: usize,
    // #[arg(long, default_value_t = 32)]
    // pub prediction_max_batch_size: usize,
    // #[arg(long, default_value_t = 2048)]
    // pub prediction_max_tokens_per_request: usize,
    // #[arg(long, default_value_t = 0.4)]
    // pub attention_decode_batching_overhead_fraction: f32,
    // #[arg(long, default_value_t = 0.1)]
    // pub attention_prefill_batching_overhead_fraction: f32,
    #[arg(long)]
    pub nccl_cpu_launch_overhead_ms: Option<f32>,
    #[arg(long)]
    pub nccl_cpu_skew_overhead_per_device_ms: Option<f32>,
    #[arg(long, default_value_t = true)]
    pub skip_cpu_overhead_modeling: bool,
}
#[derive(Default, Args, Debug)]
pub struct DeviceConfig {
    // #[arg(long, default_value = "cuda")]
    // pub device: String,
    // #[arg(long, default_value = "eth0")]
    // pub network_device: String,
    // #[arg(long, default_value_t = 0.1)]
    // pub memory_margin_fraction: f64,
}

#[derive(Args, Debug)]
pub struct SimulationConfig {
    #[command(flatten)]
    pub device_config: DeviceConfig,
    #[command(flatten)]
    pub predictor_config: PredictorConfig,
    #[command(flatten)]
    pub replica_config: ReplicaConfig,
    #[command(flatten)]
    pub model_config: ModelConfig,
    #[command(flatten)]
    pub scheduler_config: SchedulerConfig,

    #[arg(long, default_value_t = false)]
    pub fake_backend: bool,

    #[arg(long, default_value_t = false)]
    pub online_refine: bool,
}
