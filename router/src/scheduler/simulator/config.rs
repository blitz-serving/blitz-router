// Configuration for the latency-prediction subsystem.
//
// Initial defaults mirror everparadise's `SimulationConfig` in
// `tmp/blitz-infer-pack-sim/router_v2/src/simulator/config.rs`. Wire CLI/env
// plumbing in a follow-up; the immediate piggyback path constructs this
// programmatically.

use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelKind {
    /// Dense Llama-family architecture (attention + 3-component MLP per block).
    Llama,
    /// Mixture-of-Experts (attention + single `moe_linear` per block).
    Moe,
}

#[derive(Clone, Debug)]
pub struct SimulatorConfig {
    /// MD5(config_str)[..8] of the model from Modified_vidur. Used to find CSVs.
    /// Qwen2.5 = `9f4b3b9a`, Llama3 = `d29f0375`.
    pub model_hash: String,
    /// Directory containing `{op_name}_{model_hash}_predictions.csv` files.
    pub cache_dir: PathBuf,
    pub model_kind: ModelKind,

    /// Per-pipeline-stage layer count = num_layers / num_pipeline_stages.
    pub num_layers: usize,
    pub num_pipeline_stages: usize,
    pub tensor_parallel_size: usize,
    pub post_attn_norm: bool,

    /// Multiplicative overhead applied to attention prefill latency when more
    /// than one prefill request is in the batch (GQA models only — set to 0
    /// for MHA models).
    pub attention_prefill_batching_overhead_fraction: f32,
    pub attention_decode_batching_overhead_fraction: f32,

    /// Number of tokens per forward step the scheduler caps at. Used for the
    /// `full_token_cache` fast-path when num_tokens_rounded == token_budget.
    pub token_budget: u32,

    /// Engine block size in tokens (typically 16). Used to round
    /// `num_tokens` → `num_tokens_rounded` when reconstructing the
    /// just-completed step's `BatchForPredictor` from SSE.
    pub block_size: usize,

    /// Engine total KV-cache block count. Used to size the per-replica
    /// `IncrementalMirror`.
    pub num_blocks: usize,

    pub kv_cache_prediction_granularity: usize,
    pub flops_prediction_granularity: usize,

    pub nccl_cpu_launch_overhead_ms: Option<f32>,
    pub nccl_cpu_skew_overhead_per_device_ms: Option<f32>,
    pub skip_cpu_overhead_modeling: bool,

    /// SGD learning rate for the `LinregCorrected` online-correction wrapper.
    pub learning_rate: f32,
    /// Warmup samples skipped before the linreg starts updating (vLLM startup
    /// latency variance).
    pub linreg_warmup: usize,
    /// Reject calibration samples whose error magnitude (real - predicted, in
    /// ms) exceeds this threshold to avoid poisoning the model with outliers.
    pub linreg_outlier_threshold_ms: f32,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            model_hash: "9f4b3b9a".to_string(),
            cache_dir: PathBuf::from("/nvme/zkx/Modified_vidur/cache"),
            model_kind: ModelKind::Llama,
            num_layers: 28,
            num_pipeline_stages: 1,
            tensor_parallel_size: 1,
            post_attn_norm: true,
            attention_prefill_batching_overhead_fraction: 0.1,
            attention_decode_batching_overhead_fraction: 0.4,
            token_budget: 1024,
            block_size: 16,
            num_blocks: 80860,
            kv_cache_prediction_granularity: 64,
            flops_prediction_granularity: 1024,
            nccl_cpu_launch_overhead_ms: None,
            nccl_cpu_skew_overhead_per_device_ms: None,
            skip_cpu_overhead_modeling: true,
            learning_rate: 1e-4,
            linreg_warmup: 10,
            linreg_outlier_threshold_ms: 0.5,
        }
    }
}
