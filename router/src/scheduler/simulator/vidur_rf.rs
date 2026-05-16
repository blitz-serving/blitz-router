// VidurRfPredictor — port of everparadise's `LlamaPredictor` / `MoePredictor`
// (`tmp/blitz-infer-pack-sim/router_v2/src/simulator/predictor.rs`), slimmed:
//   - Returns `f32` (ms) directly instead of `Box<dyn ExecutionTime>`.
//   - Single struct handles both Llama and MoE via `ModelKind`.
//   - Per-op math inlined into a single `predict_step_seconds` function.
//
// Loads precomputed prediction grids from CSV files at
// `{cache_dir}/{op}_{model_hash}_predictions.csv`. Each grid is a hash-map
// from `(usize, usize)` keys to `f32` per-op latency. No interpolation today
// (matches everparadise — missing keys log a warning and fall through to a
// per-op fallback constant).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use super::batch::BatchForPredictor;
use super::config::{ModelKind, SimulatorConfig};
use super::predictor::Predictor;

type Grid = HashMap<(usize, usize), f32>;

pub struct VidurRfPredictor {
    config: Arc<SimulatorConfig>,
    grids: HashMap<&'static str, Grid>,
    /// Precomputed `(token_budget, 0)` lookup per op for the common-case fast
    /// path when `num_tokens_rounded == token_budget`. Keyed by op name.
    full_token_cache: HashMap<&'static str, f32>,
}

/// Per-op fallback latencies (ms) used when a CSV lookup misses. Matches the
/// per-op constants everparadise's predictor.rs falls back to with a tracing
/// error. Missing entries default to 0.0.
fn fallback_ms(op: &str) -> f32 {
    match op {
        "attn_prefill" => 400.0,
        "attn_decode" => 30.0,
        "schedule" => 10.0,
        "sampler_e2e" => 80.0,
        "attn_pre_proj" | "attn_post_proj" => 40.0,
        "moe_linear" => 100.0,
        "input_layernorm"
        | "post_attention_layernorm"
        | "add"
        | "attn_rope"
        | "attn_kv_cache_save" => 10.0,
        _ => 0.0,
    }
}

/// Llama-arch op set used by `load_models` and the per-step lookups below.
const LLAMA_FULL_TOKEN_OPS: &[&str] = &[
    "attn_pre_proj",
    "attn_post_proj",
    "mlp_up_proj",
    "mlp_down_proj",
    "mlp_act",
    "attn_rope",
    "attn_kv_cache_save",
    "input_layernorm",
    "post_attention_layernorm",
    "add",
];

const MOE_FULL_TOKEN_OPS: &[&str] = &[
    "attn_pre_proj",
    "attn_post_proj",
    "moe_linear",
    "attn_rope",
    "attn_kv_cache_save",
    "input_layernorm",
    "post_attention_layernorm",
    "add",
];

impl VidurRfPredictor {
    pub fn new(config: Arc<SimulatorConfig>) -> std::io::Result<Self> {
        let mut p = Self { config, grids: HashMap::new(), full_token_cache: HashMap::new() };
        p.load_models()?;
        Ok(p)
    }

    fn load_models(&mut self) -> std::io::Result<()> {
        // Attention models — same for Llama and MoE.
        let mut ops: Vec<&'static str> = vec![
            "attn_decode",
            "attn_prefill",
            "attn_pre_proj",
            "attn_post_proj",
            "attn_rope",
            "attn_kv_cache_save",
            "input_layernorm",
            "post_attention_layernorm",
            "add",
            "sampler_e2e",
            "schedule",
        ];

        // MLP / MoE arch-specific ops.
        match self.config.model_kind {
            ModelKind::Llama => ops.extend(["mlp_up_proj", "mlp_down_proj", "mlp_act"]),
            ModelKind::Moe => ops.push("moe_linear"),
        }

        if self.config.num_pipeline_stages > 1 {
            ops.push("send_recv");
        }
        if self.config.tensor_parallel_size > 1 {
            ops.push("all_reduce");
        }

        let full_token_ops: &[&str] = match self.config.model_kind {
            ModelKind::Llama => LLAMA_FULL_TOKEN_OPS,
            ModelKind::Moe => MOE_FULL_TOKEN_OPS,
        };

        for op in ops {
            let grid = self.load_csv(op)?;
            if full_token_ops.contains(&op) {
                let key = (self.config.token_budget as usize, 0);
                if let Some(&v) = grid.get(&key) {
                    self.full_token_cache.insert(op, v);
                } else {
                    tracing::warn!(
                        target: "simulator",
                        "missing full_token_cache entry for op={} key={:?}",
                        op,
                        key
                    );
                }
            }
            self.grids.insert(op, grid);
            tracing::debug!(target: "simulator", "loaded grid for op {}", op);
        }
        Ok(())
    }

    fn load_csv(&self, op: &str) -> std::io::Result<Grid> {
        let path = self
            .config
            .cache_dir
            .join(format!("{op}_{hash}_predictions.csv", hash = self.config.model_hash));
        let file = File::open(&path).map_err(|e| {
            tracing::error!(target: "simulator", "failed to open {}: {}", path.display(), e);
            e
        })?;
        let reader = BufReader::new(file);
        let mut grid: Grid = HashMap::new();
        // Skip the header row.
        for (line_no, line) in reader.lines().enumerate().skip(1) {
            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 2 {
                continue;
            }
            let r1: usize = match parts[0].trim().parse() {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        target: "simulator",
                        "skipping malformed line {} in {}: {}",
                        line_no + 1,
                        path.display(),
                        e
                    );
                    continue;
                }
            };
            let r2: usize = if parts.len() == 2 { 0 } else { parts[1].trim().parse().unwrap_or(0) };
            let v: f32 = parts.last().unwrap().trim().parse().unwrap_or(0.0);
            grid.insert((r1, r2), v);
        }
        Ok(grid)
    }

    fn lookup(&self, op: &str, key: (usize, usize)) -> f32 {
        match self.grids.get(op).and_then(|g| g.get(&key).copied()) {
            Some(v) => v,
            None => {
                tracing::warn!(
                    target: "simulator",
                    "missing prediction for op={} key={:?}, falling back",
                    op,
                    key
                );
                fallback_ms(op)
            }
        }
    }

    fn lookup_full_or(&self, op: &str, batch: &BatchForPredictor) -> f32 {
        if batch.num_tokens_rounded as u32 == self.config.token_budget {
            if let Some(&v) = self.full_token_cache.get(op) {
                return v;
            }
        }
        self.lookup(op, (batch.num_tokens_rounded, 0))
    }

    fn round_up(&self, value: usize, granularity: usize) -> usize {
        (value + granularity - 1) / granularity * granularity
    }

    fn attn_prefill_ms(&self, batch: &BatchForPredictor) -> f32 {
        if batch.num_prefill_tokens.is_empty() {
            return 0.0;
        }
        let kv_g = self.config.kv_cache_prediction_granularity;
        let fl_g = self.config.flops_prediction_granularity;

        let mut max_t = 0.0f32;
        for (chunk, computed) in
            batch.num_prefill_tokens.iter().zip(batch.num_prefill_computed_tokens.iter())
        {
            let kv = self.round_up(*computed, kv_g);
            let flops = chunk * ((chunk + 1) / 2 + kv);
            let flops_r = self.round_up(flops, fl_g);
            let t = self.lookup("attn_prefill", (kv, flops_r));
            if t > max_t {
                max_t = t;
            }
        }
        if batch.num_prefill_tokens.len() > 1 {
            max_t * (1.0 + self.config.attention_prefill_batching_overhead_fraction)
        } else {
            max_t
        }
    }

    fn attn_decode_ms(&self, batch: &BatchForPredictor) -> f32 {
        let n = batch.num_decode_computed_tokens.len();
        if n == 0 {
            return 0.0;
        }
        let kv_sum: usize = batch.num_decode_computed_tokens.iter().sum();
        let kv_avg = kv_sum / n;
        let kv_avg_r = self.round_up(kv_avg, self.config.kv_cache_prediction_granularity);
        let base = self.lookup("attn_decode", (n, kv_avg_r));
        let overhead =
            if n > 1 { self.config.attention_decode_batching_overhead_fraction } else { 0.0 };
        base * (1.0 + overhead)
    }

    fn schedule_ms(&self, batch: &BatchForPredictor) -> f32 {
        self.lookup("schedule", (batch.size, 0))
    }

    fn sampler_e2e_ms(&self, batch: &BatchForPredictor) -> f32 {
        self.lookup("sampler_e2e", (batch.size, 0))
    }

    fn attn_norm_ms(&self, batch: &BatchForPredictor) -> f32 {
        self.lookup_full_or("input_layernorm", batch)
    }

    fn mlp_norm_ms(&self, batch: &BatchForPredictor) -> f32 {
        if !self.config.post_attn_norm {
            return 0.0;
        }
        self.lookup_full_or("post_attention_layernorm", batch)
    }

    fn add_ms(&self, batch: &BatchForPredictor) -> f32 {
        self.lookup_full_or("add", batch)
    }

    fn attn_rope_ms(&self, batch: &BatchForPredictor) -> f32 {
        self.lookup_full_or("attn_rope", batch)
    }

    fn attn_kv_cache_save_ms(&self, batch: &BatchForPredictor) -> f32 {
        // Note: everparadise keys this on `num_tokens` (not `num_tokens_rounded`).
        if batch.num_tokens as u32 == self.config.token_budget {
            if let Some(&v) = self.full_token_cache.get("attn_kv_cache_save") {
                return v;
            }
        }
        self.lookup("attn_kv_cache_save", (batch.num_tokens, 0))
    }

    fn attn_pre_proj_ms(&self, batch: &BatchForPredictor) -> f32 {
        self.lookup_full_or("attn_pre_proj", batch)
    }

    fn attn_post_proj_ms(&self, batch: &BatchForPredictor) -> f32 {
        self.lookup_full_or("attn_post_proj", batch)
    }

    fn mlp_up_proj_ms(&self, batch: &BatchForPredictor) -> f32 {
        self.lookup_full_or("mlp_up_proj", batch)
    }

    fn mlp_down_proj_ms(&self, batch: &BatchForPredictor) -> f32 {
        // Matches everparadise's quirk: shave 0.125ms off when latency > 0.78125.
        let mut latency = self.lookup_full_or("mlp_down_proj", batch);
        if latency > 0.78125 {
            latency -= 0.125;
        }
        latency
    }

    fn mlp_act_ms(&self, batch: &BatchForPredictor) -> f32 {
        self.lookup_full_or("mlp_act", batch)
    }

    fn moe_linear_ms(&self, batch: &BatchForPredictor) -> f32 {
        self.lookup_full_or("moe_linear", batch)
    }

    fn tp_comm_ms(&self, batch: &BatchForPredictor) -> f32 {
        if self.config.tensor_parallel_size == 1 {
            return 0.0;
        }
        let base = self.lookup("all_reduce", (batch.num_tokens_rounded, 0));
        let launch = self.config.nccl_cpu_launch_overhead_ms.unwrap_or(0.0);
        let skew = self.config.nccl_cpu_skew_overhead_per_device_ms.unwrap_or(0.0);
        base + launch + skew * (self.config.tensor_parallel_size as f32).powf(1.25)
    }

    fn pp_comm_ms(&self, batch: &BatchForPredictor) -> f32 {
        if self.config.num_pipeline_stages == 1 {
            return 0.0;
        }
        self.lookup("send_recv", (batch.num_tokens_rounded, 0))
    }

    fn cpu_overhead_ms(&self, batch: &BatchForPredictor) -> f32 {
        if self.config.skip_cpu_overhead_modeling {
            return 0.0;
        }
        self.schedule_ms(batch) + self.sampler_e2e_ms(batch)
    }

    fn block_ms(&self, batch: &BatchForPredictor) -> f32 {
        let attn = self.attn_pre_proj_ms(batch)
            + self.attn_post_proj_ms(batch)
            + self.attn_rope_ms(batch)
            + self.attn_kv_cache_save_ms(batch)
            + self.attn_decode_ms(batch)
            + self.attn_prefill_ms(batch)
            + self.tp_comm_ms(batch)
            + self.attn_norm_ms(batch);
        let mlp = match self.config.model_kind {
            ModelKind::Llama => {
                self.mlp_up_proj_ms(batch)
                    + self.mlp_down_proj_ms(batch)
                    + self.mlp_act_ms(batch)
                    + self.tp_comm_ms(batch)
                    + self.mlp_norm_ms(batch)
            }
            ModelKind::Moe => {
                self.moe_linear_ms(batch) + self.tp_comm_ms(batch) + self.mlp_norm_ms(batch)
            }
        };
        attn + mlp + self.add_ms(batch)
    }

    fn model_ms(&self, batch: &BatchForPredictor) -> f32 {
        let layers_per_stage = self.config.num_layers / self.config.num_pipeline_stages.max(1);
        self.block_ms(batch) * layers_per_stage as f32 + self.pp_comm_ms(batch)
    }
}

impl Predictor for VidurRfPredictor {
    fn predict(&self, batch: &BatchForPredictor) -> f32 {
        // CSV grids are in milliseconds; sampler/schedule are also in ms.
        // No unit conversion needed.
        self.model_ms(batch) + self.cpu_overhead_ms(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_csv(dir: &std::path::Path, name: &str, rows: &[(usize, usize, f32)]) {
        let path = dir.join(name);
        let mut f = std::fs::File::create(path).unwrap();
        writeln!(f, "k1,k2,v").unwrap();
        for (a, b, v) in rows {
            writeln!(f, "{},{},{}", a, b, v).unwrap();
        }
    }

    fn write_csv1d(dir: &std::path::Path, name: &str, rows: &[(usize, f32)]) {
        let path = dir.join(name);
        let mut f = std::fs::File::create(path).unwrap();
        writeln!(f, "k,v").unwrap();
        for (a, v) in rows {
            writeln!(f, "{},{}", a, v).unwrap();
        }
    }

    fn build_minimal_grids(dir: &std::path::Path, hash: &str) {
        // 2D grids
        write_csv(
            dir,
            &format!("attn_decode_{hash}_predictions.csv"),
            &[(1, 1024, 0.5), (2, 1024, 0.7)],
        );
        write_csv(dir, &format!("attn_prefill_{hash}_predictions.csv"), &[(0, 1024, 1.0)]);
        // 1D-only grids (k2=0 is implied, both 2-col and 3-col formats supported)
        for op in [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "attn_rope",
            "attn_kv_cache_save",
            "input_layernorm",
            "post_attention_layernorm",
            "add",
        ] {
            write_csv1d(dir, &format!("{op}_{hash}_predictions.csv"), &[(16, 0.05), (1024, 0.5)]);
        }
        write_csv1d(dir, &format!("schedule_{hash}_predictions.csv"), &[(1, 0.1), (4, 0.2)]);
        write_csv1d(dir, &format!("sampler_e2e_{hash}_predictions.csv"), &[(1, 0.3), (4, 0.6)]);
    }

    #[test]
    fn loads_csv_and_predicts() {
        let tmp = tempfile::tempdir().unwrap();
        let hash = "testhash";
        build_minimal_grids(tmp.path(), hash);

        let mut config = SimulatorConfig::default();
        config.model_hash = hash.to_string();
        config.cache_dir = tmp.path().to_path_buf();
        config.num_layers = 1;
        config.skip_cpu_overhead_modeling = true;
        config.post_attn_norm = false;

        let predictor = VidurRfPredictor::new(Arc::new(config)).unwrap();
        let batch = BatchForPredictor {
            num_tokens: 16,
            num_tokens_rounded: 16,
            num_prefill_tokens: vec![],
            num_prefill_computed_tokens: vec![],
            num_decode_computed_tokens: vec![1],
            size: 1,
        };
        let ms = predictor.predict(&batch);
        // Single layer, decode-only. Components (all from single CSV grids):
        //   pre_proj 0.05 + post_proj 0.05 + rope 0.05 + kv_save 0.05
        //   + attn_decode (1, 1024=>0.5) + attn_norm 0.05 + 0 (mlp_norm disabled)
        //   + mlp_up 0.05 + mlp_down 0.05 (not > 0.78125 threshold) + mlp_act 0.05
        //   + add 0.05 + 0 (tp/pp comm disabled)
        // = 9 * 0.05 + 0.5 = 0.95ms
        assert!((ms - 0.95).abs() < 0.01, "predicted = {}", ms);
    }

    #[test]
    fn linreg_correction_converges() {
        use super::super::predictor::{
            LinregCorrector, Predictor, RegressionalPredictor, TrainedPredictor,
        };

        struct ConstPred(f32);
        impl Predictor for ConstPred {
            fn predict(&self, _b: &BatchForPredictor) -> f32 {
                self.0
            }
        }

        let mut cfg = SimulatorConfig::default();
        cfg.linreg_warmup = 0;
        cfg.learning_rate = 0.05;
        cfg.linreg_outlier_threshold_ms = 100.0;

        let mut wrapped =
            RegressionalPredictor::new(Arc::new(ConstPred(2.0)), LinregCorrector::new(&cfg));
        let batch = BatchForPredictor::default();
        // Inner says 2.0; truth is 4.0. After many calibration steps the
        // correction should bring the corrected output close to 4.0.
        for _ in 0..2000 {
            let p = wrapped.predict(&batch);
            wrapped.calibrate(&batch, 4.0);
            let _ = p;
        }
        let final_p = wrapped.predict(&batch);
        assert!((final_p - 4.0).abs() < 0.05, "linreg did not converge: final={}", final_p);
    }
}
