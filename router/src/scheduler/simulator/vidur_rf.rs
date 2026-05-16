// VidurRfPredictor — port of everparadise's `LlamaPredictor` / `MoePredictor`
// (`tmp/blitz-infer-pack-sim/router_v2/src/simulator/predictor.rs`), slimmed:
//   - Returns `f32` (ms) directly instead of `Box<dyn ExecutionTime>`.
//   - Keeps Vidur's attention prefill/decode feature math.
//   - Adds fixed 1D cost models for non-attention runtime components.
//
// Loads precomputed prediction grids from CSV files at
// `{cache_dir}/{op}_predictions.csv`. Each grid is a hash-map
// from `(usize, usize)` keys to `f32` per-op latency. No interpolation today
// (matches everparadise — missing keys log a warning and fall through to a
// per-op fallback constant).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use super::batch::BatchForPredictor;
use super::config::SimulatorConfig;
use super::predictor::Predictor;

type Grid = HashMap<(usize, usize), f32>;

pub struct VidurRfPredictor {
    config: Arc<SimulatorConfig>,
    grids: HashMap<&'static str, Grid>,
}

/// Per-op fallback latencies (ms) used when a CSV lookup misses. Matches the
/// per-op constants everparadise's predictor.rs falls back to with a tracing
/// error. Missing entries default to 0.0.
fn fallback_ms(op: &str) -> f32 {
    match op {
        "attn_prefill" => 400.0,
        "attn_decode" => 30.0,
        _ => 0.0,
    }
}

const TOKEN_OPS: &[&str] = &["piecegraph", "prepare_inputs"];
const BATCH_OPS: &[&str] = &["schedule", "update_from_output", "norm", "compute_logits", "sampler"];

impl VidurRfPredictor {
    pub fn new(config: Arc<SimulatorConfig>) -> std::io::Result<Self> {
        let mut p = Self { config, grids: HashMap::new() };
        p.load_models()?;
        Ok(p)
    }

    fn load_models(&mut self) -> std::io::Result<()> {
        let ops = ["attn_decode", "attn_prefill"]
            .into_iter()
            .chain(TOKEN_OPS.iter().copied())
            .chain(BATCH_OPS.iter().copied());

        for op in ops {
            let grid = self.load_csv(op)?;
            self.grids.insert(op, grid);
            tracing::debug!(target: "simulator", "loaded grid for op {}", op);
        }
        Ok(())
    }

    fn load_csv(&self, op: &str) -> std::io::Result<Grid> {
        let path = self.config.cache_dir.join(format!("{op}_predictions.csv"));
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

    fn token_op_ms(&self, op: &str, batch: &BatchForPredictor) -> f32 {
        self.lookup(op, (batch.num_tokens, 0))
    }

    fn batch_op_ms(&self, op: &str, batch: &BatchForPredictor) -> f32 {
        self.lookup(op, (batch.size, 0))
    }

    fn model_ms(&self, batch: &BatchForPredictor) -> f32 {
        let layers = self.config.num_layers as f32;
        let attention = (self.attn_decode_ms(batch) + self.attn_prefill_ms(batch)) * layers;
        let token_ops: f32 = TOKEN_OPS.iter().map(|op| self.token_op_ms(op, batch)).sum();
        let batch_ops: f32 = BATCH_OPS.iter().map(|op| self.batch_op_ms(op, batch)).sum();
        attention + token_ops + batch_ops
    }
}

impl Predictor for VidurRfPredictor {
    fn predict(&self, batch: &BatchForPredictor) -> f32 {
        // CSV grids are in milliseconds. No unit conversion needed.
        self.model_ms(batch)
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

    fn write_csv1d(dir: &std::path::Path, name: &str, header: &str, rows: &[(usize, f32)]) {
        let path = dir.join(name);
        let mut f = std::fs::File::create(path).unwrap();
        writeln!(f, "{header},prediction").unwrap();
        for (a, v) in rows {
            writeln!(f, "{},{}", a, v).unwrap();
        }
    }

    fn build_minimal_grids(dir: &std::path::Path) {
        // 2D grids
        write_csv(dir, "attn_decode_predictions.csv", &[(1, 1024, 0.5), (2, 1024, 0.7)]);
        write_csv(dir, "attn_prefill_predictions.csv", &[(0, 1024, 1.0)]);
        // 1D-only grids (k2=0 is implied, both 2-col and 3-col formats supported)
        write_csv1d(dir, "piecegraph_predictions.csv", "num_tokens", &[(17, 0.11), (1024, 0.5)]);
        write_csv1d(
            dir,
            "prepare_inputs_predictions.csv",
            "num_tokens",
            &[(17, 0.12), (1024, 0.5)],
        );
        write_csv1d(dir, "schedule_predictions.csv", "batch_size", &[(2, 0.2), (4, 0.9)]);
        write_csv1d(dir, "update_from_output_predictions.csv", "batch_size", &[(2, 0.3), (4, 0.9)]);
        write_csv1d(dir, "norm_predictions.csv", "batch_size", &[(2, 0.4), (4, 0.9)]);
        write_csv1d(dir, "compute_logits_predictions.csv", "batch_size", &[(2, 0.5), (4, 0.9)]);
        write_csv1d(dir, "sampler_predictions.csv", "batch_size", &[(2, 0.6), (4, 0.9)]);
    }

    #[test]
    fn loads_csv_and_predicts() {
        let tmp = tempfile::tempdir().unwrap();
        build_minimal_grids(tmp.path());

        let mut config = SimulatorConfig::default();
        config.cache_dir = tmp.path().to_path_buf();
        config.num_layers = 2;

        let predictor = VidurRfPredictor::new(Arc::new(config)).unwrap();
        let batch = BatchForPredictor {
            num_tokens: 17,
            num_tokens_rounded: 32,
            num_prefill_tokens: vec![16],
            num_prefill_computed_tokens: vec![0],
            num_decode_computed_tokens: vec![1],
            size: 2,
        };
        let ms = predictor.predict(&batch);
        // Two layers, mixed prefill/decode. Components:
        //   (attn_decode 0.5 + attn_prefill 1.0) * 2 layers
        //   + piecegraph 0.11 + prepare_inputs 0.12 by num_tokens=17
        //   + schedule/update/norm/logits/sampler by batch_size=2
        // = 3.0 + 0.23 + 2.0 = 5.23ms
        assert!((ms - 5.23).abs() < 0.01, "predicted = {}", ms);
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
