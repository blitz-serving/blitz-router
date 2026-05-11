// Latency simulator subsystem (lmetric scope).
//
// Two-layer design:
//   inner regressor (offline-trained, online interpolated) → per-step latency
//   outer discrete-event simulator (rolls forward engine steps)        → RolloutBuffer
//
// Owned per-replica by the colocation controller via a process-wide
// `SIMULATOR` OnceLock (avoids threading PCtx through 3 function signatures).
// Piggyback-first: observes admissions made by the active <name>-q policy,
// emits predicted-vs-actual metrics; does not influence routing.
//
// See blitz-router/.claude/memory/project_lmetric_predictor_design.md.

mod batch;
mod config;
mod mirror;
mod pctx;
mod predictor;
mod rollout;
mod sched;
mod vidur_rf;

pub use batch::BatchForPredictor;
pub use config::{ModelKind, SimulatorConfig};
pub use mirror::IncrementalMirror;
pub use pctx::PCtx;
pub use predictor::{
    Corrector, LinregCorrector, NullCorrector, Predictor, RegressionalPredictor,
    TrainedPredictor,
};
pub use radixtree::RadixTreeReqIdHash;
pub use rollout::{RolloutBuffer, RolloutGist, RolloutSlot};
pub use sched::{ReqProgress, SchedSnapshot};
pub use vidur_rf::VidurRfPredictor;

use std::sync::{Arc, OnceLock};

use crate::engine::EngineStepOutput;

/// Process-wide per-replica predictor contexts. Set once at startup by
/// `init()`; read on the SSE consumer hot path by `on_sse()` and on the
/// policy admit hook by `on_admit()`.
static SIMULATOR: OnceLock<SimulatorRuntime> = OnceLock::new();

struct SimulatorRuntime {
    pctxs: Vec<Arc<PCtx>>,
    block_size: usize,
}

/// Builder for the runtime predictor backend. The piggyback path uses
/// `init_with_predictor` directly so callers can substitute a fake / no-op
/// predictor for tests; the production path is `init_vidur_rf`.
pub fn init_with_predictor(
    num_replicas: usize,
    config: &SimulatorConfig,
    inner: Arc<dyn Predictor>,
) -> Result<(), &'static str> {
    let mut pctxs = Vec::with_capacity(num_replicas);
    for _ in 0..num_replicas {
        let trained: Box<dyn TrainedPredictor> = Box::new(RegressionalPredictor::new(
            inner.clone(),
            LinregCorrector::new(config),
        ));
        pctxs.push(Arc::new(PCtx::new(
            trained,
            config.block_size as u32,
            config.token_budget,
        )));
    }
    SIMULATOR
        .set(SimulatorRuntime { pctxs, block_size: config.block_size })
        .map_err(|_| "simulator already initialised")
}

/// Production path: build a `VidurRfPredictor` from CSV grids on disk and
/// install it as the simulator backend.
pub fn init_vidur_rf(
    num_replicas: usize,
    config: SimulatorConfig,
) -> std::io::Result<()> {
    let cfg_arc = Arc::new(config.clone());
    let inner: Arc<dyn Predictor> = Arc::new(VidurRfPredictor::new(cfg_arc)?);
    init_with_predictor(num_replicas, &config, inner)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::AlreadyExists, e))
}

/// Whether the simulator subsystem has been initialised in this process.
pub fn is_active() -> bool {
    SIMULATOR.get().is_some()
}

/// SSE consumer hook (piggyback). No-op when the simulator is inactive.
/// Reconstructs `BatchForPredictor` from the just-completed step, then
/// hands off to `PCtx::on_sse` which drives all three layers (L2
/// calibrate, L1 absorb, L3 invariant) and returns the (predicted,
/// actual) latency pair for Prometheus emission.
pub(crate) fn on_sse(replica_index: usize, m: &EngineStepOutput) {
    let Some(rt) = SIMULATOR.get() else {
        return;
    };
    let Some(pctx) = rt.pctxs.get(replica_index) else {
        return;
    };
    let batch = batch_from_step(m, rt.block_size);
    let (predicted, actual) = pctx.on_sse(&batch, m);

    metrics::histogram!("simulator_predicted_ms", predicted as f64);
    metrics::histogram!("simulator_actual_ms", actual as f64);
    let signed = actual - predicted;
    metrics::histogram!("simulator_signed_error_ms", signed as f64);
    metrics::histogram!("simulator_abs_error_ms", signed.abs() as f64);
    if actual > 1e-3 {
        metrics::histogram!(
            "simulator_relative_error",
            (signed.abs() / actual) as f64
        );
    }
}

/// Admission-time hook (piggyback). Called by `policy_runner`
/// immediately after the chosen <name>-q policy commits a request to
/// a specific replica's commit buffer. No-op when the simulator is
/// inactive.
///
/// Drives admission across all three layers in `PCtx`: L1 mirror
/// gains the request's prefix hashes, sched snapshot gains a fresh
/// `ReqProgress` in `waiting`, and L3 ephemeral runs the
/// promote-or-drop branch.
pub(crate) fn on_admit(replica_index: usize, entry: &super::policies::Entry) {
    let Some(rt) = SIMULATOR.get() else {
        return;
    };
    let Some(pctx) = rt.pctxs.get(replica_index) else {
        return;
    };
    pctx.on_admit(
        entry.request.request_id,
        entry.request.input_length,
        entry.block_hash_state.get_hashes(),
    );
}

/// Speculative rollout query for a `(replica_index, candidate)`
/// pair. Returns `None` when the simulator is inactive or the index
/// is out of range.
///
/// The caller (eventual `simulator-q` policy) supplies all candidate
/// fields plus the SCtx prefix-hit count; the simulator combines
/// the latter with its own L1 mirror lookup via A2's max-merge rule.
pub fn query(
    replica_index: usize,
    candidate_id: u64,
    input_length: u32,
    candidate_hashes: &[u64],
    sctx_prefix_hits: usize,
) -> Option<RolloutGist> {
    let rt = SIMULATOR.get()?;
    let pctx = rt.pctxs.get(replica_index)?;
    Some(pctx.query(candidate_id, input_length, candidate_hashes, sctx_prefix_hits))
}

/// Reconstruct an approximate `BatchForPredictor` from the just-completed
/// step's SSE payload.
///
/// Per-request `prev_computed_tokens` from yaullm gives us the exact KV
/// cache size for both PREFILL (= bytes the regressor's `attn_prefill`
/// 2D key needs) and DECODE (= the `attn_decode` 2D key) requests.
///
/// Remaining approximation: per-prefill chunk size is still distributed
/// evenly across PREFILL state requests because the SSE payload only
/// reports aggregate `prefill_tokens` (not per-request prefill chunk
/// size). For typical chunked-prefill configs at most one prefill is
/// active per step, so this is exact in practice.
fn batch_from_step(m: &EngineStepOutput, block_size: usize) -> BatchForPredictor {
    let mut num_prefill_tokens: Vec<usize> = Vec::new();
    let mut num_prefill_computed_tokens: Vec<usize> = Vec::new();
    let mut num_decode_computed_tokens: Vec<usize> = Vec::new();

    for o in &m.outputs {
        match o.state.as_str() {
            "PREFILL" => {
                num_prefill_computed_tokens.push(o.prev_computed_tokens as usize);
            }
            // yaullm uses "DECODE"; vLLM-core ZMQ backend uses "RUNNING".
            "DECODE" | "RUNNING" => {
                num_decode_computed_tokens.push(o.prev_computed_tokens as usize);
            }
            _ => {}
        }
    }

    let prefill_count = num_prefill_computed_tokens.len();
    if prefill_count > 0 {
        let per = m.prefill_tokens / prefill_count;
        let remainder = m.prefill_tokens % prefill_count;
        for i in 0..prefill_count {
            num_prefill_tokens.push(if i < remainder { per + 1 } else { per });
        }
    }

    let num_tokens = m.prefill_tokens + num_decode_computed_tokens.len();
    let bs_safe = block_size.max(1);
    let num_tokens_rounded = ((num_tokens + bs_safe - 1) / bs_safe) * bs_safe;
    BatchForPredictor {
        num_tokens,
        num_tokens_rounded,
        num_prefill_tokens,
        num_prefill_computed_tokens,
        num_decode_computed_tokens,
        size: m.outputs.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::RequestStepOutput;
    use nohash_hasher::{BuildNoHashHasher, IntMap};

    fn empty_step(prefill_tokens: usize, latency_ms: u64, outputs: Vec<RequestStepOutput>) -> EngineStepOutput {
        EngineStepOutput {
            prefill_tokens,
            prefill_token_budget: 1024,
            latency: latency_ms,
            outputs,
            new_block_hashes: Vec::new(),
            evicted_block_hashes: Vec::new(),
            evicted_block_ids: Vec::new(),
            cur_used_block_ids: IntMap::with_hasher(BuildNoHashHasher::default()),
            new_block_hashes_ids: IntMap::with_hasher(BuildNoHashHasher::default()),
            op_exec_log: None,
            preempted_ids: Vec::new(),
            aborted_requests: Vec::new(),
            step_id: 0,
        }
    }

    #[test]
    fn batch_from_step_decode_only() {
        let step = empty_step(
            0,
            50,
            vec![
                RequestStepOutput {
                    request_id: 1,
                    new_token_ids: vec![10],
                    state: "DECODE".to_string(),
                    is_finished: false,
                    hit_token_cnt: 64,
                    prev_computed_tokens: 128,
                },
                RequestStepOutput {
                    request_id: 2,
                    new_token_ids: vec![11],
                    state: "DECODE".to_string(),
                    is_finished: false,
                    hit_token_cnt: 64,
                    prev_computed_tokens: 256,
                },
            ],
        );
        let b = batch_from_step(&step, 16);
        assert_eq!(b.size, 2);
        assert_eq!(b.num_tokens, 2);
        assert_eq!(b.num_tokens_rounded, 16);
        assert_eq!(b.num_prefill_tokens, Vec::<usize>::new());
        // Now driven by prev_computed_tokens, not hit_token_cnt.
        assert_eq!(b.num_decode_computed_tokens, vec![128, 256]);
    }

    #[test]
    fn batch_from_step_chunked_prefill_distributes_tokens() {
        let step = empty_step(
            100,
            200,
            vec![
                RequestStepOutput {
                    request_id: 1,
                    new_token_ids: vec![],
                    state: "PREFILL".to_string(),
                    is_finished: false,
                    hit_token_cnt: 0,
                    // 1st request: 200 tokens already prefilled before this step
                    prev_computed_tokens: 200,
                },
                RequestStepOutput {
                    request_id: 2,
                    new_token_ids: vec![],
                    state: "PREFILL".to_string(),
                    is_finished: false,
                    hit_token_cnt: 0,
                    // 2nd request: hadn't started yet
                    prev_computed_tokens: 0,
                },
                RequestStepOutput {
                    request_id: 3,
                    new_token_ids: vec![10],
                    state: "DECODE".to_string(),
                    is_finished: false,
                    hit_token_cnt: 0,
                    prev_computed_tokens: 64,
                },
            ],
        );
        let b = batch_from_step(&step, 16);
        assert_eq!(b.size, 3);
        // 100 prefill tokens still split evenly across 2 PREFILL reqs
        assert_eq!(b.num_prefill_tokens, vec![50, 50]);
        // KV size at step start now exact (was hardcoded 0 before).
        assert_eq!(b.num_prefill_computed_tokens, vec![200, 0]);
        assert_eq!(b.num_decode_computed_tokens, vec![64]);
        // num_tokens = 100 prefill + 1 decode = 101 → rounded to 112
        assert_eq!(b.num_tokens, 101);
        assert_eq!(b.num_tokens_rounded, 112);
    }

    #[test]
    fn on_sse_end_to_end_calibrates() {
        // End-to-end piggyback verification:
        //   1. Initialise the simulator with a constant inner predictor (1.0ms).
        //   2. Feed a sequence of synthetic SSE steps with actual=4.0ms.
        //   3. Verify that internally the linreg correction has nudged the
        //      corrected prediction toward the actual.
        // This is the local-CI proxy for piggyback success — runs without
        // CSV grids or a real engine.
        use super::predictor::Predictor;

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
        cfg.block_size = 16;
        cfg.num_blocks = 1024;

        // We can't re-init the OnceLock across tests, so this test is wrapped
        // to handle the case where another test set it first. In a real
        // single-replica process this is set exactly once at startup.
        if init_with_predictor(1, &cfg, std::sync::Arc::new(ConstPred(1.0))).is_err() {
            // Already initialised by another test in the same binary; skip.
            return;
        }

        let step = empty_step(
            0,
            4, // actual = 4ms
            vec![RequestStepOutput {
                request_id: 1,
                new_token_ids: vec![10],
                state: "DECODE".to_string(),
                is_finished: false,
                hit_token_cnt: 0,
                prev_computed_tokens: 64,
            }],
        );
        for _ in 0..3000 {
            on_sse(0, &step);
        }
        // After many calibrations, the per-replica regressor's corrected
        // prediction should be close to 4ms.
        let rt = SIMULATOR.get().expect("simulator must be initialised");
        let pctx = rt.pctxs[0].clone();
        let batch = batch_from_step(&step, cfg.block_size);
        let final_pred = pctx.predict(&batch);
        assert!(
            (final_pred - 4.0).abs() < 0.1,
            "linreg did not converge through on_sse: final_pred={}",
            final_pred
        );
    }
}
