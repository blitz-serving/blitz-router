// PCtx — per-replica predictor context owned by the colocation controller.
//
// Holds the trained inner regressor (with online linreg correction), the
// incremental KV-cache mirror, and a small per-replica state used by the
// outer discrete-event simulator. Public methods are the surface called by
// the colocation SSE consumer and the policy_runner admission hook.
//
// In piggyback mode the only hot-path call is `observe_step()`: the SSE
// consumer reconstructs the just-completed step's `BatchForPredictor`,
// passes it in along with the engine-reported actual latency, and PCtx
// returns the predicted-actual pair for metrics emission.

use std::sync::Mutex;

use super::batch::BatchForPredictor;
use super::mirror::IncrementalMirror;
use super::predictor::TrainedPredictor;

pub struct PCtx {
    /// Online-corrected predictor (linreg wrapper around an inner ML model).
    /// `Mutex` because `TrainedPredictor::calibrate` takes `&mut self`. The
    /// critical section is microseconds; std `Mutex` is fine.
    regressor: Mutex<Box<dyn TrainedPredictor>>,
    /// Per-replica incremental KV-cache mirror. Updated on admission /
    /// eviction. Read by the outer DES rollout (when implemented).
    mirror: Mutex<IncrementalMirror>,
}

impl PCtx {
    pub fn new(regressor: Box<dyn TrainedPredictor>, num_blocks: usize) -> Self {
        Self {
            regressor: Mutex::new(regressor),
            mirror: Mutex::new(IncrementalMirror::new(num_blocks)),
        }
    }

    /// Predict the latency of the supplied batch using the current regressor
    /// state, calibrate the linreg using the engine-observed actual latency
    /// (ms), and return `(predicted_ms, actual_ms)` for metrics emission.
    /// Single critical section across predict + calibrate.
    pub fn observe_step(&self, batch: &BatchForPredictor, actual_ms: f32) -> (f32, f32) {
        let mut g = self.regressor.lock().expect("PCtx regressor poisoned");
        let predicted = g.predict(batch);
        g.calibrate(batch, actual_ms);
        (predicted, actual_ms)
    }

    /// Predict-only (no calibration). Used for forward-looking queries
    /// such as observe_admission scoring.
    pub fn predict(&self, batch: &BatchForPredictor) -> f32 {
        self.regressor.lock().expect("PCtx regressor poisoned").predict(batch)
    }

    /// Speculatively insert the candidate request's blocks into the
    /// incremental mirror at admission time. Idempotent per request_id.
    pub fn insert_in_flight(&self, request_id: u64, hashes: &[u64], indices: Vec<u64>) {
        self.mirror
            .lock()
            .expect("PCtx mirror poisoned")
            .insert_request(request_id, hashes, indices);
    }

    /// Drop a request's mirror entry (on SSE-reported finish or abort).
    pub fn remove_in_flight(&self, request_id: u64) {
        self.mirror.lock().expect("PCtx mirror poisoned").remove_request(request_id);
    }

    /// Drop blocks reported as evicted by SSE.
    pub fn evict_blocks(&self, block_indices: &[u64]) {
        self.mirror.lock().expect("PCtx mirror poisoned").remove_blocks(block_indices);
    }

    /// Number of in-flight requests currently mirrored. Observability only.
    pub fn in_flight_count(&self) -> usize {
        self.mirror.lock().expect("PCtx mirror poisoned").in_flight_count()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::simulator::config::SimulatorConfig;
    use crate::simulator::predictor::{LinregCorrected, Predictor};

    /// A predictor that returns `value`, useful for closed-form testing of
    /// the linreg correction loop without any CSV/file dependency.
    struct ConstPredictor(f32);
    impl Predictor for ConstPredictor {
        fn predict(&self, _b: &BatchForPredictor) -> f32 {
            self.0
        }
    }

    #[test]
    fn observe_step_calibrates() {
        let mut cfg = SimulatorConfig::default();
        cfg.linreg_warmup = 0;
        cfg.learning_rate = 0.05;
        cfg.linreg_outlier_threshold_ms = 100.0;

        let inner = Arc::new(ConstPredictor(2.0));
        let trained = Box::new(LinregCorrected::new(inner, &cfg));
        let pctx = PCtx::new(trained, 1024);

        let batch = BatchForPredictor::default();

        // Inner says 2.0, truth is 4.0. Many calibrations should nudge the
        // corrected output toward 4.0.
        let mut last_pred = 0.0f32;
        for _ in 0..1500 {
            let (pred, actual) = pctx.observe_step(&batch, 4.0);
            assert!((actual - 4.0).abs() < 1e-6);
            last_pred = pred;
        }
        // After convergence the *new* prediction (read after thousands of
        // calibrations) should be near 4.0. `observe_step` returns the
        // pre-calibration prediction, so the LAST returned `pred` reflects
        // the state right before the last calibrate; close enough for assert.
        assert!(
            (last_pred - 4.0).abs() < 0.05,
            "linreg did not converge inside PCtx: last_pred={}",
            last_pred
        );
    }

    #[test]
    fn mirror_lifecycle() {
        let inner = Arc::new(ConstPredictor(1.0));
        let trained = Box::new(LinregCorrected::new(inner, &SimulatorConfig::default()));
        let pctx = PCtx::new(trained, 1024);

        pctx.insert_in_flight(1, &[10, 20], vec![0, 1]);
        pctx.insert_in_flight(2, &[10, 30], vec![2, 3]);
        assert_eq!(pctx.in_flight_count(), 2);

        pctx.remove_in_flight(1);
        assert_eq!(pctx.in_flight_count(), 1);

        // Evict a leaf belonging to request 2.
        pctx.evict_blocks(&[3]);
        assert_eq!(pctx.in_flight_count(), 0);
    }
}
