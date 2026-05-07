// Predictor traits — best-of-both merge of the in-house design with
// everparadise's `predictor.rs`. Two traits, no per-op breakdown trait, no
// double indirection. Returns scalar `f32` (milliseconds) directly to avoid
// per-call heap allocation under the <1ms query budget.

use std::sync::Arc;

use super::batch::BatchForPredictor;
use super::config::SimulatorConfig;

/// Inner regressor — pure: features → scalar latency in milliseconds.
///
/// Implementors are offline-trained (e.g. Vidur RandomForest, llm-d XGBoost)
/// and serve runtime queries via interpolation/lookup over a precomputed grid.
pub trait Predictor: Send + Sync {
    fn predict(&self, batch: &BatchForPredictor) -> f32;
}

/// Trained wrapper — applies online linear-regression correction
/// (`actual ≈ w0 · raw + w1`) on top of an inner `Predictor`.
///
/// `calibrate` takes the batch (NOT the prediction) and recomputes the raw
/// inner prediction internally. The cost is one extra hash-map lookup per
/// calibration step, which is microseconds for grid-lookup backends like
/// VidurRfPredictor — acceptable to avoid the alternative API surface where
/// the caller must hand back the raw prediction explicitly.
pub trait TrainedPredictor: Send + Sync {
    fn predict(&self, batch: &BatchForPredictor) -> f32;
    fn calibrate(&mut self, batch: &BatchForPredictor, actual: f32);
}

/// Online linear-regression correction: `corrected = w0 · raw + w1`. Updates
/// `(w0, w1)` via stochastic gradient descent on each calibration sample after
/// a warmup period. Outlier rejection prevents large errors (e.g. cold-start
/// vLLM spikes) from poisoning the weights.
///
/// Mirrors everparadise's `LinearRegressionPredictor` (predictor.rs:127–216),
/// but `calibrate` accepts the already-computed prediction so the caller does
/// not re-invoke the inner regressor on the calibration path.
pub struct LinregCorrected<P: Predictor + ?Sized> {
    inner: Arc<P>,
    weight: (f32, f32),
    learning_rate: f32,
    warmup_remaining: usize,
    outlier_threshold_ms: f32,
}

impl<P: Predictor + ?Sized> LinregCorrected<P> {
    pub fn new(inner: Arc<P>, config: &SimulatorConfig) -> Self {
        Self {
            inner,
            weight: (1.0, 0.0),
            learning_rate: config.learning_rate,
            warmup_remaining: config.linreg_warmup,
            outlier_threshold_ms: config.linreg_outlier_threshold_ms,
        }
    }

    /// Current correction weights `(w0, w1)`. Exposed for observability.
    pub fn weights(&self) -> (f32, f32) {
        self.weight
    }
}

impl<P: Predictor + ?Sized> TrainedPredictor for LinregCorrected<P> {
    fn predict(&self, batch: &BatchForPredictor) -> f32 {
        let raw = self.inner.predict(batch);
        self.weight.0 * raw + self.weight.1
    }

    fn calibrate(&mut self, batch: &BatchForPredictor, actual: f32) {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            return;
        }
        let raw = self.inner.predict(batch);
        let corrected = self.weight.0 * raw + self.weight.1;
        let error = actual - corrected;
        if error.abs() > self.outlier_threshold_ms {
            return;
        }
        // Gradient of (actual - (w0·raw + w1))² with respect to (w0, w1):
        //   ∂L/∂w0 = -2 · error · raw
        //   ∂L/∂w1 = -2 · error
        // We absorb the factor of 2 into the learning rate convention.
        self.weight.0 += self.learning_rate * error * raw;
        self.weight.1 += self.learning_rate * error;
    }
}
