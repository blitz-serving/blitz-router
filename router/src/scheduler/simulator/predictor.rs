// Predictor traits — L2 cost oracle (offline `Predictor` + online
// `Corrector`). Two type parameters per concern:
//
//   * `Predictor`     — pure offline model: features → ms scalar.
//   * `Corrector`     — online correction strategy: (raw_ms, actual_ms) → ms.
//   * `RegressionalPredictor<P, C>` — wraps both into a `TrainedPredictor`.
//
// Concrete `Corrector` impls today:
//
//   * `LinregCorrector`  — affine `w0 · raw + w1` with SGD updates,
//     warmup, and outlier rejection.
//   * `NullCorrector`    — passthrough (`correct(raw) = raw`,
//     `calibrate(_, _) = no-op`). Useful as a bypass switch and for
//     A/B'ing the contribution of online correction.
//
// See `docs/predictor/three-layer-architecture.md` §2 and
// `docs/predictor/behavior.md` for the locked naming and the rationale.
// Returns scalar `f32` (milliseconds) directly to avoid per-call heap
// allocation under the <1ms query budget.

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

/// Trained wrapper — exposes a single `predict(batch) -> ms` plus a
/// `calibrate(batch, actual)` callback the SSE consumer drives once per
/// engine forward step. The concrete implementation owns whatever inner
/// `Predictor` + `Corrector` pair is in use.
pub trait TrainedPredictor: Send + Sync {
    fn predict(&self, batch: &BatchForPredictor) -> f32;
    fn calibrate(&mut self, batch: &BatchForPredictor, actual: f32);
}

/// Online correction strategy. `correct` is applied at predict time;
/// `calibrate` is applied at observe time and updates the strategy's
/// internal state from the (raw, actual) pair the wrapper supplies.
pub trait Corrector: Send + Sync {
    fn correct(&self, raw_ms: f32) -> f32;
    fn calibrate(&mut self, raw_ms: f32, actual_ms: f32);
}

/// Passthrough corrector: `correct(raw) = raw`, `calibrate` is a no-op.
/// Used as the bypass switch for diagnosing offline-grid quality and as
/// the A/B baseline against `LinregCorrector`.
pub struct NullCorrector;

impl Corrector for NullCorrector {
    fn correct(&self, raw_ms: f32) -> f32 {
        raw_ms
    }
    fn calibrate(&mut self, _raw: f32, _actual: f32) {}
}

/// Online linear-regression correction: `corrected = w0 · raw + w1`. Updates
/// `(w0, w1)` via stochastic gradient descent on each calibration sample after
/// a warmup period. Outlier rejection prevents large errors (e.g. cold-start
/// vLLM spikes) from poisoning the weights.
pub struct LinregCorrector {
    weight: (f32, f32),
    learning_rate: f32,
    warmup_remaining: usize,
    outlier_threshold_ms: f32,
}

impl LinregCorrector {
    pub fn new(config: &SimulatorConfig) -> Self {
        Self {
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

impl Corrector for LinregCorrector {
    fn correct(&self, raw_ms: f32) -> f32 {
        self.weight.0 * raw_ms + self.weight.1
    }

    fn calibrate(&mut self, raw_ms: f32, actual_ms: f32) {
        if self.learning_rate == 0.0 {
            return;
        }
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            return;
        }
        let corrected = self.weight.0 * raw_ms + self.weight.1;
        let error = actual_ms - corrected;
        if error.abs() > self.outlier_threshold_ms {
            return;
        }
        // Gradient of (actual - (w0·raw + w1))² with respect to (w0, w1):
        //   ∂L/∂w0 = -2 · error · raw
        //   ∂L/∂w1 = -2 · error
        // We absorb the factor of 2 into the learning rate convention.
        self.weight.0 += self.learning_rate * error * raw_ms;
        self.weight.1 += self.learning_rate * error;
    }
}

/// Wraps an inner offline `Predictor` with an online `Corrector` to
/// produce a `TrainedPredictor`. Generics propagate to the storage type;
/// the trait-object surface (`Box<dyn TrainedPredictor>`) at the PCtx
/// field is unchanged.
pub struct RegressionalPredictor<P: Predictor + ?Sized, C: Corrector> {
    inner: Arc<P>,
    corrector: C,
}

impl<P: Predictor + ?Sized, C: Corrector> RegressionalPredictor<P, C> {
    pub fn new(inner: Arc<P>, corrector: C) -> Self {
        Self { inner, corrector }
    }
}

impl<P: Predictor + ?Sized, C: Corrector> TrainedPredictor
    for RegressionalPredictor<P, C>
{
    fn predict(&self, batch: &BatchForPredictor) -> f32 {
        self.corrector.correct(self.inner.predict(batch))
    }

    fn calibrate(&mut self, batch: &BatchForPredictor, actual: f32) {
        let raw = self.inner.predict(batch);
        self.corrector.calibrate(raw, actual);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockPredictor(f32);
    impl Predictor for MockPredictor {
        fn predict(&self, _b: &BatchForPredictor) -> f32 {
            self.0
        }
    }

    /// `RegressionalPredictor` wrapping `NullCorrector` is a passthrough:
    /// `predict` returns the inner prediction unchanged and `calibrate`
    /// has no observable effect.
    #[test]
    fn regressional_with_null_corrector_returns_raw() {
        let inner = Arc::new(MockPredictor(2.5));
        let mut wrapped = RegressionalPredictor::new(inner.clone(), NullCorrector);
        let batch = BatchForPredictor::default();

        // Several batches return raw inner prediction unchanged.
        for _ in 0..5 {
            let p = wrapped.predict(&batch);
            assert!((p - 2.5).abs() < 1e-6, "passthrough must return raw: {}", p);
        }

        // Calibration is a no-op — predict still returns raw.
        wrapped.calibrate(&batch, 100.0);
        wrapped.calibrate(&batch, -50.0);
        wrapped.calibrate(&batch, 0.0);
        let after = wrapped.predict(&batch);
        assert!(
            (after - 2.5).abs() < 1e-6,
            "NullCorrector::calibrate must not shift the prediction: {}",
            after
        );
    }

    /// End-to-end convergence test: drive the SGD loop against a known
    /// `actual` until the wrapper's prediction approaches the truth.
    /// Mirrors the legacy `LinregCorrected` convergence test pattern.
    #[test]
    fn regressional_with_linreg_corrector_matches_legacy_linreg_corrected() {
        let mut cfg = SimulatorConfig::default();
        cfg.linreg_warmup = 0;
        cfg.learning_rate = 0.05;
        cfg.linreg_outlier_threshold_ms = 100.0;

        let inner = Arc::new(MockPredictor(2.0));
        let mut wrapped =
            RegressionalPredictor::new(inner.clone(), LinregCorrector::new(&cfg));
        let batch = BatchForPredictor::default();

        // Inner says 2.0; truth is 4.0. After many calibration steps the
        // correction should bring the corrected output close to 4.0.
        for _ in 0..2000 {
            let _p = wrapped.predict(&batch);
            wrapped.calibrate(&batch, 4.0);
        }
        let final_p = wrapped.predict(&batch);
        assert!(
            (final_p - 4.0).abs() < 0.05,
            "RegressionalPredictor<_, LinregCorrector> did not converge: final={}",
            final_p
        );
    }
}
