use std::sync::OnceLock;
use std::time::{Duration, Instant};
use std::ops::{AddAssign, SubAssign};

use super::kvcache::PrefixBlockHash;

use serde::Serialize;

/// Parameter for TBT EMA updation
static TBT_EMA_GAMMA: f32 = 0.5;
/// Prefill token bound, used in JBSQ(1), set to 2⨉ CP size
pub(crate) static WAITINGT_PREFILL_TOKEN_BOUND: usize = 2048;
/// Parameters for Bailian (set once from CLI in `main.rs` via
/// [`init_bailian_params`]; defaults 0.7 / 0.15 / 0.15 if unset).
/// The statics themselves are unconditional so `policies/bailian.rs` (which
/// is compiled regardless of feature flags, like every other policy module)
/// can read them; only the `init_bailian_params` setter and the
/// `--bailian-{alpha,beta,gamma}` CLI surface are gated by
/// `feature = "bailian-impl-q"`.
pub(crate) static BAILIAN_ALPHA: OnceLock<f32> = OnceLock::new();
pub(crate) static BAILIAN_BETA: OnceLock<f32> = OnceLock::new();
pub(crate) static BAILIAN_GAMMA: OnceLock<f32> = OnceLock::new();

/// Install the Bailian scoring weights from CLI flags. Called once at
/// startup; subsequent calls are no-ops (the values cannot change at
/// runtime).
#[cfg(feature = "bailian-impl-q")]
pub fn init_bailian_params(alpha: f32, beta: f32, gamma: f32) {
    let _ = BAILIAN_ALPHA.set(alpha);
    let _ = BAILIAN_BETA.set(beta);
    let _ = BAILIAN_GAMMA.set(gamma);
}
/// Preble branch-split threshold on
/// `(global_match_blocks * block_size) / |req|`. Set once from CLI
/// in `main.rs` via [`init_preble_params`]; default `0.5` if unset
/// (the abstract spec value, `docs/preble-design.md` §QUERY).
/// The static itself is unconditional so `policies/preble/mod.rs`
/// can read it; only the `init_preble_params` setter and the
/// `--preble-match-ratio-threshold` CLI surface are gated by
/// `feature = "preble-q"`.
pub(crate) static PREBLE_MATCH_RATIO_T: OnceLock<f32> = OnceLock::new();

/// Install the Preble branch-split match-ratio threshold from a CLI
/// flag. Called once at startup; subsequent calls are no-ops.
/// Shared by all three Preble flavours (`preble-q` / `preble-bs-q` /
/// `preble-tps-q`) since they share the KV$-aware-branch filter.
#[cfg(any(feature = "preble-q", feature = "preble-bs-q", feature = "preble-tps-q"))]
pub fn init_preble_params(match_ratio_threshold: f32) {
    let _ = PREBLE_MATCH_RATIO_T.set(match_ratio_threshold);
}

/// Preble-TPS sliding-window duration in seconds. CLI-tunable via
/// `--preble-tps-window-secs`; defaults to 180s (3 min) if unset.
/// Shorter windows react faster to load shifts; longer windows give
/// more statistical noise immunity.
pub(crate) static PREBLE_TPS_WINDOW_SECS: OnceLock<u64> = OnceLock::new();

/// Preble-TPS idle-period compensation rate in forward-steps per
/// second. CLI-tunable via `--preble-tps-decode-fps`; defaults to 120
/// (pure-decode peak rate for a saturated engine). When an engine
/// transitions from idle (bs=0) to busy (bs>0), the gap is
/// retroactively credited as if the engine had been ticking at this
/// rate — so a recently-idle engine looks competitive with a
/// continuously-busy peer, instead of being penalised for having no
/// real samples in its window.
pub(crate) static PREBLE_TPS_DECODE_FPS: OnceLock<f32> = OnceLock::new();

/// Install the Preble-TPS tunables from CLI flags. Called once at
/// startup; subsequent calls are no-ops.
#[cfg(feature = "preble-tps-q")]
pub fn init_preble_tps_params(window_secs: u64, decode_fps: f32) {
    let _ = PREBLE_TPS_WINDOW_SECS.set(window_secs);
    let _ = PREBLE_TPS_DECODE_FPS.set(decode_fps);
}
/// llm-d load-aware-scorer's queue-depth threshold (default in upstream)
pub(crate) static LOAD_AWARE_QUEUE_T: f32 = 128.0;
/// most-hit-load-q (llm-d precise-prefix-cache + load-aware combo)
pub(crate) static MOST_HIT_LOAD_W_HIT: f32 = 10.0;
pub(crate) static MOST_HIT_LOAD_W_LOAD: f32 = 1.0;
/// most-hit-load-active-q (above + kv-cache-utilization)
pub(crate) static MOST_HIT_LOAD_ACTIVE_W_HIT: f32 = 10.0;
pub(crate) static MOST_HIT_LOAD_ACTIVE_W_LOAD: f32 = 1.0;
pub(crate) static MOST_HIT_LOAD_ACTIVE_W_KV: f32 = 1.0;

fn serialize_f32_3<S>(x: &f32, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    s.serialize_str(&format!("{:.3}", x))
}

#[derive(Debug, Serialize)]
pub(crate) struct LMetric {
    /// Active request number, comform with Bailian's terminology.
    pub bs: usize,
    /// Number of requests that have been assinged but not started,
    /// conform with vllm's terminology.
    /// Thus `running_reqs` in vLLM's terminology is `bs - waitting_reqs`
    pub waiting_reqs: usize,
    /// Token sum of all waiting (to prefill) request
    pub prefill_tokens: isize,
    /// All tokens of all requests, just summation
    pub all_tokens: usize,
    /// Number of decoding times per second, conform with Bailian's terminology
    pub tps: usize,
    #[serde(skip_serializing)]
    tps_window: radixtree::SlidingWindow<(), radixtree::Count>,
    /// TPOT averaged on all running requests within instance, in milisecond
    #[serde(serialize_with = "serialize_f32_3")]
    pub tpot: f32,
    /// TBT, in seconds (set from `Duration::as_secs_f32`).
    #[serde(serialize_with = "serialize_f32_3")]
    pub tbt: f32,
}

impl Default for LMetric {
    fn default() -> Self {
        Self {
            bs: 0,
            waiting_reqs: 0,
            prefill_tokens: 0,
            all_tokens: 0,
            tps: 0,
            tps_window: radixtree::SlidingWindow::new(
                Duration::from_secs(1),
                radixtree::Count,
            ),
            tpot: f32::NAN,
            tbt: 0.,
        }
    }
}

impl LMetric {
    pub fn to_jsonl_with_id(&self, replica_index: usize) -> String {
        let mut obj = serde_json::to_value(self).unwrap();
        if let serde_json::Value::Object(ref mut map) = obj {
            map.insert("id".to_string(), serde_json::json!(replica_index));
        }
        serde_json::to_string(&obj).unwrap()
    }
}

unsafe impl Send for LMetric {}
unsafe impl Sync for LMetric {}

/// All **necessary** information to perform LMetric stepping for SSE
#[derive(Debug)]
pub(crate) struct LMetricDec {
    /// Decrement in request number
    pub bs_dec: usize,
    /// Decrement in number of requests that have been assinged but not started
    pub waiting_reqs_dec: usize,
    /// Decrement in the token sum of all waiting (to prefill) request
    pub prefill_tokens_dec: isize,
    /// Decrement in the sum of tokens of all requests
    pub all_tokens_inc: isize,
    /// New TBT, must be set
    pub tbt: f32,
    /// New TPOT, calculated delta iteratively, and derive requestwise
    pub tpot: f32,
}

impl LMetricDec {
    pub fn new(tbt: &Duration) -> Self {
        LMetricDec {
            bs_dec: 0,
            waiting_reqs_dec: 0,
            prefill_tokens_dec: 0,
            all_tokens_inc: 0,
            tbt: tbt.as_secs_f32(),
            tpot: 0.,
        }
    }
}

impl SubAssign<LMetricDec> for LMetric {
    fn sub_assign(&mut self, rhs: LMetricDec) {
        // Decrements
        if cfg!(debug_assertions) {
            let mut underflow;
            (self.bs, underflow) = self.bs.overflowing_sub(rhs.bs_dec);
            debug_assert!(
                !underflow,
                "Old batch size = {}; new batch size = {}",
                self.bs + rhs.bs_dec,
                self.bs
            );
            (self.waiting_reqs, underflow) =
                self.waiting_reqs.overflowing_sub(rhs.waiting_reqs_dec);
            debug_assert!(
                !underflow,
                "Old waiting requests = {}; new waiting requests = {}",
                self.waiting_reqs + rhs.waiting_reqs_dec,
                self.waiting_reqs
            );
            (self.prefill_tokens, underflow) =
                self.prefill_tokens.overflowing_sub(rhs.prefill_tokens_dec);
            debug_assert!(
                !underflow,
                "Old prefill tokens = {}; new prefill tokens = {}",
                self.prefill_tokens + rhs.prefill_tokens_dec,
                self.prefill_tokens
            );
        } else {
            self.bs = self.bs.saturating_sub(rhs.bs_dec);
            self.waiting_reqs = self.waiting_reqs.saturating_sub(rhs.waiting_reqs_dec);
            self.prefill_tokens = self.prefill_tokens.saturating_sub(rhs.prefill_tokens_dec);
        }
        self.all_tokens = (self.all_tokens as isize + rhs.all_tokens_inc) as usize;

        // EMAs
        self.tbt = self.tbt * (1. - TBT_EMA_GAMMA) + rhs.tbt * TBT_EMA_GAMMA;

        // Replacements
        // TPOT
        self.tpot = rhs.tpot / self.bs as f32;
        // TPS: count of decoding ticks within a 1 s sliding window.
        // `SlidingWindow::push` + `len_at` performs the same
        // push-back-then-pop-old-front pattern; lazy expire on read.
        let now = Instant::now();
        self.tps_window.push(now, ());
        self.tps = self.tps_window.len_at(now);
    }
}

#[derive(Debug)]
pub(crate) struct LMetricInc {
    pub bs_inc: usize,
    pub waiting_reqs_inc: usize,
    pub prefill_tokens_inc: usize,
    pub all_tokens_inc: usize,
}

impl AddAssign<LMetricInc> for LMetric {
    fn add_assign(&mut self, rhs: LMetricInc) {
        self.bs += rhs.bs_inc;
        self.waiting_reqs += rhs.waiting_reqs_inc;
        self.prefill_tokens += rhs.prefill_tokens_inc as isize;
        self.all_tokens += rhs.all_tokens_inc;
    }
}

pub(crate) struct ScheduleContext {
    pub lmetric: LMetric,
    pub block_hash: PrefixBlockHash,
}
