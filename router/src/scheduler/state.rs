use std::sync::OnceLock;
use std::time::Duration;
use std::collections::VecDeque;
use std::ops::{AddAssign, SubAssign};

use super::kvcache::PrefixBlockHash;

use serde::Serialize;
use tokio::time::Instant;

/// Parameter for prefill token/s EMA updation
static PREFILL_TKN_FREQ_EMA_GAMMA: f32 = 0.75;
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
/// PolyServe SLO thresholds (set once from CLI in `main.rs` via
/// [`init_polyserve_params`]). The policy consumes both thresholds as
/// milliseconds and compares them against simulator projections.
pub(crate) static POLYSERVE_TTFT_SLO_MS: OnceLock<f32> = OnceLock::new();
pub(crate) static POLYSERVE_TPOT_SLO_MS: OnceLock<f32> = OnceLock::new();

/// Install the Bailian scoring weights from CLI flags. Called once at
/// startup; subsequent calls are no-ops (the values cannot change at
/// runtime).
#[cfg(feature = "bailian-impl-q")]
pub fn init_bailian_params(alpha: f32, beta: f32, gamma: f32) {
    let _ = BAILIAN_ALPHA.set(alpha);
    let _ = BAILIAN_BETA.set(beta);
    let _ = BAILIAN_GAMMA.set(gamma);
}
/// Install the PolyServe TTFT / TPOT SLO thresholds from CLI flags.
/// Called once at startup; subsequent calls are no-ops.
#[cfg(feature = "polyserve-q")]
pub fn init_polyserve_params(ttft_slo_ms: f32, tpot_slo_ms: f32) {
    let _ = POLYSERVE_TTFT_SLO_MS.set(ttft_slo_ms);
    let _ = POLYSERVE_TPOT_SLO_MS.set(tpot_slo_ms);
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
    tps_timestamps: VecDeque<Instant>,
    /// TPOT averaged on all running requests within instance, in milisecond
    #[serde(serialize_with = "serialize_f32_3")]
    pub tpot: f32,
    /// TBT, in milisecond
    #[serde(serialize_with = "serialize_f32_3")]
    pub tbt: f32,
    /// Estimated waiting to prefill time, in milisecond
    #[serde(skip_serializing)]
    pub time_of_left_prefill: f32,
    /// Number of token prefilled per milisecond, an estimation, just work around
    #[serde(skip_serializing)]
    pub prefill_token_freq: f32,
}

impl Default for LMetric {
    fn default() -> Self {
        Self {
            bs: 0,
            waiting_reqs: 0,
            prefill_tokens: 0,
            all_tokens: 0,
            tps: 0,
            tps_timestamps: VecDeque::with_capacity(32),
            tpot: f32::NAN,
            tbt: 0.,
            time_of_left_prefill: f32::default(),
            prefill_token_freq: f32::default(),
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
        // first-order estimation
        self.time_of_left_prefill = self.time_of_left_prefill - rhs.tbt;

        // EMAs
        self.tbt = self.tbt * (1. - TBT_EMA_GAMMA) + rhs.tbt * TBT_EMA_GAMMA;
        self.prefill_token_freq = if rhs.prefill_tokens_dec == 0 {
            // Decoding only, keep prefill_token_freq estimation unchanged
            self.prefill_token_freq
        } else {
            let mut prefill_token_freq = rhs.prefill_tokens_dec as f32 / rhs.tbt;
            prefill_token_freq = self.prefill_token_freq * (1. - PREFILL_TKN_FREQ_EMA_GAMMA)
                + prefill_token_freq * PREFILL_TKN_FREQ_EMA_GAMMA;
            prefill_token_freq
        };

        // Replacements
        // TPOT
        self.tpot = rhs.tpot / self.bs as f32;
        // TPS: wrap around new 1 second interval
        let end = tokio::time::Instant::now();
        let d = tokio::time::Duration::from_secs(1);
        let begin = end - d;
        self.tps_timestamps.push_back(end);
        while let Some(&t) = self.tps_timestamps.front() {
            if t < begin {
                self.tps_timestamps.pop_front();
            } else {
                break;
            }
        }
        self.tps = self.tps_timestamps.len();
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
        self.time_of_left_prefill += self.prefill_token_freq * rhs.prefill_tokens_inc as f32;
    }
}

pub(crate) struct ScheduleContext {
    pub lmetric: LMetric,
    pub block_hash: PrefixBlockHash,
}
