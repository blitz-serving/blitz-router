use std::time::Duration;
use std::{
    collections::{HashMap, VecDeque},
    ops::{AddAssign, SubAssign},
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicUsize, Ordering},
        Arc,
    },
};

use crate::{kvcache::PrefixBlockHash, KV_BLOCK_SIZE};

use serde::{Deserialize, Serialize};
use tokio::{
    sync::{Mutex, RwLock},
    time::Instant,
};
use tracing::instrument;

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub(crate) enum ReplicaState {
    /// Inactive replica, w/ dst_mtx
    Inactive,
    /// Normal prefill replica
    Prefill,
    /// Zigzag::New, prefill rep. for fst half
    NewPrefill,
    /// Zigzag::Old, prefill rep. for snd half
    OldPrefill,
    /// Zigzag::New => Normal transient st., w/ model loaded
    RefractoryPrefill,
    /// Normal Decode replica.
    Decode,
    /// Prefill => Null transient st., waiting unfinished kV$ migration
    ShuttingPrefill,
    /// Decode => Null transient st., waiting unfinished decoding req.
    ShuttingDecode,
    /// Shutted down w/o unfinished job left, w/ dst_mtx
    ShuttingNull,
    /// Prefill => Decode emplace, w/o dst_mtx
    MutatingToDecode,
    /// eligible to Prefill, weakest premise
    AusPrefill,
    /// elgible to Decode, weakest premise
    AusDecode,
    /// Marker state for Worker, another coroutine eventually modify this state to Prefill
    LoadingPrefill,
    /// Marker state for Worker, another coroutine eventually modify this state to Decode
    LoadingDecode,
    /// Marker state for Planner
    RdmaSending,
    /// Marker state for Planner
    RdmaLoading,
    /// Marker state for Planner
    NvlinkSending,
    /// NVLink dst st., => Decode when transfer is done
    NvlCasting,
    /// RDMA BCast dst st., => Prefill when transfer is done
    RdmaCasting,
    /// Tanz BCast dst st., ranks join the Tänze
    TanzCasting,
}

/// Logic of Transition System
///
/// aRb := a -> b
/// reflexitivity := aRa
///
/// TODO: sanity checker left for further work
// macro_rules! match_trans {
//     ($value:expr, $($pattern:pat => $result:expr),*) => {
//         match $value {
//             $($pattern => $result,)*
//         }
//     };
// }
// impl PartialOrd for ReplicaState {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         use ReplicaState::*;
//         use std::cmp::Ordering;
//         // reflexitivity
//         match match_trans!((self, other),
//             (Inactive, Inactive) | (Prefill, Prefill) | (Decode, Decode) => Some(Ordering::Equal),
//             (Inactive, LoadingDecode) | (Inactive, LoadingPrefill) | (Inactive, RdmaCastingPrefill {..}) | (Inactive, RdmaCastingDecode{..}) | (Inactive, NvlCastingDecode{..}) | (Inactive, NvlCastingPrefill{..}) | (Inactive, WalzerCastingNull{..}) => Some(Ordering::Less),
//             (Prefill, NewPrefill) | (Prefill, OldPrefill) | (Prefill, MutatingToDecode) | (Prefill, ShuttingPrefill) => Some(Ordering::Less),
//             (Decode, )
//         )
//     }
// }

#[derive(Debug)]
pub(crate) struct SystemMetric {
    pub(crate) prefill_tokens: AtomicUsize,
    pub(crate) decode_tokens: AtomicUsize,
    pub(crate) loop_counts: Vec<AtomicUsize>,
    pub(crate) token_in_queue: AtomicUsize,
}

impl SystemMetric {
    pub(crate) fn new() -> Self {
        Self {
            prefill_tokens: AtomicUsize::new(0),
            decode_tokens: AtomicUsize::new(0),
            loop_counts: (0..32).map(|_| AtomicUsize::new(0)).collect(),
            token_in_queue: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Flow {
    pub(crate) flow_out: HashMap<usize, AtomicUsize>,
    pub(crate) flow_in: HashMap<usize, AtomicUsize>,
}

impl Flow {
    pub(crate) fn new() -> Self {
        Self { flow_out: HashMap::new(), flow_in: HashMap::new() }
    }

    pub(crate) fn append_token(&mut self, replica_index: usize, token_num: usize) {
        // tracing::info!("Migrating token num {}", token_num);
        self.flow_out
            .entry(replica_index)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add((token_num as f32 * 0.5) as usize, Ordering::AcqRel);
    }

    pub(crate) fn recv_token(&mut self, replica_index: usize, token_num: usize) {
        self.flow_in
            .entry(replica_index)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add((token_num as f32 * 0.5) as usize, Ordering::AcqRel);
    }

    pub(crate) fn append_param(&mut self, replica_index: usize, param_size_in_gb: usize) {
        self.flow_out
            .entry(replica_index)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(param_size_in_gb * 1024, Ordering::AcqRel);
    }

    pub(crate) fn recv_param(&mut self, replica_index: usize, param_size_in_gb: usize) {
        self.flow_in
            .entry(replica_index)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(param_size_in_gb * 1024, Ordering::AcqRel);
    }

    pub(crate) fn get_all(&self) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let mut ret_flow_out = Vec::new();
        let mut ret_flow_in = Vec::new();
        for (&replica_index, flow) in self.flow_out.iter() {
            let flow = flow.load(Ordering::Acquire);
            ret_flow_out.append(&mut vec![(replica_index, flow)]);
        }
        for (&replica_index, flow) in self.flow_in.iter() {
            let flow = flow.load(Ordering::Acquire);
            ret_flow_in.append(&mut vec![(replica_index, flow)]);
        }
        (ret_flow_out, ret_flow_in)
    }

    pub(crate) fn clear(&mut self) {
        self.flow_out.clear();
        self.flow_in.clear();
    }
}

// ----------------------------------- //
// ====== END project BlitzScale ===== //
// ----------------------------------- //

// ----------------------------------- //
// ====== BEGIN project LMetric ====== //
// ----------------------------------- //

/// Only turn on throttle when filled with enough requests
pub(crate) static THROTTLE_THLD: usize = 999;
/// Colocated replica keeps tps >= TPS_THRESHOLD,
/// or it will be throttled from add prefill requests
pub(crate) static TPS_THRESHOLD: usize = 10;
/// Colocated replica keeps tpot_mili <= TPOT_THRESHOLD
/// or it will be throttled from add prefill requests
pub(crate) static TPOT_THRESHOLD: usize = 50;
/// Parameter for prefill token/s EMA updation
static PREFILL_TKN_FREQ_EMA_GAMMA: f32 = 0.75;
/// Parameter for TBT EMA updation
static TBT_EMA_GAMMA: f32 = 0.5;
/// Prefill token bound, used in JBSQ(1), set to 2⨉ CP size
pub(crate) static WAITINGT_PREFILL_TOKEN_BOUND: usize = 2048;
/// Parameters for Bailian
pub(crate) static BAILIAN_ALPHA: f32 = 0.33; // prefix cache hit block
pub(crate) static BAILIAN_BETA: f32 = 0.33; // num requests on instance
pub(crate) static BAILIAN_GAMMA: f32 = 0.33; // num tokens on instance

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
    /// ✓
    pub bs_dec: usize,
    /// Decrement in number of requests that have been assinged but not started
    /// ✓
    pub waiting_reqs_dec: usize,
    /// Decrement in the token sum of all waiting (to prefill) request
    /// ✓
    pub prefill_tokens_dec: isize,
    /// Decrement in the sum of tokens of all requests
    /// ✓
    pub all_tokens_inc: isize,
    /// New TBT, must be set
    /// ✓
    pub tbt: f32,
    /// New TPOT, calculated delta iteratively, and derive requestwise
    /// ✓
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
            self.bs -= rhs.bs_dec;
            self.waiting_reqs -= rhs.waiting_reqs_dec;
            self.prefill_tokens -= rhs.prefill_tokens_dec;
        }
        self.all_tokens = (self.all_tokens as isize + rhs.all_tokens_inc) as usize;
        // first-order estimation
        self.time_of_left_prefill = self.time_of_left_prefill - rhs.tbt;
        // second-order estimation?

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

// ----------------------------------- //
// ======= END project LMetric ======= //
// ----------------------------------- //

// ----------------------------------- //
// ===== BEGIN project BlitzScale ==== //
// ----------------------------------- //

#[derive(Debug)]
pub(crate) struct ReplicaMetric {
    pub(crate) block_size: u32,
    used_blocks: AtomicU32,
    model_loaded: AtomicBool,

    #[allow(unused)]
    replica_index: usize,
    /// lock() <-> act as decode
    pub(crate) dst_mutex: Arc<Mutex<()>>,
    /// replica state for event loop
    pub(crate) state: RwLock<ReplicaState>,
    /// ongoing Zigzag partial layer migration tasks.
    pub(crate) flying_partial_migration_batches: Arc<AtomicI64>,
}

#[allow(unused)]
impl ReplicaMetric {
    pub(crate) fn new(
        model_loaded: bool,
        replica_index: usize,
        state: RwLock<ReplicaState>,
        dst_mutex: Arc<Mutex<()>>,
    ) -> Self {
        Self {
            block_size: KV_BLOCK_SIZE,
            used_blocks: AtomicU32::new(0),
            model_loaded: AtomicBool::new(model_loaded),
            replica_index,
            dst_mutex,
            state,
            flying_partial_migration_batches: Arc::new(AtomicI64::new(0)),
        }
    }

    pub(crate) fn set_used_blocks(&self, used_blocks: u32) {
        self.used_blocks.store(used_blocks, Ordering::Release);
    }

    pub(crate) fn get_used_blocks(&self) -> u32 {
        self.used_blocks.load(Ordering::Acquire)
    }

    #[instrument(skip_all)]
    pub(crate) fn add_used_blocks(&self, used_blocks: u32) {
        self.used_blocks.fetch_add(used_blocks, Ordering::AcqRel);
        // tracing::info!(
        //     "Add used blocks: {}",
        //     used_blocks,
        // );
    }

    #[instrument(skip_all)]
    pub(crate) fn sub_used_blocks(&self, used_blocks: u32) {
        self.used_blocks.fetch_sub(used_blocks, Ordering::AcqRel);
        // tracing::info!(
        //     "Sub used blocks: {}",
        //     used_blocks,
        // );
    }

    pub(crate) fn set_model_loaded(&self, model_loaded: bool) {
        self.model_loaded.store(model_loaded, Ordering::Release);
    }

    pub(crate) fn is_model_loaded(&self) -> bool {
        self.model_loaded.load(Ordering::Acquire)
    }

    pub(crate) fn add_partial_migration_cnt(&self) {
        self.flying_partial_migration_batches.fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn sub_partial_migration_cnt(&self) {
        self.flying_partial_migration_batches.fetch_sub(1, Ordering::AcqRel);
    }

    pub(crate) fn get_partial_migration_cnt(&self) -> i64 {
        self.flying_partial_migration_batches.load(Ordering::Acquire)
    }
}
