use core::panic;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::Arc;
use std::vec;

use nohash_hasher::IntMap;
use tracing_subscriber::field::debug;

use crate::simulator::batch::{BatchForPredictor, Request};
use crate::simulator::config::SimulationConfig;
use crate::simulator::kvcache::KVCacheManager;
use crate::simulator::{batch, VllmMetric, VllmRequestStatus};
use crate::SystemMetric;

use std::time::SystemTime;

// #[derive(Clone)]
pub struct SchedulerState {
    // Add fields relevant to the scheduler state here
    pub kvcache_manager: KVCacheManager,

    // clone
    pub waiting_reqs: VecDeque<Request>,
    pub running_reqs: HashMap<u64, Request>,
    pub commit_request: HashSet<u64>,
    pub config: Arc<SimulationConfig>,
    pub replace_bit: bool,
    pub num_decode_computed_tokens: Vec<usize>,
    pub prefill_request_ids: Vec<u64>,

    pub checkpoint_waiting: Option<VecDeque<Request>>,
    pub checkpoint_running: Option<HashMap<u64, Request>>,
}

// impl PartialEq for SchedulerState {
//     fn eq(&self, other: &Self) -> bool {
//         self.kvcache_manager == other.kvcache_manager
//             && self.waiting_reqs == other.waiting_reqs
//             && self.running_reqs == other.running_reqs
//             && self.checkpoint_waiting == other.checkpoint_waiting
//             && self.checkpoint_running == other.checkpoint_running
//         // ✅ 注意：config 被故意跳过
//     }
// }

// impl Eq for SchedulerState {}

pub enum ScheduleError {
    KVCacheAllocationFailed,
}

pub enum PredictState<'a> {
    Simulate(&'a VecDeque<(BatchForPredictor, f32)>),
    Predict(&'a VecDeque<(BatchForPredictor, f32)>),
    InitialPredict(BatchForPredictor),
    Tempt,
}
// implement Debug trait for PredictState
impl<'a> std::fmt::Debug for PredictState<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictState::Simulate(_) => write!(f, "PredictState::Simulate"),
            PredictState::Predict(_) => write!(f, "PredictState::Predict"),
            PredictState::InitialPredict(batch) => {
                write!(f, "PredictState::InitialPredict({:?})", batch)
            }
            PredictState::Tempt => write!(f, "PredictState::Tempt"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ScheduleCacheOperation {
    Cache,
    NotCache,
    Replace,
}

impl SchedulerState {
    pub fn new(config: Arc<SimulationConfig>) -> Self {
        SchedulerState {
            kvcache_manager: KVCacheManager::new(config.clone()),
            waiting_reqs: VecDeque::new(),
            running_reqs: HashMap::new(),
            config: config,
            checkpoint_waiting: None,
            checkpoint_running: None,
            num_decode_computed_tokens: Vec::new(),
            prefill_request_ids: Vec::new(),
            commit_request: HashSet::new(),
            replace_bit: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.waiting_reqs.is_empty() && self.running_reqs.is_empty()
    }

    /// Enable simulation mode (copy-on-write) for kvcache_manager
    pub fn enable_simulation_mode(&mut self) {
        debug_assert!(
            self.checkpoint_waiting.is_none() && self.checkpoint_running.is_none(),
            "Simulation mode already enabled, cp_waiting_len: {:?}, cp_running_len: {:?}",
            self.checkpoint_waiting.as_ref().unwrap().len(),
            self.checkpoint_running.as_ref().unwrap().len(),
        );
        self.kvcache_manager.enable_simulation_mode();
        self.checkpoint();
    }

    /// Disable simulation mode and clear all COW state
    pub fn disable_simulation_mode(&mut self) {
        debug_assert!(self.checkpoint_waiting.is_some() && self.checkpoint_running.is_some());
        self.kvcache_manager.disable_simulation_mode();
        self.redo();
    }

    pub fn flush_existing_state(&mut self) {
        if self.checkpoint_waiting.is_some() && self.checkpoint_running.is_some() {
            self.kvcache_manager.disable_simulation_mode();
            self.redo();
        }
    }

    pub fn add_request(&mut self, mut request: Request) {
        // tracing::debug!(
        //     "request_id: {}, hashes: {:?}",
        //     request.request_id,
        //     request.hashes.as_ref().unwrap()
        // );
        self.kvcache_manager.add_request(request.request_id, request.hashes.take().unwrap());
        self.commit_request.insert(request.request_id);

        let num_new_blocks =
            Self::compute_new_block(request.prompt_len as usize, &request, &self.config);
        self.kvcache_manager.allocate_block(request.request_id, num_new_blocks);
        self.waiting_reqs.push_back(request.clone());

        if let Some(checkpoint_running) = self.checkpoint_running.as_mut() {
            request.processed_tokens = request.prompt_len;
            checkpoint_running.insert(request.request_id, request);
        }
        // if let Some(checkpoint_waiting) = self.checkpoint_waiting.as_mut() {
        //     checkpoint_waiting.push_back(request);
        // }
    }

    fn finish_request(&mut self, request_id: u64) {
        self.commit_request.remove(&request_id);
        self.kvcache_manager.free_request(request_id);
    }

    fn redo(&mut self) {
        // let start_time = std::time::Instant::now();
        // self.kvcache_manager.redo();
        // self.waiting_reqs = self.checkpoint_waiting.take().unwrap();
        // self.running_reqs = self.checkpoint_running.take().unwrap();
        self.checkpoint_waiting = None;
        self.checkpoint_running = None;
        // let duration = start_time.elapsed();
        // tracing::debug!("Simulator redo state took {:.2?} ms", duration.as_millis());
        // let mut operations = self.operation_stack.take().unwrap_or_default();
        // while let Some(op) = operations.pop() {
        //     match op {
        //         SchedulerOperation::Schedule(batch) => {
        //             self.redo_batch(&batch);
        //         }
        //     }
        // }
        // self.operation_stack = Some(Vec::new());
    }

    fn checkpoint(&mut self) {
        let start_time = std::time::Instant::now();
        self.checkpoint_waiting = Some(self.waiting_reqs.clone());
        self.checkpoint_running = Some(self.running_reqs.clone());
        let duration = start_time.elapsed();
        tracing::debug!("Simulator checkpointing state took {:.2?} µs", duration.as_micros());
    }

    pub fn sync(&mut self, event: &VllmMetric) -> BatchForPredictor {
        // need to reconsider the logic of preemption

        // evict kvcache block
        //
        // 1. update request states: computed_tokens += 1, waiting -> running, collect finished request and free at last
        // 2. construct request -> kvcache block mapping
        // 3. kvcache -> block id
        // 4. free finished requests and evicted blocks
        // 5. construct batch for predictor training
        let start_time = std::time::Instant::now();
        let mut num_prefill_tokens: Vec<usize> = Vec::new();
        let mut prefill_request_ids: Vec<u64> = Vec::new();

        let mut num_prefill_computed_tokens: Vec<usize> = Vec::new();
        let mut num_decode_computed_tokens: Vec<usize> = Vec::new();

        let mut output_ids: HashSet<u64> = HashSet::new();
        let mut finished_ids: HashSet<u64> = HashSet::new();

        let mut full_prefill_token_cnt = 0;
        for request_id in event.preempted_ids.iter() {
            self.kvcache_manager.free_request_block(*request_id);
            let req = self.running_reqs.remove(request_id).unwrap();
            self.waiting_reqs.push_front(req);
        }

        for (i, evict_id) in event.evicted_block_ids.iter().enumerate() {
            // if !self.kvcache_manager.is_block_free(*evict_id as u32) {
            //     panic!("Evicting a block that is not free: {}", evict_id);
            // }
            self.kvcache_manager.evict_block_hash(*evict_id as u32);
            // debug_assert!(self.kvcache_manager.is_block_free(*evict_id as u32));
            // need more assertion, evict_id == block_ud in freelist according index
        }

        // construct batch for predictor
        for output in &event.outputs {
            let rid = output.request_id;
            output_ids.insert(rid);
            if output.state == "PREFILL" {
                let req = match self.running_reqs.get_mut(&output.request_id) {
                    Some(r) => r,
                    None => {
                        let req = self.remove_from_waiting(rid);
                        self.running_reqs.insert(rid, req);
                        self.running_reqs.get_mut(&rid).unwrap()
                    }
                };
                let processed_tokens_cur_step =
                    (req.prompt_len - output.prev_computed_tokens) as usize;
                num_prefill_tokens.push(processed_tokens_cur_step);
                num_prefill_computed_tokens.push(output.prev_computed_tokens as usize);
                req.processed_tokens = req.prompt_len;
                full_prefill_token_cnt += processed_tokens_cur_step;
                prefill_request_ids.push(output.request_id);
            } else {
                let req = self.running_reqs.get_mut(&rid).unwrap();
                num_decode_computed_tokens.push(req.processed_tokens as usize);
                req.processed_tokens += 1 as u32;
            }

            if output.is_finished {
                finished_ids.insert(output.request_id);
            }
        }

        // Set KVCache block ids for each request(Only when new blocks are allocated, vllm return all block ids the request used)
        for (rid, bids) in event.cur_used_block_ids.iter() {
            self.kvcache_manager.set_req_block_ids(rid.clone(), bids);

            // chunked prefill request, we need to account for the tokens already processed
            let processed_tokens = event.prefill_tokens - full_prefill_token_cnt;
            if !output_ids.contains(rid) {
                // tracing::debug!(
                //     "Handling chunked prefill request: {}, processed_tokens: {}",
                //     rid,
                //     processed_tokens
                // );

                // Move the request from waiting to running
                let computed_tokens = match self.running_reqs.get_mut(rid) {
                    Some(r) => {
                        let computed_tokens = r.processed_tokens;
                        r.processed_tokens += processed_tokens as u32;
                        computed_tokens
                    }
                    None => {
                        let mut r = self.remove_from_waiting(*rid);
                        // first time seeing this request, may prefix cache hit, so we need to take hitted blocks into account
                        let num_allocated_block_cur_step =
                            (processed_tokens + self.config.scheduler_config.block_size - 1)
                                / self.config.scheduler_config.block_size;
                        let num_hit_blocks = bids.len() - num_allocated_block_cur_step;
                        let num_hit_tokens =
                            (num_hit_blocks * self.config.scheduler_config.block_size) as u32;
                        r.processed_tokens += num_hit_tokens + processed_tokens as u32;

                        // num_prefill_computed_tokens.push(
                        //     (num_hit_blocks * self.config.scheduler_config.block_size) as usize,
                        // );
                        // r.processed_tokens +=
                        //     (num_hit_blocks * self.config.scheduler_config.block_size) as u32;

                        // now reinsert into running_reqs
                        self.running_reqs.insert(*rid, r);
                        num_hit_tokens
                    }
                };
                num_prefill_tokens.push(processed_tokens);
                num_prefill_computed_tokens.push(computed_tokens as usize);
                prefill_request_ids.push(*rid);
            }
        }

        // update request hashes
        for output in &event.outputs {
            if let Some(hash) = output.new_block_hash {
                self.kvcache_manager.req_update_hash(output.request_id, hash);
            }
        }

        for request_id in finished_ids {
            // free all request that finished in this step
            self.finish_request(request_id);
            self.running_reqs.remove(&request_id);
        }

        let num_scheduled_tokens = (self.config.scheduler_config.token_budget as usize
            - event.prefill_token_budget)
            + event.prefill_tokens;

        let duration = start_time.elapsed();
        tracing::debug!("Simulator sync processing output took {:.2?} ms", duration.as_millis());
        self.num_decode_computed_tokens = num_decode_computed_tokens;
        self.prefill_request_ids = prefill_request_ids;
        BatchForPredictor {
            num_tokens: num_scheduled_tokens,
            num_prefill_tokens: num_prefill_tokens,
            num_prefill_computed_tokens: num_prefill_computed_tokens,
            num_decode_computed_tokens: vec![],
            num_tokens_rounded: num_scheduled_tokens + 7 / 8 * 8,
            size: event.outputs.len(),
            doing_prefill_req: Vec::new(),
            prefill_done_request: Vec::new(),
        }
    }

    /// Remove a request from waiting_reqs by request_id
    fn remove_from_waiting(&mut self, request_id: u64) -> Request {
        // as for efficiency, we first check the front of the queue
        // if so, we can directly pop it from VecDeque(Common case)
        if self.waiting_reqs.front().as_ref().unwrap().request_id == request_id {
            return self.waiting_reqs.pop_front().unwrap();
        }

        // otherwise, we search through the queue and remove it
        if let Some(pos) = self.waiting_reqs.iter().position(|req| req.request_id == request_id) {
            return self.waiting_reqs.remove(pos).unwrap();
        }
        panic!("Request not found");
    }

    pub fn schedule(
        &mut self,
        // prediction_bucket: &mut VecDeque<(BatchForPredictor, f32)>,
        request: &mut Option<Request>,
        predict_state: PredictState,
        instance_id: usize,
    ) -> Result<(BatchForPredictor, ScheduleCacheOperation), ScheduleError> {
        // Implement scheduling logic here
        let schedule_start_time = SystemTime::now();
        let mut token_budget = self.config.scheduler_config.token_budget;
        let mut operation = ScheduleCacheOperation::Cache;
        debug_assert!(self.checkpoint_running.is_some());
        debug_assert!(self.checkpoint_waiting.is_some());
        let mut curr_batch = match predict_state {
            PredictState::Tempt => {
                panic!("Should not be Tempt here")
            }
            PredictState::InitialPredict(prev_batch) => {
                tracing::debug!("Instance: {}, Scheduling in InitialPredict state", instance_id);
                let mut curr_batch = prev_batch;
                let mut prev_prefill = std::mem::take(&mut curr_batch.prefill_done_request);
                prev_prefill.append(&mut curr_batch.doing_prefill_req);
                curr_batch.num_prefill_tokens.clear();
                curr_batch.num_prefill_computed_tokens.clear();

                token_budget -= curr_batch.curr_num_tokens() as u32;
                for rid in prev_prefill.iter() {
                    if let Some(req) = self.checkpoint_running.as_mut().unwrap().get_mut(rid) {
                        tracing::debug!(
                            "Instance: {}, Scheduling request from InitialPredict: {:?}",
                            instance_id,
                            req,
                        );
                        let mut num_new_tokens = req.all_token_lens() - req.processed_tokens;
                        num_new_tokens = num_new_tokens.min(token_budget as u32);

                        token_budget -= num_new_tokens;
                        if req.processed_tokens >= req.prompt_len {
                            curr_batch
                                .add_decode(num_new_tokens as usize, req.processed_tokens as usize);
                        } else {
                            curr_batch.add_prefill(
                                num_new_tokens as usize,
                                req.processed_tokens as usize,
                                req.request_id,
                            );
                            if req.processed_tokens + num_new_tokens == req.prompt_len {
                                curr_batch.prefill_done_request.push(req.request_id);
                            } else {
                                curr_batch.doing_prefill_req.push(req.request_id);
                            }
                        }
                        match Self::compute_new_block(num_new_tokens as usize, &req, &self.config) {
                            0 => {}
                            1 => {}
                            // !!!!allocate
                            new_block_num => {
                                let allocate_res = self
                                    .kvcache_manager
                                    .allocate_block(req.request_id, new_block_num as usize);
                                if !allocate_res {
                                    return Err(ScheduleError::KVCacheAllocationFailed);
                                }
                            }
                        }

                        req.processed_tokens += num_new_tokens;
                    } else {
                        let checkpoint_waiting = self.checkpoint_waiting.as_mut().unwrap();
                        let mut req = if let Some(pos) =
                            checkpoint_waiting.iter().position(|req| req.request_id == *rid)
                        {
                            checkpoint_waiting.remove(pos).unwrap()
                        } else {
                            // silent bugs, Todo in the future
                            tracing::error!(
                                "Instance: {} request not found in running_reqs, searching waiting_reqs, rid: {}, checkpoint_waiting: {:?}, checkpoint_running: {:?}",
                                instance_id, rid, checkpoint_waiting, self.checkpoint_running
                            );
                            continue;
                        };

                        let mut num_new_tokens = req.all_token_lens() - req.processed_tokens;
                        num_new_tokens = num_new_tokens.min(token_budget as u32);

                        token_budget -= num_new_tokens;

                        curr_batch.add_prefill(
                            num_new_tokens as usize,
                            req.processed_tokens as usize,
                            req.request_id,
                        );
                        if req.processed_tokens + num_new_tokens == req.prompt_len {
                            curr_batch.prefill_done_request.push(req.request_id);
                        } else {
                            curr_batch.doing_prefill_req.push(req.request_id);
                        }

                        match Self::compute_new_block(num_new_tokens as usize, &req, &self.config) {
                            0 => {}
                            1 => {}
                            // !!!!allocate
                            new_block_num => {
                                let allocate_res = self
                                    .kvcache_manager
                                    .allocate_block(req.request_id, new_block_num as usize);
                                if !allocate_res {
                                    return Err(ScheduleError::KVCacheAllocationFailed);
                                }
                            }
                        }

                        req.processed_tokens += num_new_tokens;
                    }
                }

                curr_batch
            }
            // Predict state means we are scheduling for prefill of predicted request
            // don't change the state of kvcache_manager and request_queue
            // because the request don't actually scheduled to this replica, it shouldn't affect the simulated state
            // so we [only construct the batch_for_predictor], and scheduled predicted request specificily, [don't touch the running_queue]
            // when control flow goes to waiting_queue, there should be no requests, so [skip waiting_queue]
            // because the Predict state only happens when the predicted request scheduled in previous step
            // which meann all previous waiting request had been finished prefill
            PredictState::Predict(uncommit_prediction) => {
                // when run into predict state from waiting_queue, the last prefill request is the predicted request
                // so we don't need to construct prefill part

                tracing::debug!("Instance: {}, Scheduling in Predict state", instance_id);
                let (cached_last_batch, _) = uncommit_prediction.back().as_ref().unwrap();
                operation = ScheduleCacheOperation::NotCache;
                let mut curr_batch = BatchForPredictor {
                    num_tokens: 0,
                    num_prefill_tokens: Vec::new(),
                    num_prefill_computed_tokens: Vec::new(),
                    num_decode_computed_tokens: cached_last_batch
                        .num_decode_computed_tokens
                        .clone(),
                    num_tokens_rounded: 0,
                    size: 0,
                    doing_prefill_req: Vec::new(),
                    prefill_done_request: Vec::new(),
                };

                // add decode part from last batch except the last prefill chunk(predicted request)
                let len = cached_last_batch.num_prefill_computed_tokens.len();
                if len != 1 {
                    if len == 0 {
                        panic!(
                            "Instance: {}, Cached last batch has no prefill tokens",
                            instance_id
                        );
                    }
                    cached_last_batch
                        .num_prefill_computed_tokens
                        .iter()
                        .zip(cached_last_batch.num_prefill_tokens.iter())
                        .enumerate()
                        .for_each(|(i, (computed_token, chunk_size))| {
                            if i + 1 < len {
                                curr_batch.add_decode(1, *computed_token + *chunk_size);
                            }
                        });
                }

                token_budget -= curr_batch.curr_num_tokens() as u32;
                curr_batch
            }

            // Simulate state means we are scheduling for request that actually running on this replica
            // so we need to update the state of kvcache_manager and request_queue
            // but for speed up, we can leverage the previous batch_for_predictor to avoid recomputing the whole batch from running_queue
            // when control flow goes to waiting_queue, we need to schedule requests from waiting_queue as usual.
            // But be careful that when the predicted request is scheduled in this step, transform state Simulate => Predict
            // when there is no previous batch, we need to construct the batch from scratch(running_queue -> waiting_queue)
            PredictState::Simulate(prediction_bucket) => {
                let mut curr_batch = if let Some(prev_batch) = prediction_bucket.back().as_ref() {
                    // fast path, prediction cache hit, don't scan running_queue
                    let prev_batch = prev_batch.0.as_ref();

                    // last batch in previous simulate, prefill done request not in running q, means it don't affect the state
                    if !prev_batch.doing_prefill_req.is_empty()
                        && !self
                            .commit_request
                            .contains(prev_batch.doing_prefill_req.last().unwrap())
                        || !prev_batch.prefill_done_request.is_empty()
                            && !self
                                .commit_request
                                .contains(prev_batch.prefill_done_request.last().unwrap())
                    {
                        // construct curr_batch based on prev_batch, skip the last prefill done request because it is not in system
                        let prefill_done_request = if !prev_batch.doing_prefill_req.is_empty()
                            && !self
                                .commit_request
                                .contains(prev_batch.doing_prefill_req.last().unwrap())
                        {
                            prev_batch.prefill_done_request.clone()
                        } else {
                            prev_batch.prefill_done_request
                                [..prev_batch.prefill_done_request.len() - 1]
                                .to_vec()
                        };
                        let mut curr_batch = BatchForPredictor {
                            num_tokens: 0,
                            num_prefill_tokens: prev_batch.num_prefill_tokens
                                [..prev_batch.num_prefill_tokens.len() - 1]
                                .to_vec(),
                            num_prefill_computed_tokens: prev_batch.num_prefill_computed_tokens
                                [..prev_batch.num_prefill_computed_tokens.len() - 1]
                                .to_vec(),
                            num_decode_computed_tokens: prev_batch
                                .num_decode_computed_tokens
                                .iter()
                                .map(|&x| x + 1)
                                .collect(),
                            num_tokens_rounded: 0,
                            size: 0,
                            doing_prefill_req: Vec::new(),
                            prefill_done_request: prefill_done_request,
                        };
                        let len = prev_batch.num_prefill_computed_tokens.len() - 1;
                        prev_batch
                            .num_prefill_computed_tokens
                            .iter()
                            .zip(prev_batch.num_prefill_tokens.iter())
                            .enumerate()
                            .for_each(|(i, (&computed_token, &chunk_size))| {
                                if i + 1 < len {
                                    curr_batch.add_decode(1, computed_token + chunk_size);
                                }
                            });

                        tracing::debug!(
                            "Instance: {}, previous prediction batch contains uncommit state, adjusting batch accordingly, batch: {:?}, previous_batch: {:?}",
                            instance_id,
                            curr_batch,
                            prev_batch
                        );
                        token_budget -= curr_batch.curr_num_tokens() as u32;
                        operation = ScheduleCacheOperation::Replace;
                        curr_batch
                    } else {
                        tracing::debug!(
                            "Instance: {}, previous prediction batch not contains uncommit state",
                            instance_id,
                        );
                        let mut curr_batch = BatchForPredictor {
                            num_tokens: 0,
                            num_prefill_tokens: Vec::new(),
                            num_prefill_computed_tokens: Vec::new(),
                            num_decode_computed_tokens: prev_batch
                                .num_decode_computed_tokens
                                .clone(),
                            num_tokens_rounded: 0,
                            size: 0,
                            doing_prefill_req: Vec::new(),
                            prefill_done_request: Vec::new(),
                        };
                        token_budget -= curr_batch.curr_num_tokens() as u32;
                        // now schedule the chunk prefill requests in doing_prefill_req
                        for rid in prev_batch.doing_prefill_req.iter() {
                            if let Some(req) =
                                self.checkpoint_running.as_mut().unwrap().get_mut(rid)
                            {
                                let mut num_new_tokens =
                                    req.all_token_lens() - req.processed_tokens;
                                num_new_tokens = num_new_tokens.min(token_budget as u32);

                                token_budget -= num_new_tokens;
                                if req.processed_tokens >= req.prompt_len {
                                    curr_batch.add_decode(
                                        num_new_tokens as usize,
                                        req.processed_tokens as usize,
                                    );
                                } else {
                                    curr_batch.add_prefill(
                                        num_new_tokens as usize,
                                        req.processed_tokens as usize,
                                        req.request_id,
                                    );
                                    if req.processed_tokens + num_new_tokens == req.prompt_len {
                                        curr_batch.prefill_done_request.push(req.request_id);
                                    } else {
                                        curr_batch.doing_prefill_req.push(req.request_id);
                                    }
                                }
                                match Self::compute_new_block(
                                    num_new_tokens as usize,
                                    &req,
                                    &self.config,
                                ) {
                                    0 => {}
                                    1 => {}
                                    // !!!!allocate
                                    new_block_num => {
                                        let allocate_res = self
                                            .kvcache_manager
                                            .allocate_block(req.request_id, new_block_num as usize);
                                        if !allocate_res {
                                            return Err(ScheduleError::KVCacheAllocationFailed);
                                        }
                                    }
                                }

                                req.processed_tokens += num_new_tokens;
                            } else {
                                panic!(
                                    "Instance: {} request not found in running_reqs, rid: {}",
                                    instance_id, rid
                                );
                            }
                        }
                        curr_batch
                    }
                } else {
                    // slow path, need to construct batch from running_queue
                    tracing::debug!("Instance: {}, No previous prediction batch, constructing from running_queue", instance_id);
                    let mut curr_batch = BatchForPredictor {
                        num_tokens: 0,
                        num_prefill_tokens: Vec::new(),
                        num_prefill_computed_tokens: Vec::new(),
                        num_decode_computed_tokens: self.num_decode_computed_tokens.clone(),
                        num_tokens_rounded: 0,
                        size: 0,
                        doing_prefill_req: vec![],
                        prefill_done_request: vec![],
                    };
                    token_budget -= curr_batch.curr_num_tokens() as u32;
                    for rid in self.prefill_request_ids.iter() {
                        if let Some(req) = self.checkpoint_running.as_mut().unwrap().get_mut(rid) {
                            let mut num_new_tokens = req.all_token_lens() - req.processed_tokens;
                            num_new_tokens = num_new_tokens.min(token_budget as u32);

                            token_budget -= num_new_tokens;
                            if req.processed_tokens >= req.prompt_len {
                                curr_batch.add_decode(
                                    num_new_tokens as usize,
                                    req.processed_tokens as usize,
                                );
                            } else {
                                curr_batch.add_prefill(
                                    num_new_tokens as usize,
                                    req.processed_tokens as usize,
                                    req.request_id,
                                );
                                if req.processed_tokens + num_new_tokens == req.prompt_len {
                                    curr_batch.prefill_done_request.push(req.request_id);
                                } else {
                                    curr_batch.doing_prefill_req.push(req.request_id);
                                }
                            }
                            match Self::compute_new_block(
                                num_new_tokens as usize,
                                &req,
                                &self.config,
                            ) {
                                0 => {}
                                1 => {}
                                // !!!!allocate
                                new_block_num => {
                                    let allocate_res = self
                                        .kvcache_manager
                                        .allocate_block(req.request_id, new_block_num as usize);
                                    if !allocate_res {
                                        return Err(ScheduleError::KVCacheAllocationFailed);
                                    }
                                }
                            }

                            req.processed_tokens += num_new_tokens;
                        } else {
                            panic!(
                                "Instance: {} request not found in running_reqs, rid: {}",
                                instance_id, rid
                            );
                        }
                    }
                    curr_batch
                };
                if let Some(checkpoint_waiting) = self.checkpoint_waiting.as_mut() {
                    while !checkpoint_waiting.is_empty() && token_budget > 0 {
                        let mut req = checkpoint_waiting.pop_front().unwrap();
                        let num_local_computed_tokens =
                            self.kvcache_manager.get_computed_blocks(&req);

                        tracing::debug!(
                            "Instance: {}, waiting request_info: {:?}",
                            instance_id,
                            req,
                        );
                        let mut hit_cnt = 0;
                        if let Some(local_blocks) = num_local_computed_tokens {
                            hit_cnt =
                                (local_blocks * self.config.scheduler_config.block_size) as u32;
                            req.processed_tokens += hit_cnt;
                            // self.kvcache_manager.cache_hit_block(local_blocks);
                        }
                        tracing::debug!(
                            "Instance: {}, Scheduling request from waiting_queue: {}, hit_cnt: {}, processed_tokens: {}",
                            instance_id,
                            req.request_id,
                            hit_cnt,
                            req.processed_tokens
                        );
                        let mut num_new_tokens = req.prompt_len - hit_cnt;
                        num_new_tokens = num_new_tokens.min(token_budget);
                        assert!(num_new_tokens > 0);

                        match Self::compute_new_block(num_new_tokens as usize, &req, &self.config) {
                            0 => {}
                            // !!!!allocate
                            new_block_num => {
                                let allocate_res = self
                                    .kvcache_manager
                                    .allocate_block(req.request_id, new_block_num as usize);
                                if !allocate_res {
                                    return Err(ScheduleError::KVCacheAllocationFailed);
                                }
                            }
                        }
                        curr_batch.add_prefill(
                            num_new_tokens as usize,
                            req.processed_tokens as usize,
                            req.request_id,
                        );

                        req.processed_tokens += num_new_tokens;
                        if req.processed_tokens == req.prompt_len {
                            curr_batch.prefill_done_request.push(req.request_id);
                        } else {
                            curr_batch.doing_prefill_req.push(req.request_id);
                        }

                        token_budget -= num_new_tokens;
                        self.checkpoint_running.as_mut().unwrap().insert(req.request_id, req);
                    }
                }
                curr_batch
            }
        };
        let schedule_previous_duration = schedule_start_time.elapsed().unwrap();

        tracing::debug!(
            "Instance: {}, curr_batch before predicted request: {:?}, token_budget: {}, new_request: {:?}, duration: {:.2?} micro second",
            instance_id,
            curr_batch,
            token_budget,
            request,
            schedule_previous_duration.as_micros()
        );
        if let Some(request) = request.as_mut() {
            if token_budget > 0 && request.prompt_len > request.processed_tokens {
                tracing::debug!(
                    "Instance: {}, Scheduling predicted request: {}, processed_tokens: {}, prompt_len: {}",
                    instance_id,
                    request.request_id,
                    request.processed_tokens,
                    request.prompt_len
                );
                if request.processed_tokens == 0 {
                    request.processed_tokens = self.kvcache_manager.check_matched_kvcache(request)
                        * (self.config.scheduler_config.block_size as u32);
                }

                let num_new_tokens =
                    token_budget.min(request.prompt_len - request.processed_tokens);
                curr_batch.add_prefill(
                    num_new_tokens as usize,
                    request.processed_tokens as usize,
                    request.request_id,
                );
                request.processed_tokens += num_new_tokens;

                if request.processed_tokens == request.prompt_len {
                    curr_batch.prefill_done_request.push(request.request_id);
                } else {
                    curr_batch.doing_prefill_req.push(request.request_id);
                }

                // when curr batch only has this predicted request to prefill, means it is not commit state
                if curr_batch.prefill_done_request.len() + curr_batch.doing_prefill_req.len() == 1 {
                    debug_assert!(operation != ScheduleCacheOperation::Replace);
                    operation = ScheduleCacheOperation::NotCache;
                }
            }
        }

        curr_batch.calculate_batch_metric();
        debug_assert!(
            curr_batch.curr_num_tokens() as u32 <= self.config.scheduler_config.token_budget,
            "Instance: {}, curr_batch: {:?}, token_budget: {}",
            instance_id,
            curr_batch,
            self.config.scheduler_config.token_budget
        );
        debug_assert!(
            curr_batch.prefill_done_request.len() + curr_batch.doing_prefill_req.len()
                == curr_batch.num_prefill_tokens.len(),
            "Instance: {}, curr_batch: {:?}",
            instance_id,
            curr_batch
        );
        debug_assert!(
            curr_batch.num_decode_computed_tokens.len() >= self.num_decode_computed_tokens.len(),
            "Instance: {}, curr_batch: {:?}, previous_decode_len: {}",
            instance_id,
            curr_batch,
            self.num_decode_computed_tokens.len()
        );
        debug_assert!(
            curr_batch.doing_prefill_req.len() <= 1,
            "Instance: {}, curr_batch: {:?}",
            instance_id,
            curr_batch
        );
        debug_assert!(
            if request.is_some() { curr_batch.num_prefill_tokens.len() > 0 } else { true },
            "Instance: {}, curr_batch: {:?}",
            instance_id,
            curr_batch
        );
        debug_assert!(
            curr_batch.num_tokens != 0,
            "Instance: {}, curr_batch: {:?}",
            instance_id,
            curr_batch
        );
        Ok((curr_batch, operation))
    }

    fn compute_new_block(num_new_token: usize, req: &Request, config: &SimulationConfig) -> usize {
        let block_size = config.scheduler_config.block_size;
        let origin_blocks = (req.processed_tokens as usize + block_size - 1) / block_size;
        let next_blocks =
            (req.processed_tokens as usize + num_new_token + block_size - 1) / block_size;
        next_blocks - origin_blocks
    }
}
