pub mod batch;
pub mod config;
mod execution_time;
mod kvcache;
pub mod metrics;
pub mod predictor;
mod state;
use crate::simulator::batch::{BatchForPredictor, Request};
use crate::simulator::config::SimulationConfig;
use crate::simulator::metrics::SystemMetrics;
use crate::simulator::predictor::{LinearRegressionPredictor, Predictor, TrainedPredictor};
use crate::simulator::state::{
    PredictState, ScheduleCacheOperation, ScheduleError, SchedulerState,
};
use crate::{VllmMetric, VllmRequestStatus};
use crossbeam::channel::Receiver as CbReceiver;
use crossbeam::channel::Sender as CbSender;
use pb::generate::v2::Batch;
use std::collections::VecDeque;
use std::f32::INFINITY;
use std::iter::Cloned;
use std::pin::Pin;
use std::sync::Arc;
use std::thread::sleep;
use std::time::SystemTime;
use tokio::spawn;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Receiver, Sender};
// pub struct VllmMetric {
//     pub prefill_tokens: usize,
//     pub prefill_token_budget: usize,
//     pub latency: u64,
//     pub outputs: Vec<VllmRequestStatus>,
//     pub new_block_hashes: Vec<BackendBlockHash>,
//     pub evicted_block_ids: Vec<u64>,
//     pub cur_used_block_ids: IntMap<u64, Vec<u64>>,
//     pub log_info: Option<String>,
// }
const TOKENIZE_TIME: f32 = 20.0;
pub enum SimulatorCommand {
    AddRequest(Request),
    StepSync(VllmMetric),
    // EvictKVCache(Vec<usize>),
    // CreateKVCache((Vec<u32>, Vec<usize>)),
    QuerySim((Request, CbSender<SystemMetrics>)),
}

pub struct Simulator {
    // state: SchedulerState,
    // predictor: Box<dyn Predictor>,
    command_sender: CbSender<SimulatorCommand>,
    config: Arc<SimulationConfig>,
}
// debug: tracing (addRequest + stepsync)
// profile: tracing (querySim)
// redo: tracing (evictKVCache + createKVCache)

impl Simulator {
    pub fn new(
        config: Arc<SimulationConfig>,
        predictor: Arc<Box<dyn Predictor>>,
        instance_id: usize,
    ) -> Self {
        let (command_sender, command_receiver) = crossbeam::channel::unbounded();
        let state = SchedulerState::new(config.clone());
        // let predictor: Box<dyn Predictor> =
        //     Box::new(predictor::LlamaPredictor::new(config.clone()));

        let trained_predictor = Box::new(LinearRegressionPredictor::new(predictor, config.clone()));
        let config_clone = config.clone();
        std::thread::spawn(move || {
            Self::scheduler_loop(
                command_receiver,
                trained_predictor,
                state,
                config_clone,
                instance_id,
            );
        });

        Simulator { command_sender, config }
    }

    pub async fn sync(&self, event: VllmMetric) {
        // tracing::info!("Simulator sync called");
        self.command_sender.send(SimulatorCommand::StepSync(event)).unwrap();
    }

    pub async fn add_request(&self, request: Request) {
        // tracing::info!("Simulator add_request called");
        self.command_sender.send(SimulatorCommand::AddRequest(request)).unwrap();
    }

    // currently not integrated into main loop
    pub fn query_sim(&self, request: Request) -> SystemMetrics {
        // tracing::info!("Simulator query_sim called");
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.command_sender.send(SimulatorCommand::QuerySim((request, tx))).unwrap();
        rx.recv().unwrap()
    }

    fn scheduler_loop(
        mut receiver: CbReceiver<SimulatorCommand>,
        mut predictor: Box<dyn TrainedPredictor>,
        mut state: SchedulerState,
        config: Arc<SimulationConfig>,
        instance_id: usize,
    ) {
        // Implement the main scheduling loop logic here
        let mut commit_prediction: VecDeque<(BatchForPredictor, f32)> = VecDeque::new();
        let mut uncommit_prediction: VecDeque<(BatchForPredictor, f32)> = VecDeque::new();
        while let Ok(command) = receiver.recv() {
            match command {
                SimulatorCommand::AddRequest(request) => {
                    let start = tokio::time::Instant::now();
                    tracing::debug!(
                        "Instance: {}, Simulator received AddRequest, commit_prediction_len: {}, uncommit_prediction_len: {}",
                        instance_id,
                        commit_prediction.len(),
                        uncommit_prediction.len(),
                    );
                    state.add_request(request);
                    if state.replace_bit {
                        debug_assert!(
                            uncommit_prediction.len() != 0,
                            "Instance: {}, replace_bit is true but uncommit_prediction is empty",
                            instance_id
                        );
                        debug_assert!(commit_prediction.len() >= 1, "Instance: {}, replace_bit is true but commit_prediction has less than 1 element", instance_id);
                        commit_prediction.pop_back();
                        state.replace_bit = false;
                    }

                    commit_prediction.append(&mut uncommit_prediction);
                    let duration = start.elapsed();
                    tracing::debug!(
                        "Instance: {}, Simulator AddRequest processed in {:?}",
                        instance_id,
                        duration,
                    );
                }
                SimulatorCommand::StepSync(event) => {
                    tracing::debug!(
                        "Instance: {}, Simulator received StepSync, SSE Event: {:?}",
                        instance_id,
                        event,
                    );
                    let start = tokio::time::Instant::now();
                    let latency = event.latency as f32;
                    let batch_for_predict = state.sync(&event);

                    if commit_prediction.len() != 0 {
                        loop {
                            let prediction = commit_prediction.front().unwrap();
                            // 1. prefill computed tokens should match
                            //    1. prefill computed
                            //    2. chunk size
                            // 2. batch size should match(loose requirement)
                            if state.num_decode_computed_tokens.len()
                                != prediction.0.num_decode_computed_tokens.len()
                            {
                                // the prediction diverges from actual execution, clear the bucket
                                tracing::debug!("Instance: {}, decode batch size not match, prediction: {:?}, actual {:?}", instance_id, prediction, batch_for_predict);
                                commit_prediction.clear();
                                uncommit_prediction.clear();
                                state.replace_bit = false;
                                state.disable_simulation_mode();
                                break;
                            } else if batch_for_predict.num_prefill_tokens.is_empty()
                                && prediction.0.num_prefill_computed_tokens.is_empty()
                            {
                                tracing::debug!(
                                "Instance: {}, both prefill computed tokens are empty, popping front prediction",
                                instance_id
                                );
                                commit_prediction.pop_front().unwrap();
                                break;
                            } else if batch_for_predict.num_prefill_tokens.is_empty()
                                && !prediction.0.num_prefill_computed_tokens.is_empty()
                            {
                                // vllm is still waiting for request tokenize, don't flush prediction bucket
                                tracing::debug!("Instance: {}, vllm still waiting for request tokenize, don't flush prediction bucket", instance_id);
                                break;
                            } else if !batch_for_predict.num_prefill_tokens.is_empty()
                                && prediction.0.num_prefill_tokens.is_empty()
                            {
                                commit_prediction.pop_front().unwrap();
                                continue;
                            } else if &batch_for_predict.num_prefill_computed_tokens
                                != &prediction.0.num_prefill_computed_tokens
                            {
                                // the prediction diverges from actual execution, clear the bucket
                                tracing::debug!("Instance: {}, Simulator StepSync prediction diverged, clearing prediction bucket, prediction: {:?}, actual {:?}", instance_id, prediction, batch_for_predict);
                                commit_prediction.clear();
                                uncommit_prediction.clear();
                                state.replace_bit = false;
                                state.disable_simulation_mode();
                                break;
                            } else {
                                tracing::debug!("Instance: {}, Simulator StepSync prediction matched, popping front prediction", instance_id);
                                commit_prediction.pop_front().unwrap();
                                break;
                            }
                        }
                    } else {
                        commit_prediction.clear();
                        uncommit_prediction.clear();
                        state.flush_existing_state();
                    }

                    let duration = start.elapsed();
                    tracing::debug!(
                        "Instance: {}, Simulator StepSync processed in {:?} μs, commit_prediction_len: {}, uncommit_prediction_len: {}",
                        instance_id,
                        duration.as_micros(),
                        commit_prediction.len(),
                        uncommit_prediction.len()
                    );
                }
                SimulatorCommand::QuerySim((request, response_sender)) => {
                    // uncommit_prediction not empty means the prediction is not committed in add_request
                    if uncommit_prediction.len() > 0 {
                        tracing::debug!(
                            "Instance: {}, Simulator QuerySim means previous prediction not committed, clearing uncommit_prediction",
                            instance_id
                        );
                        state.replace_bit = false;
                        uncommit_prediction.clear();
                    }

                    let system_metrics = Self::predict(
                        &mut state,
                        predictor.as_ref(),
                        Some(request),
                        instance_id,
                        &mut commit_prediction,
                        &mut uncommit_prediction,
                    )
                    .unwrap_or(SystemMetrics { ttft: vec![INFINITY] });

                    response_sender.send(system_metrics).unwrap();
                    // let duration = start_time.elapsed();
                    // tracing::debug("Cloning state took {:?}", duration);
                }
                _ => {
                    panic!("Unknown command")
                }
            };
        }

        tracing::info!("Simulator scheduler loop exited");
    }

    fn predict(
        state: &mut SchedulerState,
        predictor: &dyn TrainedPredictor,
        // request_id: u64,
        mut request: Option<Request>,
        instance_id: usize,
        commit_prediction: &mut VecDeque<(BatchForPredictor, f32)>,
        uncommit_prediction: &mut VecDeque<(BatchForPredictor, f32)>,
    ) -> Result<SystemMetrics, ScheduleError> {
        let mut cur_timestamp = 0.0;
        let mut ttfts = Vec::new();
        let request_id = request.as_ref().unwrap().request_id;
        let start_time = tokio::time::Instant::now();
        let start_system_time = SystemTime::now();

        let mut iter_duration: Vec<(std::time::Duration, std::time::Duration)> = Vec::new();
        // Enable simulation mode (copy-on-write) instead of cloning
        if state.checkpoint_running.is_none() || commit_prediction.len() == 0 {
            state.enable_simulation_mode();
        }

        // now use the cached prediction to fast forward the previous prediction
        let mut skip_last_uncommit_prediction = false;
        for (i, prediction) in commit_prediction.iter().enumerate() {
            // the last one is not the same as current prediction
            // because last one contains part of the previous predicted request, which made the prediction diverge
            // this part will be re-simulated in the main loop
            if i != commit_prediction.len() - 1
                || !prediction.0.doing_prefill_req.is_empty()
                    && state.commit_request.contains(prediction.0.doing_prefill_req.last().unwrap())
                || !prediction.0.prefill_done_request.is_empty()
                    && state
                        .commit_request
                        .contains(prediction.0.prefill_done_request.last().unwrap())
            {
                cur_timestamp += prediction.1;
                prediction.0.prefill_done_request.iter().for_each(|req_id| {
                    let ttft = if let Some(request) =
                        state.checkpoint_running.as_ref().unwrap().get(req_id)
                    {
                        start_system_time
                            .duration_since(*request.arrival_time.as_ref().unwrap())
                            .unwrap()
                            .as_millis() as f32
                            + cur_timestamp
                    } else {
                        cur_timestamp
                    };
                    ttfts.push(ttft);
                });
            } else if i == commit_prediction.len() - 1 {
                skip_last_uncommit_prediction = true;
            }
        }

        let mut scheduler_state = PredictState::Simulate(&commit_prediction);
        let mut predict_request: Option<Request> = None;
        if cur_timestamp < TOKENIZE_TIME && skip_last_uncommit_prediction {
            // now reconstruct the last uncommit prediction and fast forward it
            // need to replace it in commit_prediction
            let prediction = commit_prediction.pop_back().unwrap();
            debug_assert!(prediction.0.doing_prefill_req.len() == 0);
            let mut batch = BatchForPredictor {
                num_tokens: 0,
                num_prefill_tokens: prediction.0.num_prefill_tokens,
                num_prefill_computed_tokens: prediction.0.num_prefill_computed_tokens,
                num_decode_computed_tokens: prediction.0.num_decode_computed_tokens,
                num_tokens_rounded: 0,
                size: 0,
                doing_prefill_req: Vec::new(),
                prefill_done_request: prediction.0.prefill_done_request,
            };
            batch.num_prefill_computed_tokens.pop();
            batch.num_prefill_tokens.pop();
            batch.prefill_done_request.pop();
            batch.calculate_batch_metric();
            tracing::debug!(
                "Instance: {}, reconstructing last uncommit prediction to fast forward, batch: {:?}",
                instance_id,
                batch
            );

            debug_assert!(
                batch.curr_num_tokens() as u32 <= state.config.scheduler_config.token_budget,
                "Instance: {}, batch: {:?}, token_budget: {}",
                instance_id,
                batch,
                state.config.scheduler_config.token_budget
            );
            debug_assert!(
                batch.prefill_done_request.len() + batch.doing_prefill_req.len()
                    == batch.num_prefill_tokens.len(),
                "Instance: {}, batch: {:?}",
                instance_id,
                batch
            );
            debug_assert!(
                batch.num_decode_computed_tokens.len() >= state.num_decode_computed_tokens.len(),
                "Instance: {}, batch: {:?}, previous_decode_len: {}",
                instance_id,
                batch,
                state.num_decode_computed_tokens.len()
            );
            debug_assert!(
                batch.doing_prefill_req.len() <= 1,
                "Instance: {}, batch: {:?}",
                instance_id,
                batch
            );
            // debug_assert!(
            //     if request.is_some() { batch.num_prefill_tokens.len() > 0 } else { true },
            //     "Instance: {}, batch: {:?}",
            //     instance_id,
            //     batch
            // );
            debug_assert!(batch.num_tokens != 0, "Instance: {}, batch: {:?}", instance_id, batch);
            let execution_time = predictor.get_execution_time(&batch);

            batch.prefill_done_request.iter().for_each(|req_id| {
                let request = state.checkpoint_running.as_ref().unwrap().get(req_id).unwrap();
                let ttft = start_system_time
                    .duration_since(*request.arrival_time.as_ref().unwrap())
                    .unwrap()
                    .as_millis() as f32
                    + cur_timestamp;
                ttfts.push(ttft);
            });
            commit_prediction.push_back((batch.clone(), execution_time));
            scheduler_state = PredictState::InitialPredict(batch);
            predict_request = request.take();
            cur_timestamp += execution_time;
        } else if cur_timestamp < TOKENIZE_TIME
            && state.checkpoint_waiting.as_ref().unwrap().is_empty()
        {
            tracing::debug!(
                "Instance: {}, reconstructing initial predict batch to fast forward",
                instance_id,
            );
            let batch = BatchForPredictor {
                num_tokens: 0,
                num_prefill_tokens: Vec::new(),
                num_prefill_computed_tokens: Vec::new(),
                num_decode_computed_tokens: state.num_decode_computed_tokens.clone(),
                num_tokens_rounded: 0,
                size: 0,
                doing_prefill_req: state.prefill_request_ids.clone(),
                prefill_done_request: Vec::new(),
            };
            scheduler_state = PredictState::InitialPredict(batch);
            predict_request = request.take();
            cur_timestamp = TOKENIZE_TIME;
        }

        tracing::debug!(
            "Instance: {}, state: {:?}, predict_request: {:?}",
            instance_id,
            scheduler_state,
            predict_request,
        );
        loop {
            // We don't want to support PP(quite few PP settings in online serving environment)
            // So the logic timer is much simpler
            if cur_timestamp >= TOKENIZE_TIME && predict_request.is_none() {
                predict_request = request.take();
            }

            // start this step
            let iter_start = tokio::time::Instant::now();
            let (batch_for_predictor, cache_operation) =
                state.schedule(&mut predict_request, scheduler_state, instance_id)?;
            let if_stopped = predict_request.is_some()
                && predict_request.as_ref().unwrap().prompt_len
                    == predict_request.as_ref().unwrap().processed_tokens;

            // schedule done
            let schedule_duration = iter_start.elapsed();

            let execution_time = predictor.get_execution_time(&batch_for_predictor);
            tracing::debug!(
                "Instance: {}, batch: {:?}, if_commit: {:?}, latency: {:?}",
                instance_id,
                batch_for_predictor,
                cache_operation,
                execution_time,
            );
            let predict_duration = iter_start.elapsed();

            // execution done
            cur_timestamp += execution_time;

            batch_for_predictor.prefill_done_request.iter().for_each(|req_id| {
                let ttft =
                    if let Some(request) = state.checkpoint_running.as_ref().unwrap().get(req_id) {
                        start_system_time
                            .duration_since(*request.arrival_time.as_ref().unwrap())
                            .unwrap()
                            .as_millis() as f32
                            + cur_timestamp
                    } else {
                        debug_assert!(
                            predict_request.is_some()
                                && predict_request.as_ref().unwrap().request_id == *req_id
                        );
                        cur_timestamp
                    };

                ttfts.push(ttft);
            });
            iter_duration.push((schedule_duration, predict_duration));
            match cache_operation {
                ScheduleCacheOperation::NotCache => {
                    uncommit_prediction.push_back((batch_for_predictor, execution_time));
                    scheduler_state = PredictState::Predict(&uncommit_prediction);
                    tracing::debug!(
                        "Instance: {}, Not cache, push batch to uncommit_prediction, curr_len: {}",
                        instance_id,
                        uncommit_prediction.len(),
                    );
                }
                ScheduleCacheOperation::Cache => {
                    // debug_assert!(matches!(scheduler_state, PredictState::Simulate(_)));
                    if let Some(req_id) = batch_for_predictor.doing_prefill_req.last() {
                        if *req_id == request_id {
                            let mut cloned_batch = BatchForPredictor {
                                num_tokens: 0,
                                num_prefill_tokens: Vec::new(),
                                num_prefill_computed_tokens: Vec::new(),
                                num_decode_computed_tokens: batch_for_predictor
                                    .num_decode_computed_tokens
                                    .clone(),
                                num_tokens_rounded: 0,
                                size: 0,
                                doing_prefill_req: batch_for_predictor.doing_prefill_req.clone(),
                                prefill_done_request: batch_for_predictor
                                    .prefill_done_request
                                    .clone(),
                            };
                            commit_prediction.push_back((batch_for_predictor, execution_time));
                            cloned_batch.doing_prefill_req.pop().unwrap();

                            scheduler_state = PredictState::InitialPredict(cloned_batch);
                        } else {
                            commit_prediction.push_back((batch_for_predictor, execution_time));
                            scheduler_state = PredictState::Simulate(&commit_prediction);
                        }
                    } else {
                        commit_prediction.push_back((batch_for_predictor, execution_time));
                        scheduler_state = PredictState::Simulate(&commit_prediction);
                    }
                    tracing::debug!(
                        "Instance: {}, Cache, push batch to commit_prediction, curr_len: {}",
                        instance_id,
                        commit_prediction.len(),
                    );
                }
                ScheduleCacheOperation::Replace => {
                    // debug_assert!(matches!(scheduler_state, PredictState::Simulate(_)));
                    state.replace_bit = true;
                    debug_assert!(uncommit_prediction.len() == 0);
                    uncommit_prediction.push_back((batch_for_predictor, execution_time));
                    scheduler_state = PredictState::Predict(&uncommit_prediction);
                    tracing::debug!(
                        "Instance: {}, Replace, push batch to uncommit_prediction, curr_len: {}, pop one from commit_prediction, curr_len: {}",
                        instance_id,
                        uncommit_prediction.len(),
                        commit_prediction.len()
                    );
                }
            }

            if if_stopped {
                break;
            }
        }

        let duration = start_time.elapsed();
        tracing::info!(
            "Instance: {}, Simulation loop for request_{} took {:?}, iter_duration: {:?}, commit_prediction_len: {}, uncommit_prediction_len: {}",
            instance_id,
            predict_request.unwrap().request_id,
            duration,
            iter_duration,
            commit_prediction.len(),
            uncommit_prediction.len()
        );
        let system_metrics = SystemMetrics { ttft: ttfts };

        Ok(system_metrics)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {}
}
