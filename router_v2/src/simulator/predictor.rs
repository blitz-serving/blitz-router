use crate::simulator::batch::BatchForPredictor;
use crate::simulator::config::SimulationConfig;
use crate::simulator::execution_time::{ExecutionTime, LlamaExecutionTime};
use clap::ValueEnum;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

// Refine the base predictor with trained model
pub trait TrainedPredictor: Send + Sync {
    fn get_execution_time(&self, batch: &BatchForPredictor) -> f32;
    fn update_params(&mut self, train_data: (&BatchForPredictor, f32));
}

#[derive(Clone, ValueEnum, Debug, Default, PartialEq, Eq)]
pub enum TrainedPredictorType {
    // learning_rate
    #[default]
    LinearRegression,

    LatencyWiseLinearRegression,

    // Accuracy of Simulation to decode and prefill are quite different
    // So they may need different weights
    // Not implemented yet
    StagewiseLinearRegression,
}

type Weight = (f32, f32);

pub struct StagewiseLinearRegressionPredictor {}

pub struct LatencyWiseLinearRegressionPredictor {
    weight: (Weight, Weight),
    predictor: Box<dyn Predictor>,
    learning_rate: f32,
    threshold: f32,
    warmup_time: usize,
}

impl LatencyWiseLinearRegressionPredictor {
    pub fn new(predictor: Box<dyn Predictor>, config: Arc<SimulationConfig>) -> Self {
        let weight = ((1.0, 0.0), (1.0, 0.0));
        let learning_rate = match config.predictor_config.trained_type {
            TrainedPredictorType::LinearRegression => {
                config.predictor_config.learning_rate.unwrap()
            }
            _ => panic!("Unsupported trained predictor type"),
        };
        LatencyWiseLinearRegressionPredictor {
            weight,
            predictor,
            learning_rate,
            threshold: config.predictor_config.refine_threshold,
            warmup_time: 0,
        }
    }
    fn apply_weight(base_time: f32, weight: &Weight) -> f32 {
        weight.0 * base_time + weight.1
    }

    fn adjust_weight(weight: &mut Weight, error: f32, predicted_res: f32, learning_rate: f32) {
        // Gradient descent to update weight
        let grad_w0 = error * predicted_res;
        let grad_w1 = error;

        weight.0 += learning_rate * grad_w0;
        weight.1 += learning_rate * grad_w1;
    }
}

impl TrainedPredictor for LatencyWiseLinearRegressionPredictor {
    fn get_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        let mut base_time = self.predictor.get_execution_time(batch).get_total_execution_time();
        // second -> milli second
        base_time *= 1000.0;
        base_time
    }

    fn update_params(&mut self, train_data: (&BatchForPredictor, f32)) {
        // Online gradient descent to update weight, so we can fine-tune the predictor based on the machine

        let predicted_res = self.get_execution_time(train_data.0);

        // Decide which weight to use based on the threshold(but now doesn't work well, may seperate decode and prefill later)
        let (trained_res, weight) = if predicted_res < self.threshold {
            (Self::apply_weight(predicted_res, &self.weight.0), &mut self.weight.0)
        } else {
            (Self::apply_weight(predicted_res, &self.weight.1), &mut self.weight.1)
        };
        let real_res = train_data.1;
        let error = real_res - trained_res;

        // Warmup phase, vllm latency variance is high at the beginning
        self.warmup_time += 1;
        if self.warmup_time >= 10 {
            Self::adjust_weight(weight, error, predicted_res, self.learning_rate);
        }
        tracing::debug!(
            "Updating LinearRegressionPredictor: predicted = {}, real = {}, error = {}, curr_weight = ({:?}, {:?})",
            trained_res,
            real_res,
            error / real_res,
            self.weight.0,
            self.weight.1
        );
    }
}

pub struct LinearRegressionPredictor {
    weight: Weight,
    predictor: Arc<Box<dyn Predictor>>,
    learning_rate: f32,
    threshold: f32,
    warmup_time: usize,
    config: Arc<SimulationConfig>,
}

impl LinearRegressionPredictor {
    pub fn new(predictor: Arc<Box<dyn Predictor>>, config: Arc<SimulationConfig>) -> Self {
        let weight = (1.0, 0.0);
        let learning_rate = match config.predictor_config.trained_type {
            TrainedPredictorType::LinearRegression => {
                config.predictor_config.learning_rate.unwrap()
            }
            _ => panic!("Unsupported trained predictor type"),
        };
        LinearRegressionPredictor {
            weight,
            predictor,
            learning_rate,
            threshold: config.predictor_config.refine_threshold,
            warmup_time: 0,
            config,
        }
    }
    fn apply_weight(base_time: f32, weight: &Weight) -> f32 {
        weight.0 * base_time + weight.1
        // base_time
    }

    fn adjust_weight(weight: &mut Weight, error: f32, predicted_res: f32, learning_rate: f32) {
        // Gradient descent to update weight
        if error > 0.5 {
            return;
        }
        let grad_w0 = error * predicted_res;
        let grad_w1 = error;

        weight.0 += learning_rate * grad_w0;
        weight.1 += learning_rate * grad_w1;
    }
}

impl TrainedPredictor for LinearRegressionPredictor {
    fn get_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        let execution_time = self.predictor.get_execution_time(batch);
        // let log_info = execution_time.log_info();
        // tracing::info!("Log info from base predictor: {}", log_info);
        let mut base_time = execution_time.get_total_execution_time();
        // second -> micro second
        base_time *= 1000.0;
        base_time
    }

    fn update_params(&mut self, train_data: (&BatchForPredictor, f32)) {
        // Online gradient descent to update weight, so we can fine-tune the predictor based on the machine

        let predicted_res = self.get_execution_time(train_data.0);

        let (trained_res, weight) = if self.config.online_refine {
            (Self::apply_weight(predicted_res, &self.weight), &mut self.weight)
        } else {
            (predicted_res, &mut self.weight)
        };
        // // Decide which weight to use based on the threshold(but now doesn't work well, may seperate decode and prefill later)
        // let (trained_res, weight) = if predicted_res < self.threshold {
        //     (Self::apply_weight(predicted_res, &self.weight.0), &mut self.weight.0)
        // } else {
        //     (Self::apply_weight(predicted_res, &self.weight.1), &mut self.weight.1)
        // };
        let real_res = train_data.1;
        let error = real_res - trained_res;

        // Warmup phase, vllm latency variance is high at the beginning
        self.warmup_time += 1;
        if self.warmup_time >= 10 {
            Self::adjust_weight(weight, error, predicted_res, self.learning_rate);
        }
        tracing::debug!(
            "Updating LinearRegressionPredictor: predicted = {}, real = {}, error = {}, curr_weight = ({:?}, {:?})",
            trained_res,
            real_res,
            error / real_res,
            self.weight.0,
            self.weight.1
        );
    }
}

pub trait Predictor: Send + Sync {
    fn get_execution_time(&self, batch: &BatchForPredictor) -> Box<dyn ExecutionTime>;
}

const TOKEN_BUDGET: usize = 1024;
pub struct LlamaPredictor {
    predictions: HashMap<String, HashMap<(usize, usize), f32>>,
    config: Arc<SimulationConfig>,
    full_token_cache: HashMap<&'static str, f32>,
}

impl LlamaPredictor {
    pub fn new(config: Arc<SimulationConfig>) -> Self {
        let mut predictor = LlamaPredictor {
            predictions: HashMap::new(),
            config,
            full_token_cache: HashMap::new(),
        };

        predictor.load_attention_layer_model();
        predictor.load_mlp_layer_model();

        predictor
    }

    fn load_attention_layer_model(&mut self) {
        let model_names = vec!["attn_decode", "attn_prefill"];

        for model_name in model_names {
            let prediction_map: HashMap<(usize, usize), f32> = self.load_model_prediction_cache(
                model_name,
                self.config.predictor_config.model_hash.clone(),
                self.config.predictor_config.predict_cache_path_prefix.as_ref(),
            );
            if model_name == "attn_decode" {
                // tracing::info!("model_entries: {:?}", prediction_map);
                let key = (1, 128);
                let val = prediction_map[&key];
                tracing::debug!("attn_decode prediction for key {:?}: {}", key, val);
            }
            self.predictions.insert(model_name.to_string(), prediction_map);
            tracing::debug!("Loaded model {} ", model_name);
        }
        // Implement logic to load attention layer model from cache
    }
    fn load_mlp_layer_model(&mut self) {
        // Implement logic to load MLP layer model from cache
        let mut model_names = vec![
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "attn_rope",
            "attn_kv_cache_save",
            "input_layernorm",
            "post_attention_layernorm",
            "add",
            "sampler_e2e",
            "schedule",
        ];

        let mut full_token_cache_name = vec![
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "attn_rope",
            "attn_kv_cache_save",
            "input_layernorm",
            "post_attention_layernorm",
            "add",
        ];
        if self.config.replica_config.num_pipeline_stages > 1 {
            model_names.push("send_recv");
        }
        if self.config.replica_config.tensor_parallel_size > 1 {
            model_names.push("all_reduce");
        }

        for model_name in model_names {
            let prediction_map = self.load_model_prediction_cache(
                model_name,
                self.config.predictor_config.model_hash.clone(),
                self.config.predictor_config.predict_cache_path_prefix.as_ref(),
            );
            if full_token_cache_name.contains(&model_name) {
                let full_token_time = prediction_map[&(TOKEN_BUDGET, 0)];
                self.full_token_cache.insert(model_name, full_token_time);
            }

            self.predictions.insert(model_name.to_string(), prediction_map);

            tracing::debug!("Loaded model {} ", model_name);
        }
    }

    fn load_model_prediction_cache(
        &self,
        model_name: &str,
        model_hash: String,
        predict_cache_path_prefix: &str,
    ) -> HashMap<(usize, usize), f32> {
        let file_name =
            format!("{predict_cache_path_prefix}/{model_name}_{model_hash}_predictions.csv",);
        tracing::debug!("Loading model prediction cache from {}", file_name);
        let file = File::open(file_name).unwrap();

        let reader = BufReader::new(file);
        let mut prediction_map: HashMap<(usize, usize), f32> = HashMap::new();
        for line in reader.lines().skip(1) {
            let line = line.unwrap();
            let parts: Vec<&str> = line.split(',').collect();
            // tracing::debug!("Loaded prediction line: {:?}", parts);
            let record1 = parts[0].parse::<usize>().unwrap();
            let record2;
            if parts.len() == 2 {
                record2 = 0;
            } else {
                record2 = parts[1].parse::<usize>().unwrap();
            }
            let prediction = parts.last().unwrap().parse::<f32>().unwrap();
            prediction_map.insert((record1, record2), prediction);
        }
        prediction_map
    }

    fn get_attention_rope_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get attention rope execution time from predictions

        if batch.num_tokens_rounded == TOKEN_BUDGET {
            self.full_token_cache["attn_rope"]
        } else {
            self.predictions.get("attn_rope").unwrap()[&(batch.num_tokens_rounded, 0)]
        }
    }

    fn get_attention_kv_cache_save_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get attention kv cache save execution time from predictions

        if batch.num_tokens == TOKEN_BUDGET {
            self.full_token_cache["attn_kv_cache_save"]
        } else {
            self.predictions.get("attn_kv_cache_save").unwrap()[&(batch.num_tokens, 0)]
        }
    }
    fn get_batch_prefill_attention_params(
        &self,
        batch: &BatchForPredictor,
    ) -> Vec<(usize, usize, usize)> {
        let mut prefill_params = Vec::new();
        for (num_tokens, num_computed_tokens) in
            batch.num_prefill_tokens.iter().zip(batch.num_prefill_computed_tokens.iter())
        {
            let prefill_chunk_size = num_tokens.clone();
            let kv_cache_size = (num_computed_tokens
                + self.config.predictor_config.kv_cache_prediction_granularity
                - 1)
                / self.config.predictor_config.kv_cache_prediction_granularity
                * self.config.predictor_config.kv_cache_prediction_granularity;
            prefill_params.push((kv_cache_size, prefill_chunk_size, num_computed_tokens.clone()));
            // tracing::info!(
            //     "Prefill attention params - kv_cache_size: {}, prefill_chunk_size: {}, num_computed_tokens: {}",
            //     kv_cache_size,
            //     prefill_chunk_size,
            //     num_computed_tokens
            // );
        }
        prefill_params
    }

    fn get_attention_prefill_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get attention prefill execution time from predictions
        let prefill_params = self.get_batch_prefill_attention_params(batch);
        if prefill_params.len() == 0 {
            return 0.0;
        }

        let max_predcited_time = prefill_params
            .iter()
            .map(|(kvcache_size, prefill_chunk_size, _)| {
                let compute_operation_count =
                    prefill_chunk_size * ((prefill_chunk_size + 1) / 2 + kvcache_size);
                let computed_operation_count_rounded = (compute_operation_count
                    + self.config.predictor_config.flops_prediction_granularity
                    - 1)
                    / self.config.predictor_config.flops_prediction_granularity
                    * self.config.predictor_config.flops_prediction_granularity;
                self.predictions.get("attn_prefill").unwrap()
                    [&(*kvcache_size, computed_operation_count_rounded)]
            })
            .fold(0. / 0., f32::max);
        if prefill_params.len() > 1 {
            max_predcited_time
                * (1.0 + self.config.model_config.attention_prefill_batching_overhead_fraction)
        } else {
            max_predcited_time
        }
        // tracing::info!(
        //     "agg_kvcache_size: {}, agg_prefill_chunk_size: {}",
        //     agg_kvcache_size,
        //     agg_prefill_chunk_size
        // );

        // let (agg_kvcache_size, mut agg_prefill_chunk_size) = prefill_params.iter().fold(
        //     (0, 0),
        //     |(sum_kv, agg_prefill), (kvcache_size, prefill_chunk_size, num_computed_tokens)| {
        //         (
        //             sum_kv + kvcache_size,
        //             agg_prefill + (num_computed_tokens + prefill_chunk_size) * prefill_chunk_size,
        //         )
        //     },
        // );
        // agg_prefill_chunk_size = (agg_prefill_chunk_size
        //     + self.config.predictor_config.flops_prediction_granularity
        //     - 1)
        //     / self.config.predictor_config.flops_prediction_granularity
        //     * self.config.predictor_config.flops_prediction_granularity;

        // self.predictions.get("attn_prefill").unwrap()[&(agg_kvcache_size, agg_prefill_chunk_size)]
        //     * (1.0 + self.config.model_config.attention_prefill_batching_overhead_fraction)
    }

    fn get_batch_decode_attention_params(&self, batch: &BatchForPredictor) -> (usize, usize) {
        if batch.num_decode_computed_tokens.len() == 0 {
            return (0, 0);
        }
        // let mut decode_max_kvcache_size: usize =
        //     *batch.num_decode_computed_tokens.iter().max().unwrap_or(&0);
        // decode_max_kvcache_size = (decode_max_kvcache_size
        //     + self.config.predictor_config.kv_cache_prediction_granularity
        //     - 1)
        //     / self.config.predictor_config.kv_cache_prediction_granularity
        //     * self.config.predictor_config.kv_cache_prediction_granularity;
        // (batch.num_decode_computed_tokens.len(), decode_max_kvcache_size)

        let decode_agg_kvcache_size: usize = batch.num_decode_computed_tokens.iter().sum();
        let mut decode_avg_kvcache_size =
            decode_agg_kvcache_size / batch.num_decode_computed_tokens.len();
        decode_avg_kvcache_size = (decode_avg_kvcache_size
            + self.config.predictor_config.kv_cache_prediction_granularity
            - 1)
            / self.config.predictor_config.kv_cache_prediction_granularity
            * self.config.predictor_config.kv_cache_prediction_granularity;

        (batch.num_decode_computed_tokens.len(), decode_avg_kvcache_size)
    }

    fn get_attention_decode_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get attention decode execution time from predictions
        let (decode_batch_size, max_kvcache_size) = self.get_batch_decode_attention_params(batch);
        if decode_batch_size == 0 {
            return 0.0;
        }

        // tracing::info!(
        //     "decode_batch_size: {}, avg_kvcache_size: {}",
        //     decode_batch_size,
        //     max_kvcache_size
        // );
        self.predictions.get("attn_decode").unwrap()[&(decode_batch_size, max_kvcache_size)]
            * (1.0
                + self.config.model_config.attention_decode_batching_overhead_fraction
                    * (decode_batch_size > 1) as u32 as f32)
    }

    fn get_schedule_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get schedule time from predictions
        // if self.config.predictor_config.skip_cpu_overhead_modeling {
        //     return 0.0;
        // }
        return self.predictions.get("schedule").unwrap()[&(batch.size, 0)];
        // 3.0
        // return self.predictions.get("schedule").unwrap()[&(batch.size, 0)];
        // 0.5
    }

    fn get_sampler_e2e_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get sampler end-to-end time from predictions
        self.predictions.get("sampler_e2e").unwrap()[&(batch.size, 0)]
        // if batch.num_decode_computed_tokens.len() + batch.num_prefill_computed_tokens.len() <= 78 {
        //     return 17.0;
        // } else if batch.num_decode_computed_tokens.len() + batch.num_prefill_computed_tokens.len()
        //     <= 156
        // {
        //     return 35.0;
        // } else {
        //     return 52.0;
        // }
        // 1.0
    }

    fn get_prepare_inputs_e2e_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get prepare inputs end-to-end time from predictions
        // if self.config.predictor_config.skip_cpu_overhead_modeling {
        //     return 0.0;
        // }
        // return self.predictions.get("prepare_inputs_e2e").unwrap()[&(batch.size, 0)];

        0.0
    }

    fn get_process_model_outputs_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get process model outputs time from predictions
        // if self.config.predictor_config.skip_cpu_overhead_modeling {
        //     return 0.0;
        // }
        // return self.predictions.get("process_model_outputs").unwrap()[&(batch.size, 0)];
        // 6.0
        0.0
    }

    fn get_ray_comm_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get ray communication time from predictions
        if self.config.predictor_config.skip_cpu_overhead_modeling {
            return 0.0;
        }
        return self.predictions.get("ray_comm").unwrap()[&(batch.size, 0)];
    }

    fn get_attention_layer_pre_proj_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get attention layer pre projection execution time from predictions
        if batch.num_tokens_rounded == TOKEN_BUDGET {
            self.full_token_cache["attn_pre_proj"]
        } else {
            self.predictions.get("attn_pre_proj").unwrap()[&(batch.num_tokens_rounded, 0)]
        }
    }

    fn get_attention_layer_post_proj_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get attention layer post projection execution time from predictions
        if batch.num_tokens_rounded == TOKEN_BUDGET {
            self.full_token_cache["attn_post_proj"]
        } else {
            self.predictions.get("attn_post_proj").unwrap()[&(batch.num_tokens_rounded, 0)]
        }
    }

    fn get_mlp_layer_up_proj_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get MLP layer up projection execution time from predictions
        if batch.num_tokens_rounded == TOKEN_BUDGET {
            self.full_token_cache["mlp_up_proj"]
        } else {
            self.predictions.get("mlp_up_proj").unwrap()[&(batch.num_tokens_rounded, 0)] * 1.12
        }
    }

    fn get_mlp_layer_down_proj_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get MLP layer down projection execution time from predictions
        let mut latency = if batch.num_tokens_rounded == TOKEN_BUDGET {
            self.full_token_cache["mlp_down_proj"]
        } else {
            self.predictions.get("mlp_down_proj").unwrap()[&(batch.num_tokens_rounded, 0)]
        };
        if latency > 0.78125 {
            latency -= 0.125;
        }
        latency
    }

    fn get_mlp_layer_act_execution_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get MLP layer activation execution time from predictions
        if batch.num_tokens_rounded == TOKEN_BUDGET {
            self.full_token_cache["mlp_act"]
        } else {
            self.predictions.get("mlp_act").unwrap()[&(batch.num_tokens_rounded, 0)]
        }
    }

    fn get_attn_norm_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get attention normalization time from predictions
        if batch.num_tokens_rounded == TOKEN_BUDGET {
            self.full_token_cache["input_layernorm"]
        } else {
            self.predictions.get("input_layernorm").unwrap()[&(batch.num_tokens_rounded, 0)]
        }
    }

    fn get_mlp_norm_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get MLP normalization time from predictions
        if !self.config.model_config.post_attn_norm {
            return 0.0;
        }
        if batch.num_tokens_rounded == TOKEN_BUDGET {
            self.full_token_cache["post_attention_layernorm"]
        } else {
            self.predictions.get("post_attention_layernorm").unwrap()
                [&(batch.num_tokens_rounded, 0)]
        }
    }

    fn get_add_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get addition time from predictions

        if batch.num_tokens_rounded == TOKEN_BUDGET {
            self.full_token_cache["add"]
        } else {
            self.predictions.get("add").unwrap()[&(batch.num_tokens_rounded, 0)]
        }
    }

    fn get_tensor_parallel_communication_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get tensor parallel communication time from predictions
        self.predictions.get("all_reduce").unwrap()[&(batch.num_tokens_rounded, 0)]
            + self.config.predictor_config.nccl_cpu_launch_overhead_ms.unwrap()
            + self.config.predictor_config.nccl_cpu_skew_overhead_per_device_ms.unwrap()
                * (self.config.replica_config.tensor_parallel_size as f32).powf(1.25)
    }

    fn get_pipeline_parallel_communication_time(&self, batch: &BatchForPredictor) -> f32 {
        // Implement logic to get pipeline parallel communication time from predictions
        self.predictions.get("send_recv").unwrap()[&(batch.num_tokens_rounded, 0)]
    }
}

impl Predictor for LlamaPredictor {
    fn get_execution_time(&self, batch: &BatchForPredictor) -> Box<dyn ExecutionTime> {
        // Implement logic to calculate execution time based on the batch and pipeline stage
        let tensor_parallel_time;
        if self.config.replica_config.tensor_parallel_size == 1 {
            tensor_parallel_time = 0.0;
        } else {
            tensor_parallel_time = self.get_tensor_parallel_communication_time(&batch);
        }

        let pipeline_parallel_time;
        if self.config.replica_config.num_pipeline_stages == 1 {
            pipeline_parallel_time = 0.0;
        } else {
            pipeline_parallel_time = self.get_pipeline_parallel_communication_time(&batch);
        }
        return Box::new(LlamaExecutionTime::new(
            self.config.model_config.num_layers / self.config.replica_config.num_pipeline_stages,
            self.get_attention_rope_execution_time(&batch),
            self.get_attention_kv_cache_save_execution_time(&batch),
            self.get_attention_decode_execution_time(&batch),
            self.get_attention_prefill_execution_time(&batch),
            self.get_attention_layer_pre_proj_execution_time(&batch),
            self.get_attention_layer_post_proj_execution_time(&batch),
            self.get_mlp_layer_up_proj_execution_time(&batch),
            self.get_mlp_layer_down_proj_execution_time(&batch),
            self.get_mlp_layer_act_execution_time(&batch),
            self.get_attn_norm_time(&batch),
            self.get_mlp_norm_time(&batch),
            self.get_add_time(&batch),
            tensor_parallel_time,
            pipeline_parallel_time,
            self.get_schedule_time(&batch),
            self.get_sampler_e2e_time(&batch),
            self.get_prepare_inputs_e2e_time(&batch),
            self.get_process_model_outputs_time(&batch),
            self.get_ray_comm_time(&batch),
        ));
    }
}

mod tests {
    use super::*;
    use crate::simulator::config::*;
    use crate::simulator::predictor::LlamaPredictor;
    use crate::simulator::predictor::Predictor;
    use crate::simulator::predictor::TrainedPredictor;
    use crate::simulator::predictor::TrainedPredictorType;

    // #[test]
    // fn test_llama_predictor() {
    //     let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
    //     let args = SimulationConfig {
    //         replica_config: ReplicaConfig { num_pipeline_stages: 1, tensor_parallel_size: 1 },
    //         scheduler_config: SchedulerConfig {
    //             token_budget: 1024,
    //             block_size: 16,
    //             num_blocks: 35172,
    //         },
    //         model_config: ModelConfig {
    //             num_layers: 32,
    //             num_attention_heads: 32,
    //             post_attn_norm: true,
    //             attention_prefill_batching_overhead_fraction: 0.1,
    //             attention_decode_batching_overhead_fraction: 0.4,
    //         },
    //         predictor_config: PredictorConfig {
    //             model_hash: "d29f0375".to_string(),
    //             predict_cache_path_prefix: "/nvme/zkx/Modified_vidur/cache".to_string(),
    //             trained_type: TrainedPredictorType::LinearRegression,
    //             learning_rate: Some(0.002),
    //             refine_threshold: 50.0,
    //             kv_cache_prediction_granularity: 64,
    //             flops_prediction_granularity: 1024,
    //             nccl_cpu_launch_overhead_ms: None,
    //             nccl_cpu_skew_overhead_per_device_ms: None,
    //             skip_cpu_overhead_modeling: true,
    //         },
    //         device_config: DeviceConfig {},
    //         fake_backend: false,
    //     };

    //     let config = Arc::new(args);
    //     let predictor = LlamaPredictor::new(config.clone());

    //     let batch = BatchForPredictor {
    //         size: 5,
    //         num_tokens: 64 + 128 + 32 + 1,
    //         num_tokens_rounded: (64 + 128 + 32 + 1) + 7 / 8 * 8,
    //         num_prefill_tokens: vec![64, 128, 32, 0],
    //         num_prefill_computed_tokens: vec![0, 64, 0, 0],
    //         num_decode_computed_tokens: vec![100],
    //     };

    //     let execution_time: Box<dyn ExecutionTime> = predictor.get_execution_time(batch);

    //     let llama_execution_time =
    //         execution_time.as_any().downcast_ref::<LlamaExecutionTime>().unwrap();
    //     assert_eq!(llama_execution_time.num_layer_per_pipeline_stage, 32);
    //     assert_eq!((64 + 128) * 128 + 64 * 64 + 32 * 32, 29696);
    //     tracing::info!("execution_time: {}", llama_execution_time.get_total_execution_time());
    //     tracing::info!("execution_time detail: {}", execution_time.log_info());
    // }
}
