use std::any::Any;

pub trait ExecutionTime: Any {
    fn get_total_execution_time(&self) -> f32;
    fn get_cpu_overhead(&self) -> f32;
    fn log_info(&self) -> String;

    fn as_any(&self) -> &dyn Any;
}

pub struct LlamaExecutionTime {
    pub num_layer_per_pipeline_stage: usize,
    pub attention_rope_execution_time: f32,
    pub attention_kv_cache_save_execution_time: f32,
    pub attention_decode_execution_time: f32,
    pub attention_prefill_execution_time: f32,
    pub attention_layer_pre_proj_execution_time: f32,
    pub attention_layer_post_proj_execution_time: f32,
    pub mlp_layer_up_proj_execution_time: f32,
    pub mlp_layer_down_proj_execution_time: f32,
    pub mlp_layer_act_execution_time: f32,
    pub attn_norm_time: f32,
    pub mlp_norm_time: f32,
    pub add_time: f32,
    pub tensor_parallel_communication_time: f32,
    pub pipeline_parallel_communication_time: f32,
    pub schedule_time: f32,
    pub sampler_e2e_time: f32,
    pub prepare_inputs_e2e_time: f32,
    pub process_model_outputs_time: f32,
    pub ray_comm_time: f32,
}

impl LlamaExecutionTime {
    pub fn new(
        num_layer_per_pipeline_stage: usize,
        attention_rope_execution_time: f32,
        attention_kv_cache_save_execution_time: f32,
        attention_decode_execution_time: f32,
        attention_prefill_execution_time: f32,
        attention_layer_pre_proj_execution_time: f32,
        attention_layer_post_proj_execution_time: f32,
        mlp_layer_up_proj_execution_time: f32,
        mlp_layer_down_proj_execution_time: f32,
        mlp_layer_act_execution_time: f32,
        attn_norm_time: f32,
        mlp_norm_time: f32,
        add_time: f32,
        tensor_parallel_communication_time: f32,
        pipeline_parallel_communication_time: f32,
        schedule_time: f32,
        sampler_e2e_time: f32,
        prepare_inputs_e2e_time: f32,
        process_model_outputs_time: f32,
        ray_comm_time: f32,
    ) -> Self {
        LlamaExecutionTime {
            num_layer_per_pipeline_stage,
            attention_rope_execution_time,
            attention_kv_cache_save_execution_time,
            attention_decode_execution_time,
            attention_prefill_execution_time,
            attention_layer_pre_proj_execution_time,
            attention_layer_post_proj_execution_time,
            mlp_layer_up_proj_execution_time,
            mlp_layer_down_proj_execution_time,
            mlp_layer_act_execution_time,
            attn_norm_time,
            mlp_norm_time,
            add_time,
            tensor_parallel_communication_time,
            pipeline_parallel_communication_time,
            schedule_time,
            sampler_e2e_time,
            prepare_inputs_e2e_time,
            process_model_outputs_time,
            ray_comm_time,
        }
    }
}

impl LlamaExecutionTime {
    fn _get_mlp_layer_execution_time(&self) -> f32 {
        self.mlp_layer_up_proj_execution_time
            + self.mlp_layer_down_proj_execution_time
            + self.mlp_layer_act_execution_time
            + self.tensor_parallel_communication_time
            + self.mlp_norm_time
    }
    fn _get_attention_layer_execution_time(&self) -> f32 {
        self.attention_layer_pre_proj_execution_time
            + self.attention_layer_post_proj_execution_time
            + self.attention_rope_execution_time
            + self.attention_kv_cache_save_execution_time
            + self.attention_decode_execution_time
            + self.attention_prefill_execution_time
            + self.tensor_parallel_communication_time
            + self.attn_norm_time
    }

    fn _get_block_execution_time(&self) -> f32 {
        self._get_attention_layer_execution_time()
            + self._get_mlp_layer_execution_time()
            + self.add_time
    }

    fn _model_time_ms(&self) -> f32 {
        self.model_time() * 1e3
    }

    fn model_time(&self) -> f32 {
        // we are not counting the execution time for the embedding layer and last softmax layer
        let block_execution_time = self._get_block_execution_time();
        let pipeline_stage_execution_time =
            block_execution_time * self.num_layer_per_pipeline_stage as f32;
        // return in seconds
        return (pipeline_stage_execution_time + self.pipeline_parallel_communication_time) * 1e-3;
    }
}

impl ExecutionTime for LlamaExecutionTime {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_total_execution_time(&self) -> f32 {
        return self.model_time() + self.get_cpu_overhead() * 1e-3;
    }

    fn get_cpu_overhead(&self) -> f32 {
        self.schedule_time
            + self.sampler_e2e_time
            + self.prepare_inputs_e2e_time
            + self.process_model_outputs_time
            + self.ray_comm_time
    }

    fn log_info(&self) -> String {
        let scale = self.num_layer_per_pipeline_stage as f32;
        let parts = vec![
            format!("total time={:.3}ms", self.get_total_execution_time() * 1e3),
            format!("qkv_proj={:.3}ms", self.attention_layer_pre_proj_execution_time * scale),
            format!("o_proj={:.3}ms", self.attention_layer_post_proj_execution_time * scale),
            format!("rotary_emb={:.3}ms", self.attention_rope_execution_time * scale),
            format!(
                "attn_kv_cache_save={:.3}ms",
                self.attention_kv_cache_save_execution_time * scale
            ),
            format!("attn_decode={:.3}ms", self.attention_decode_execution_time * scale),
            format!("attn_prefill={:.3}ms", self.attention_prefill_execution_time * scale),
            format!("attn_norm={:.3}ms", self.attn_norm_time * scale),
            format!("mlp_gate_up_proj={:.3}ms", self.mlp_layer_up_proj_execution_time * scale),
            format!("mlp_down_proj={:.3}ms", self.mlp_layer_down_proj_execution_time * scale),
            format!("mlp_activation={:.3}ms", self.mlp_layer_act_execution_time * scale),
            format!("mlp_norm={:.3}ms", self.mlp_norm_time * scale),
            format!("add={:.3}ms", self.add_time * scale),
            format!("tp_comm(all_reduce)={:.3}ms", self.tensor_parallel_communication_time),
            format!("pp_comm(send_recv)={:.3}ms", self.pipeline_parallel_communication_time),
            format!("schedule={:.3}ms", self.schedule_time),
            format!("sampler_e2e={:.3}ms", self.sampler_e2e_time),
            format!("prepare_inputs_e2e={:.3}ms", self.prepare_inputs_e2e_time),
            format!("process_model_outputs={:.3}ms", self.process_model_outputs_time),
            format!("ray_comm_time={:.3}ms", self.ray_comm_time),
        ];
        parts.join(", ")
    }
}
