use std::time::SystemTime;
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Request {
    pub request_id: u64,
    pub prompt_len: u32,
    pub generation_len: Option<u32>,
    pub processed_tokens: u32,
    pub arrival_time: Option<SystemTime>,
    pub num_token_per_output: u32,
    pub hashes: Option<Vec<u64>>,
    pub max_generation_len: u32,

    pub hit_token_cnt: u64,
    pub ttft: Option<u64>,
}

impl Request {
    pub fn all_token_lens(&self) -> u32 {
        if self.prompt_len > self.processed_tokens {
            self.prompt_len
        } else {
            self.processed_tokens + self.num_token_per_output
        }
    }

    pub fn is_finished(&self) -> bool {
        self.processed_tokens >= self.prompt_len + self.max_generation_len
    }

    pub fn finish_ttft(&self) -> bool {
        self.prompt_len <= self.processed_tokens
    }
}

#[derive(Debug, Clone)]
pub struct BatchForPredictor {
    pub num_tokens: usize,
    pub num_prefill_tokens: Vec<usize>,
    pub num_prefill_computed_tokens: Vec<usize>,
    pub num_decode_computed_tokens: Vec<usize>,
    pub num_tokens_rounded: usize,
    pub size: usize,
    pub doing_prefill_req: Vec<u64>,
    pub prefill_done_request: Vec<u64>,
}

impl BatchForPredictor {
    pub fn new() -> Self {
        Self {
            num_tokens: 0,
            num_prefill_tokens: Vec::new(),
            num_prefill_computed_tokens: Vec::new(),
            num_decode_computed_tokens: Vec::new(),
            num_tokens_rounded: 0,
            size: 0,
            doing_prefill_req: Vec::new(),
            prefill_done_request: Vec::new(),
        }
    }

    pub fn as_ref(&self) -> &Self {
        self
    }

    pub fn round_num_tokens(&mut self, round_to: usize) {
        self.num_tokens_rounded = ((self.num_tokens + round_to - 1) / round_to) * round_to;
    }

    pub fn add_prefill(&mut self, num_tokens: usize, num_computed_tokens: usize, request_id: u64) {
        // self.num_tokens += num_tokens;
        self.num_prefill_tokens.push(num_tokens);
        self.num_prefill_computed_tokens.push(num_computed_tokens);
        // self.doing_prefill_req.push(request_id);
    }

    pub fn add_decode(&mut self, num_tokens: usize, num_computed_tokens: usize) {
        self.num_decode_computed_tokens.push(num_computed_tokens);
        // self.num_tokens += num_tokens;
    }

    pub fn curr_num_tokens(&self) -> usize {
        self.num_prefill_tokens.iter().sum::<usize>() + self.num_decode_computed_tokens.len()
    }

    pub fn calculate_batch_metric(&mut self) {
        self.num_tokens =
            self.num_prefill_tokens.iter().sum::<usize>() + self.num_decode_computed_tokens.len();
        self.size = self.num_prefill_tokens.len() + self.num_decode_computed_tokens.len();
        self.num_tokens_rounded = ((self.num_tokens + 7) / 8) * 8;
    }
}
