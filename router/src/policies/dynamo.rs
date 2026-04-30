//! `dynamo-q` (T1, prefill perspective) and `dynamo-po-q` (T2, decode
//! perspective; renamed from `dynamo-decoupled-q`). See dsl-schema.md §8
//! and `.claude/memory/dynamo_pd_colocated_terms.md` for the formula
//! ablation rationale.
//!
//! T1 logit:  `w · (prefill_tokens / block_size) + (prefill_tokens / block_size).floor()`
//!            (decode_block fallback ≈ prefill_block; engine state IS
//!             folded into the prefill term)
//!
//! T2 logit:  `w · (new_tokens / block_size) + (new_blocks + decode_blocks)`
//!            (per-request prefill only; decode_block is real)
//!
//! Both bijectively match Dynamo's `selector.rs:150` formula.

use policy_dsl::policy;

const DYNAMO_OVERLAP_WEIGHT: f64 = 1.0;

// dynamo-q (T1, prefill perspective)
policy! {
    name: DynamoQ,
    gctx: (),
    body: {
        select_min_by(&root_target(&observations), |o| {
            let block_size = o.block_size.max(1);
            let prefill_token = prefill_tokens(req, o);
            let potential_prefill_block = prefill_token as f64 / block_size as f64;
            let decode_block = potential_prefill_block.floor();
            let logit = DYNAMO_OVERLAP_WEIGHT * potential_prefill_block + decode_block;
            (logit * 1000.0) as u64
        })
    },
}

// dynamo-po-q (T2, decode perspective; per-request prefill only)
policy! {
    name: DynamoDecodeQ,
    gctx: (),
    body: {
        select_min_by(&root_target(&observations), |o| {
            let block_size = o.block_size.max(1);
            let new_prefill_tokens = new_tokens(req, o);
            let potential_prefill_block = new_prefill_tokens as f64 / block_size as f64;
            let total_input_blocks = (req.input_tokens.len() + block_size - 1) / block_size;
            let new_blocks = total_input_blocks.saturating_sub(o.hit_blocks);
            let active_blocks = decode_blocks(o);
            let decode_block = (new_blocks + active_blocks) as f64;
            let logit = DYNAMO_OVERLAP_WEIGHT * potential_prefill_block + decode_block;
            (logit * 1000.0) as u64
        })
    },
}
