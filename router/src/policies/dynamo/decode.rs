// AI-Dynamo T2: Decode-instance perspective.
//
// Bijective mapping of Dynamo's logit formula as seen by a DECODE instance
// in PD-disaggregated deployment.
//
// In PD-disagg, a decode instance holds KV blocks for active requests,
// so `decode_blocks` data is present and real. The prefill queue
// (`active_tokens`) is ~0 because decode instances don't do prefill.
//
// ── Dynamo source: selector.rs:142-150 ───────────────────────────────────
//
//   let prefill_token = *prefill_tokens.get(&worker).unwrap_or(&isl);
//   let potential_prefill_block = (prefill_token as f64) / (block_size as f64);
//
//   let decode_block = *decode_blocks
//       .get(&worker)
//       .unwrap_or(&(potential_prefill_block.floor() as usize))
//       as f64;
//
//   let logit = overlap_weight * potential_prefill_block + decode_block;
//
// ── Dynamo source: single.rs:206-218 ─────────────────────────────────────
//
//   pub fn potential_blocks_and_tokens(&self, token_sequence, isl, overlap)
//       -> (usize, usize) {
//       let potential_blocks = if let Some(token_seq) = token_sequence {
//           self.new_blocks(token_seq) + self.active_blocks()
//       } else {
//           self.active_blocks()
//       };
//       let potential_tokens = self.new_tokens(isl, overlap) + self.active_tokens;
//       (potential_blocks, potential_tokens)
//   }
//
// ── Mapping (Dynamo → blitz-router, decode instance) ─────────────────────
//
//   new_tokens(isl, overlap) → new_prefill_tokens = ISL - hit_nblks * block_size
//   active_tokens            → ~0 (decode instance has no prefill queue)
//   prefill_token            → new_prefill_tokens only (per-request)
//   new_blocks(token_seq)    → total_input_blocks - hit_nblks
//   active_blocks()          → lmetric.all_tokens / block_size (proxy: total KV blocks held)
//   decode_blocks            → new_blocks + active_blocks (REAL data, no fallback)
//
// ─────────────────────────────────────────────────────────────────────────

use super::super::{
    AssignScore, DeterministicPolicy, EmptyContext, Entry, NumHitKvBlock, QueuePlusPlus,
};
use crate::kvcache::BlockHash;
use crate::ScheduleContext;

const DYNAMO_OVERLAP_WEIGHT: f64 = 1.0;

/// Dynamo T2: decode-instance perspective.
///
/// `logit = overlap_weight * potential_prefill_block + decode_block`
///
/// where `potential_prefill_block` uses per-request prefill only (no queue,
/// because decode instances don't have a prefill queue), and `decode_block`
/// is real data (`new_blocks + active_blocks`).
pub(crate) struct DynamoDecodeQ;

impl QueuePlusPlus for DynamoDecodeQ {
    type QueueContext = EmptyContext;
    type Measure = u64;
    type Weight = ();

    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<Self::Measure, ()>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash } = sctx;

        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
        let block_size = entry.block_hash_state.get_block_size();

        // Dynamo: let prefill_token = new_tokens(isl, overlap) + active_tokens;
        // Decode instance: active_tokens ≈ 0, so prefill_token ≈ new_tokens only
        let new_prefill_tokens = entry
            .request
            .input_tokens
            .len()
            .saturating_sub(hit_nblks * block_size);

        // Dynamo: let potential_prefill_block = (prefill_token as f64) / (block_size as f64);
        let potential_prefill_block = new_prefill_tokens as f64 / block_size as f64;

        // Dynamo: let potential_blocks = self.new_blocks(token_seq) + self.active_blocks();
        let total_input_blocks =
            (entry.request.input_tokens.len() + block_size - 1) / block_size;
        let new_blocks = total_input_blocks.saturating_sub(hit_nblks);
        let active_blocks = lmetric.all_tokens / block_size;

        // Dynamo: let decode_block = *decode_blocks.get(&worker)...
        // Decode instance: decode_blocks present → no fallback
        let decode_block = (new_blocks + active_blocks) as f64;

        // Dynamo: let logit = overlap_weight * potential_prefill_block + decode_block;
        let logit = DYNAMO_OVERLAP_WEIGHT * potential_prefill_block + decode_block;

        // Convert to integer for AssignScore (multiply by 1000 for precision)
        let score = (logit * 1000.0) as u64;

        Some((AssignScore::Least(score), Some(hit_nblks)))
    }
}

impl DeterministicPolicy for DynamoDecodeQ {}
