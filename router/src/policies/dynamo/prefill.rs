// AI-Dynamo T1: Prefill-instance perspective.
//
// Bijective mapping of Dynamo's logit formula as seen by a PREFILL instance
// in PD-disaggregated deployment.
//
// In PD-disagg, a prefill instance has NO decode workload, so `decode_blocks`
// data is absent for that worker. Dynamo's code handles this via fallback:
//   decode_block = potential_prefill_block.floor()
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
// ── Dynamo source: single.rs:194-204 ─────────────────────────────────────
//
//   pub fn new_tokens(&self, isl: usize, overlap: u32) -> usize {
//       let cached_tokens = (overlap as usize) * self.block_size;
//       isl.checked_sub(cached_tokens).unwrap_or(0)
//   }
//
// ── Dynamo source: single.rs:185-192 ─────────────────────────────────────
//
//   pub fn mark_prefill_completed(&mut self, request_id: &RequestId) {
//       if let Some(tokens) = self.prefill_tokens.remove(request_id) {
//           self.active_tokens = self.active_tokens.checked_sub(tokens)
//               .expect("active_tokens underflow");
//       }
//   }
//
// ── Mapping (Dynamo → blitz-router, prefill instance) ────────────────────
//
//   new_tokens(isl, overlap) → new_prefill_tokens = ISL - hit_nblks * block_size
//   active_tokens            → queued_prefill_tokens = lmetric.prefill_tokens
//   prefill_token            → new_prefill_tokens + queued_prefill_tokens
//   decode_blocks            → ABSENT (prefill instance) → fallback to prefill_block
//
// ─────────────────────────────────────────────────────────────────────────

use super::super::{
    AssignScore, DeterministicPolicy, EmptyContext, Entry, NumHitKvBlock, QueuePlusPlus,
};
use crate::kvcache::BlockHash;
use crate::ScheduleContext;

const DYNAMO_OVERLAP_WEIGHT: f64 = 1.0;

/// Dynamo T1: prefill-instance perspective.
///
/// `logit = overlap_weight * potential_prefill_block + decode_block`
///
/// where `decode_block = floor(potential_prefill_block)` (fallback,
/// because a prefill instance in PD-disagg has no decode_blocks data).
pub(crate) struct DynamoPrefillQ;

impl QueuePlusPlus for DynamoPrefillQ {
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
        let new_prefill_tokens = entry
            .request
            .input_tokens
            .len()
            .saturating_sub(hit_nblks * block_size);
        let queued_prefill_tokens = lmetric.prefill_tokens.max(0) as usize;
        let prefill_token = new_prefill_tokens + queued_prefill_tokens;

        // Dynamo: let potential_prefill_block = (prefill_token as f64) / (block_size as f64);
        let potential_prefill_block = prefill_token as f64 / block_size as f64;

        // Dynamo: let decode_block = *decode_blocks.get(&worker)
        //             .unwrap_or(&(potential_prefill_block.floor() as usize)) as f64;
        // Prefill instance: no decode_blocks → fallback
        let decode_block = potential_prefill_block.floor();

        // Dynamo: let logit = overlap_weight * potential_prefill_block + decode_block;
        let logit = DYNAMO_OVERLAP_WEIGHT * potential_prefill_block + decode_block;

        // Convert to integer for AssignScore (multiply by 1000 for precision)
        let score = (logit * 1000.0) as u64;

        Some((AssignScore::Least(score), Some(hit_nblks)))
    }
}

impl DeterministicPolicy for DynamoPrefillQ {}
