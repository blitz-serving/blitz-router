// lmetric (multiplicative) scheduling policy.
//
// Multiplies two carefully chosen indicators:
//   Score = P_token × BS
// and routes to the instance with the minimal score.
//
// Both values are PROSPECTIVE — they represent the state AFTER
// scheduling this request to the instance:
//   P_token = queued_prefill_tokens + this_request_new_prefill
//   BS      = current_batch_size + 1
//
// The +1 on BS is critical: without it, idle instances (BS=0) always
// score 0 regardless of cache, defeating KV-cache awareness.

use super::{
    AssignScore, DeterministicPolicy, EmptyContext, Entry, NumHitKvBlock, QueuePlusPlus,
};
use crate::kvcache::BlockHash;
use crate::ScheduleContext;

/// lmetric multiplicative scoring: `P_token × BS` (prospective values).
pub(crate) struct LmetricQ;

impl QueuePlusPlus for LmetricQ {
    type QueueContext = EmptyContext;
    type Measure = usize;
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

        // Prospective P-token: queued prefill + this request's new prefill
        let new_prefill_tokens = entry
            .request
            .input_tokens
            .len()
            .saturating_sub(hit_nblks * block_size);
        let p_token = lmetric.prefill_tokens.max(0) as usize + new_prefill_tokens;

        // Prospective BS: current batch size + 1 (this request)
        let bs = lmetric.bs + 1;

        Some((
            AssignScore::Least(p_token * bs),
            Some(hit_nblks),
        ))
    }
}

impl DeterministicPolicy for LmetricQ {}
