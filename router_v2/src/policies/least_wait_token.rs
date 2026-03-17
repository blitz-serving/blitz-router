// Least-wait-token scheduling policy.
//
// Assigns requests to the replica with the fewest total waiting tokens
// (existing wait queue + new tokens from this request).

use super::{
    AssignScore, DeterministicPolicy, EmptyContext, Entry, NumHitKvBlock, QueuePlusPlus,
};
use crate::kvcache::BlockHash;
use crate::ScheduleContext;

/// Least-wait-token queue: selects the replica where the total number of
/// waiting prefill tokens (including the new request) would be smallest.
pub(crate) struct JLeastWaitTokenQ;

impl QueuePlusPlus for JLeastWaitTokenQ {
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
        let request = &entry.request;

        let num_waiting_tokens = lmetric.prefill_tokens;
        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
        let num_new_tokens =
            request.input_tokens.len() - hit_nblks * entry.block_hash_state.get_block_size();

        Some((
            AssignScore::Least(num_waiting_tokens.max(0) as usize + num_new_tokens),
            Some(hit_nblks),
        ))
    }
}

impl DeterministicPolicy for JLeastWaitTokenQ {}
