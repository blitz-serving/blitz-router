// Bounded-most-hit scheduling policy.
//
// Selects the replica with the most KV cache block hits, subject to a
// configurable prefill token budget constraint.

use super::{
    AssignScore, DeterministicPolicy, EmptyContext, Entry, NumHitKvBlock, QueuePlusPlus,
};
use crate::kvcache::BlockHash;
use crate::{ScheduleContext, WAITINGT_PREFILL_TOKEN_BOUND};

/// Join-bounded-most-hit queue: picks the replica that has the highest
/// number of KV cache hits, provided its waiting prefill tokens are
/// within the bound.
pub(crate) struct JBoundMostHitQ2;

impl QueuePlusPlus for JBoundMostHitQ2 {
    type QueueContext = EmptyContext;
    type Measure = usize;
    type Weight = ();

    /// Returns: hit block count
    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        entry: &Entry,
        _qctx: &Self::QueueContext,
        sctx: &ScheduleContext,
    ) -> Option<(AssignScore<Self::Measure, ()>, Option<NumHitKvBlock>)> {
        let ScheduleContext { lmetric, block_hash } = sctx;

        if lmetric.prefill_tokens.max(0) as usize >= WAITINGT_PREFILL_TOKEN_BOUND {
            None
        } else {
            // postcond: current waiting prefill tokens are within the bound
            let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
            Some((AssignScore::Greatest(hit_nblks), Some(hit_nblks)))
        }
    }
}

impl DeterministicPolicy for JBoundMostHitQ2 {}
