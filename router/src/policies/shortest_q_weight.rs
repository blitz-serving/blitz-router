// Shortest-queue-weight scheduling policy.
//
// Uses a weighted linear combination of waiting and running request
// counts as the comparison metric.

use super::{
    AssignScore, DeterministicPolicy, EmptyContext, Entry, NumHitKvBlock, QueuePlusPlus,
};
use crate::kvcache::BlockHash;
use crate::ScheduleContext;

/// Join-shortest-queue (weighted): scores replicas by
/// `waiting_requests * 4 + running_requests` and picks the lowest.
pub(crate) struct JShortestQWeight;

impl QueuePlusPlus for JShortestQWeight {
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

        let num_waiting_requests = lmetric.waiting_reqs;
        let num_running_requests = lmetric.bs - lmetric.waiting_reqs;
        let hit_nblks = block_hash.get(entry.block_hash_state.get_hashes());
        Some((
            AssignScore::Least(num_waiting_requests * 4 + num_running_requests),
            Some(hit_nblks),
        ))
    }
}

impl DeterministicPolicy for JShortestQWeight {}
