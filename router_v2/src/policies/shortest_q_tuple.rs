// Shortest-queue-tuple scheduling policy.
//
// Uses a two-element tuple (waiting_requests, running_requests) as the
// comparison metric, selecting the replica with the smallest tuple in
// lexicographic order.

use super::{
    AssignScore, DeterministicPolicy, EmptyContext, Entry, NumHitKvBlock, QueuePlusPlus,
};
use crate::kvcache::BlockHash;
use crate::ScheduleContext;

/// Join-shortest-queue (tuple): compares replicas by
/// `(waiting_requests, running_requests)` and picks the smallest.
pub(crate) struct JShortestQTuple;

impl QueuePlusPlus for JShortestQTuple {
    type QueueContext = EmptyContext;
    type Measure = (usize, usize);
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
            AssignScore::Least((num_waiting_requests, num_running_requests)),
            Some(hit_nblks),
        ))
    }
}

impl DeterministicPolicy for JShortestQTuple {}
