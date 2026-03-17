// Round-robin scheduling policy.

use super::{
    AssignScore, DeterministicPolicy, Entry, NumHitKvBlock, QueuePlusPlus, RRContext,
};
use crate::ScheduleContext;

/// Round-robin queue: cycles through replicas sequentially.
pub(crate) struct RRQueue;

impl QueuePlusPlus for RRQueue {
    type QueueContext = RRContext;
    type Measure = ();
    type Weight = ();

    fn eligible_with_kvblock_hit(
        replica_id: usize,
        _entry: &Entry,
        qctx: &RRContext,
        _sctx: &ScheduleContext,
    ) -> Option<(AssignScore<(), ()>, Option<NumHitKvBlock>)> {
        if qctx.next_replica_id == replica_id {
            Some((AssignScore::Least(()), None))
        } else {
            None
        }
    }
}

impl DeterministicPolicy for RRQueue {}
