// Random scheduling policy.
//
// Uniformly selects a replica at random among all eligible candidates.

use super::{
    AssignScore, EmptyContext, Entry, NaiiveLattice, NumHitKvBlock, QueuePlusPlus,
    SamplerFn, ScheduleStep, StochasticPolicy, step_stochastic,
};
use crate::ScheduleContext;

use rand::{thread_rng, Rng};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Random queue: selects a replica uniformly at random.
pub(crate) struct RandomQ;

impl QueuePlusPlus for RandomQ {
    type QueueContext = EmptyContext;
    type Measure = ();
    type Weight = (); // `()` is also a lattice

    fn eligible_with_kvblock_hit(
        _replica_id: usize,
        _entry: &Entry,
        _qctx: &Self::QueueContext,
        _sctx: &ScheduleContext,
    ) -> Option<(AssignScore<(), Self::Weight>, Option<NumHitKvBlock>)> {
        Some((AssignScore::Weighted(()), None))
    }
}

fn random_sampler(
    all_scores: Vec<(usize, (), Option<usize>)>,
    _lower_bound: (),
    _upper_bound: (),
) -> (usize, Option<usize>) {
    let mut rng = thread_rng();
    let x = rng.gen_range(0..all_scores.len());
    let (id, _, hit) = all_scores.get(x).copied().unwrap();
    (id, hit)
}

impl StochasticPolicy for RandomQ {
    fn sampler() -> SamplerFn<Self::Weight> {
        random_sampler
    }
}

impl ScheduleStep for RandomQ {
    fn schedule_step(
        entry: &Entry,
        qctx: &Self::QueueContext,
        all_sctx: &[Arc<Mutex<ScheduleContext>>],
    ) -> impl std::future::Future<Output = Option<usize>> + Send {
        step_stochastic::<Self>(entry, qctx, all_sctx)
    }
}
