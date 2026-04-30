//! Preble scheduling policy — DSL form using `policy!` macro.
//!
//! See `docs/dsl-schema.md` §8 (canonical DSL listing) and the §13.1
//! rewrite table. The cost-model state lives in a `OnceLock<Mutex<...>>`
//! local to this module (its conceptual home is `GlobalContext`, but
//! migrating that belongs to a follow-up — the scope of this DSL pass
//! is to retire `QueuePlusPlus`, not to redesign Preble's state plumbing).
//!
//! Submodules `cost_model`, `histogram`, `router` are unchanged
//! utility modules used by `preble_cost` / `update_histogram` runtime
//! helpers.

pub(crate) mod cost_model;
pub(crate) mod histogram;
#[allow(dead_code)]
pub(crate) mod router;

use std::sync::{Mutex, OnceLock};

use histogram::{node_key_from_prefix, SlidingWindowHistogram};

use policy_dsl::policy;

// =========================================================================
// Global Preble state — temporary location pending GlobalContext migration
// =========================================================================

static PREBLE_STATE: OnceLock<Mutex<PrebleGlobalState>> = OnceLock::new();

pub(crate) struct PrebleGlobalState {
    histogram: SlidingWindowHistogram,
}

impl PrebleGlobalState {
    pub(crate) fn allocation_cost_per_replica(&self) -> Vec<f64> {
        self.histogram.get_allocation_cost_per_replica()
    }
}

fn get_or_init_state(num_replicas: usize) -> &'static Mutex<PrebleGlobalState> {
    PREBLE_STATE.get_or_init(|| {
        Mutex::new(PrebleGlobalState {
            histogram: SlidingWindowHistogram::new(
                num_replicas,
                cost_model::TargetGpu::default(),
            ),
        })
    })
}

/// Read-only access for the `preble_cost` runtime helper.
pub(crate) fn peek_state() -> Option<&'static Mutex<PrebleGlobalState>> {
    PREBLE_STATE.get()
}

/// Initialise the Preble state at router startup.
#[allow(unused)]
pub(crate) fn init_preble_state(num_replicas: usize) {
    let _ = get_or_init_state(num_replicas);
}

/// Update the Preble histogram after a routing decision (called from the
/// `preble_update_after` runtime helper, which is itself invoked from
/// PrebleQ's `after_extra` clause).
pub(crate) fn update_histogram(
    block_hashes: &[u64],
    hit_nblks: usize,
    input_len: usize,
    block_size: usize,
    replica_id: usize,
    num_replicas: usize,
) {
    let state_lock = get_or_init_state(num_replicas);
    if let Ok(mut state) = state_lock.lock() {
        let node_key = node_key_from_prefix(block_hashes, hit_nblks);
        let context_length = hit_nblks * block_size;
        let num_tokens = input_len.saturating_sub(context_length);
        let decoding_length = state.histogram.default_decoding_length();
        state.histogram.update(
            node_key,
            num_tokens,
            context_length,
            replica_id,
            decoding_length,
        );
    }
}

// =========================================================================
// PrebleQ — DSL form
// =========================================================================
//
// Paper-form DSL (`docs/dsl-schema.md` §8):
//
//   Filter (match_blocks(req, sctx) / req.input_tokens > 0.5)
//     (Select max by match_blocks(req, sctx))
//     (Select min by preble_cost(req, sctx))
//   after default; preble_update_after(req, &observations[chosen], Count)
//
// The dual-stage routing: stage 1 picks the longest-matching replica
// when prefix-match ratio > 50%, else stage 2 picks the lowest cost.

policy! {
    name: PrebleQ,
    gctx: (),
    body: {
        filter_then(
            &root_target(&observations),
            |o| (o.hit_blocks * o.block_size) as f64 / req.input_tokens.len().max(1) as f64 > 0.5,
            |t| select_max_by(t, |o| match_blocks(req, o)),
            |t| select_min_by(t, |o| preble_cost(req, o)),
        )
    },
    after_extra: {
        let count = observations.len().max(1);
        preble_update_after(entry, &observations[chosen], count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_global_state() {
        init_preble_state(4);
        let state = PREBLE_STATE.get().unwrap();
        let guard = state.lock().unwrap();
        assert_eq!(guard.histogram.num_replicas(), 4);
    }
}
