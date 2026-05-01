//! Preble scheduling policy — DSL form using `policy!` macro.
//!
//! See `docs/dsl-schema.md` §8 (canonical DSL listing) and the §13.1
//! rewrite table. The cost-model state lives in `PrebleGCtx` (passed
//! through `Policy::GlobalContext`), promoted from the previous
//! module-static `OnceLock` per the user's "state is a first-class
//! scope, not a backdoor" framing.
//!
//! Submodules `cost_model`, `histogram`, `router` are unchanged
//! utility modules used by `preble_cost` / `preble_update_after`
//! runtime helpers.

pub(crate) mod cost_model;
pub(crate) mod histogram;
#[allow(dead_code)]
pub(crate) mod router;

use histogram::{node_key_from_prefix, SlidingWindowHistogram};

use policy_dsl::policy;

// =========================================================================
// PrebleGCtx — Policy::GlobalContext for PrebleQ
// =========================================================================
//
// `SlidingWindowHistogram::new` requires `num_replicas`, which is only
// known when PolicyRunner first calls `schedule()`. We therefore wrap the
// histogram in `Option<...>` and lazy-init on first use via
// `ensure_init`. After init the histogram lives across all subsequent
// schedule calls — its purpose is exactly to model cross-request traffic.

#[derive(Default)]
pub(crate) struct PrebleGCtx {
    histogram: Option<SlidingWindowHistogram>,
}

impl PrebleGCtx {
    /// Initialize the histogram if not yet present. Idempotent.
    pub(crate) fn ensure_init(&mut self, num_replicas: usize) -> &mut SlidingWindowHistogram {
        if self.histogram.is_none() {
            self.histogram = Some(SlidingWindowHistogram::new(
                num_replicas,
                cost_model::TargetGpu::default(),
            ));
        }
        self.histogram.as_mut().unwrap()
    }

    /// Read-only access for `preble_cost`. Returns `None` if not yet
    /// initialized (in which case `preble_cost` falls back to the
    /// no-cost-adjustment path, equivalent to the OnceLock-uninitialized
    /// branch in the previous implementation).
    pub(crate) fn histogram(&self) -> Option<&SlidingWindowHistogram> {
        self.histogram.as_ref()
    }
}

// =========================================================================
// Histogram update — called from PrebleQ's after_extra
// =========================================================================

pub(crate) fn update_histogram_into(
    gctx: &mut PrebleGCtx,
    block_hashes: &[u64],
    hit_nblks: usize,
    input_len: usize,
    block_size: usize,
    replica_id: usize,
    num_replicas: usize,
) {
    let histogram = gctx.ensure_init(num_replicas);
    let node_key = node_key_from_prefix(block_hashes, hit_nblks);
    let context_length = hit_nblks * block_size;
    let num_tokens = input_len.saturating_sub(context_length);
    let decoding_length = histogram.default_decoding_length();
    histogram.update(
        node_key,
        num_tokens,
        context_length,
        replica_id,
        decoding_length,
    );
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
//   after default; preble_update_after(...)
//
// Stage 1 picks the longest-matching replica when prefix-match ratio
// > 50%, else stage 2 picks the lowest cost.

policy! {
    name: PrebleQ,
    gctx: PrebleGCtx,
    body: {
        filter_then(
            &root_target(&observations),
            |o| (o.hit_blocks * o.block_size) as f64 / req.input_tokens.len().max(1) as f64 > 0.5,
            |t| select_max_by(t, |o| match_blocks(req, o)),
            |t| select_min_by(t, |o| preble_cost(req, o, gctx)),
        )
    },
    after_extra: {
        let count = observations.len().max(1);
        preble_update_after(entry, &observations[chosen], count, gctx);
    }
}

