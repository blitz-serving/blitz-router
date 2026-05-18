//! Preble scheduling policy — DSL form using `policy!` macro.
//!
//! See `docs/dsl/policies.md` §2 (canonical DSL listing) and the
//! `docs/dsl/implementation.md` §2.1 rewrite table. The cost-model state lives
//! in `PrebleGCtx` (passed through `Policy::GlobalContext`), promoted from the
//! previous module-static `OnceLock` per the user's "state is a first-class
//! scope, not a backdoor" framing.
//!
//! Submodules `cost_model` and `histogram` are utility modules used by
//! `preble_cost` / `preble_load` / `preble_update_after` runtime helpers.

pub(crate) mod cost_model;
pub(crate) mod histogram;

use histogram::SlidingWindowHistogram;

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

    /// Read-only access for `preble_cost` / `preble_load`. Returns
    /// `None` if not yet initialized (callers fall back to a
    /// no-cost-adjustment / zero-load path, equivalent to the
    /// OnceLock-uninitialized branch in the previous implementation).
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
    // The "node" identity in the original AIBrix Go is the longest
    // matched prefix path (TreeNode pointer). In our port that's a
    // slice of the request's block hashes truncated to the matched
    // length.
    let prefix_len = hit_nblks.min(block_hashes.len());
    if prefix_len == 0 {
        // Go uses the root sentinel for no-prefix-match requests; we
        // elide it (it has no per-node cost contribution).
        return;
    }
    let prefix = &block_hashes[..prefix_len];
    let context_length = prefix_len * block_size;
    let num_tokens = input_len.saturating_sub(context_length);
    let decoding_length = histogram.default_decoding_length();
    histogram.update(
        prefix,
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
// Spec-form DSL (`docs/dsl/policies.md` §2):
//
//   Filter (match_blocks(req, sctx) / req.input_tokens > 0.5)
//     (Select max by (match_blocks(req, sctx), -load(sctx)))
//     (Select min by preble_cost(req, sctx))
//   after default; preble_update_after(...)
//
// Stage 1 picks the longest-matching replica when prefix-match ratio
// > 50%; ties on match_blocks are broken by lower per-replica load
// (`preble_load`), bijective with AIBrix Go's `getPodLoad` tie-break.
// Stage 2 picks the lowest-cost replica via the per-(node, replica)
// histogram cost.

policy! {
    name: PrebleQ,
    gctx: PrebleGCtx,
    body: {
        filter_then(
            &root_target(&observations),
            |o| (o.hit_blocks * o.block_size) as f64 / req.input_tokens.len().max(1) as f64 > 0.5,
            |t| select_max_by(t, |o| (match_blocks(req, o), -(preble_load(o, gctx) as i64))),
            |t| select_min_by(t, |o| preble_cost(req, o, gctx)),
        )
    },
    after_extra: {
        let count = observations.len().max(1);
        preble_update_after(entry, &observations[chosen], count, gctx);
    }
}
