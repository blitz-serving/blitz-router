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
    // Go's "node" identity from `cache.AddPrefix(tokens, ctx.Model, "")`
    // is the leaf created at the FULL input depth — every request
    // creates/touches a leaf there, regardless of how much of the
    // prefix already existed. `prefix_cache_preble.go:459, 562`.
    //
    // For the blockified port, the tree path is `block_hashes` in full
    // (the request's BlockHashState produces one hash per full block of
    // the input). The leaf's `context_length` is the FULL input token
    // count — Go's `leafNode.ContextLength()`.
    if block_hashes.is_empty() {
        // Root sentinel — Go: the node returned by AddPrefix for an
        // empty input would be root, never recorded in `histogram`.
        return;
    }
    let prefix = block_hashes;
    let context_length = input_len;
    // Cached tokens at the leaf = number of matched blocks × block_size
    // (clamped to input_len for the partial-tail-block case).
    let cached_tokens = (hit_nblks.saturating_mul(block_size)).min(input_len);
    let num_tokens = input_len.saturating_sub(cached_tokens);
    let decoding_length = histogram.default_decoding_length();
    histogram.update(
        prefix,
        num_tokens,
        context_length,
        replica_id,
        decoding_length,
    );
    // Mirror Go's `evictionLoop` (1Hz): lazy in-path eviction
    // throttled to once per second. We run AFTER `update` so the
    // freshly-inserted entry isn't immediately considered for decay.
    histogram.evict_if_due(std::time::Instant::now());
}

// =========================================================================
// PrebleQ — DSL form
// =========================================================================
//
// Spec-form DSL (`docs/dsl/policies.md` §2):
//
//   Filter (preble_global_match_blocks * block_size / |req| > 0.5)
//     (Select max by (preble_owned_match_blocks(req, sctx, gctx),
//                     -preble_load(sctx, gctx)))
//     (Select min by preble_cost(req, sctx))
//   after default; preble_update_after(...)
//
// The threshold uses the SHARED Preble tree's longest match
// (`preble_global_match_blocks` — single global value across all
// replicas) so the >0.5 entry decision is bijective with Go's
// `matchRatio := len(matchedTokens) / len(tokens)` at
// `prefix_cache_preble.go:476`.
//
// Stage 1 (when ratio > 0.5) selects max by (owned_match, -load):
//   * `owned_match`: depth of the deepest Preble-tree ancestor of the
//     request prefix that this replica owns — Go's
//     `prefixMatches[i].matchLength` after the ancestor walk.
//   * `-load`: lower-load tiebreaker among replicas with equal owned
//     match — Go's `getPodLoad` pick at
//     `prefix_cache_preble.go:511-520`.
//
// Stage 2 (otherwise) uses `preble_cost` — bare histogram cost (D5
// applied, no `(new_tokens + all_tokens)` overlay) with constant
// 0.15 s/tok decode (D6 applied, no live `LMetric.tpot`). Stage 2 is
// intentionally blind to live execution state, matching Go.

policy! {
    name: PrebleQ,
    gctx: PrebleGCtx,
    body: {
        let prefix = entry.block_hash_state.get_hashes();
        let block_size = entry.block_hash_state.get_block_size();
        let global_match_blocks = preble_global_match_blocks(gctx, prefix);
        filter_then(
            &root_target(&observations),
            |_o| (global_match_blocks * block_size) as f64
                / req.input_tokens.len().max(1) as f64
                > 0.5,
            |t| select_max_by(
                t,
                |o| (
                    preble_owned_match_blocks(o, gctx, prefix),
                    -(preble_load(o, gctx) as i64),
                ),
            ),
            |t| select_min_by(t, |o| preble_cost(o, gctx)),
        )
    },
    after_extra: {
        let count = observations.len().max(1);
        preble_update_after(entry, &observations[chosen], count, gctx);
    }
}
