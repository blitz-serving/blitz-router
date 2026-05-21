//! Preble scheduling policy — DSL form using `policy!` macro.
//!
//! Three variants share the prefix-match `BlockHash` flavour
//! ([`block_hash`]) but plug different per-pod aggregates into the
//! load-balancing branch:
//!
//! - [`PrebleQ`] (feature `preble-q`): paper-faithful — load-balancing
//!   branch is `min Σ (PT_r + DT_r)` over the 3-min window.
//! - [`PrebleBsQ`] (feature `preble-bs-q`): load-balancing branch is
//!   `min Σ BS` over the 3-min window (engine-step-sampled).
//! - [`PrebleTpsQ`] (feature `preble-tps-q`): load-balancing branch
//!   is `max forward-step count` over the 3-min window.
//!
//! All three share the KV$-aware branch: filter by global match-ratio
//! `> T` then `select_max_by(owned_match_blocks)`. The cost model is
//! unused by the BS/TPS variants.
//!
//! See `docs/preble-design.md` for the abstract spec
//! (Definitions / INSERT / EXPIRE / QUERY) and `docs/dsl/policies.md`
//! §2 for the canonical DSL listing.

pub(crate) mod block_hash;
#[cfg(feature = "preble-q")]
pub(crate) mod cost_model;

#[cfg(feature = "preble-q")]
pub(crate) use block_hash::PrebleBlockHash;
#[cfg(feature = "preble-bs-q")]
pub(crate) use block_hash::PrebleBsBlockHash;
#[cfg(feature = "preble-tps-q")]
pub(crate) use block_hash::PrebleTpsBlockHash;

#[cfg(any(feature = "preble-q", feature = "preble-bs-q", feature = "preble-tps-q"))]
use crate::scheduler::state::PREBLE_MATCH_RATIO_T;
use policy_dsl::policy;

// =========================================================================
// PrebleQ — DSL form (paper-faithful)
// =========================================================================
//
// Spec-form DSL:
//
//   Filter (preble_global_match_blocks * block_size / |req| > T)
//     (Select max by (preble_owned_match_blocks(o), -preble_load(o)))
//     (Select min by preble_cost(o))
//   after default; preble_update_after(entry, chosen, all_sctx)
//
// `T` defaults to 0.5 (paper §QUERY) and is CLI-tunable via
// `--preble-match-ratio-threshold`.

#[cfg(feature = "preble-q")]
policy! {
    name: PrebleQ,
    gctx: (),
    body: {
        let prefix = entry.block_hash_state.get_hashes();
        let block_size = entry.block_hash_state.get_block_size();
        let global_match_blocks = preble_global_match_blocks(&observations, prefix);
        let threshold = PREBLE_MATCH_RATIO_T.get().copied().unwrap_or(0.5) as f64;
        filter_then(
            &root_target(&observations),
            |_o| (global_match_blocks * block_size) as f64
                / req.input_tokens.len().max(1) as f64
                > threshold,
            |t| select_max_by(
                t,
                |o| (
                    preble_owned_match_blocks(o),
                    -(preble_load(o) as i64),
                ),
            ),
            |t| select_min_by(t, preble_cost),
        )
    },
    after_extra: {
        preble_update_after(entry, &observations[chosen], all_sctx).await;
    }
}

// =========================================================================
// PrebleBsQ — load-balancing branch = min Σ BS over 3-min window
// =========================================================================
//
// Same KV$-aware branch shape as PrebleQ — filter by match ratio,
// then max owned match blocks with the load-balancing-branch metric
// as the tie-break. Load-balancing branch replaces cost-min with
// bs-sum-min: pick the replica whose accumulated batch-size samples
// over the past 3 min are smallest, i.e. has been least loaded.
//
// The KV$-aware-branch tie-break (negated bs_sum) keeps the same
// load metric in play across both branches, so that two replicas
// with equal longest-match still resolve to the less-loaded one.
//
// No `after_extra` hook: the bs window is fed by the colocation loop
// every forward step (one push per step), independent of admissions.

#[cfg(feature = "preble-bs-q")]
policy! {
    name: PrebleBsQ,
    gctx: (),
    body: {
        let prefix = entry.block_hash_state.get_hashes();
        let block_size = entry.block_hash_state.get_block_size();
        let global_match_blocks = preble_global_match_blocks(&observations, prefix);
        let threshold = PREBLE_MATCH_RATIO_T.get().copied().unwrap_or(0.5) as f64;
        filter_then(
            &root_target(&observations),
            |_o| (global_match_blocks * block_size) as f64
                / req.input_tokens.len().max(1) as f64
                > threshold,
            |t| select_max_by(
                t,
                |o| (
                    preble_owned_match_blocks(o),
                    // f64 negation: larger tuple wins ⇒ smaller bs_sum wins.
                    -preble_bs_sum(o),
                ),
            ),
            |t| select_min_by(t, preble_bs_sum),
        )
    },
}

// =========================================================================
// PrebleTpsQ — load-balancing branch = max forward-step count over 3-min window
// =========================================================================
//
// Same KV$-aware branch shape as PrebleQ — filter by match ratio,
// then max owned match blocks with the load-balancing-branch metric
// as the tie-break. Load-balancing branch picks the replica with the
// most forward steps in the past 3 min — i.e. highest throughput,
// lowest load. Like PrebleBsQ, the window is engine-step-driven, no
// `after_extra` hook.
//
// The KV$-aware-branch tie-break is `+tps_count` (no negation:
// higher tps_count ⇒ less loaded ⇒ preferred), keeping the same
// load metric in play across both branches.

#[cfg(feature = "preble-tps-q")]
policy! {
    name: PrebleTpsQ,
    gctx: (),
    body: {
        let prefix = entry.block_hash_state.get_hashes();
        let block_size = entry.block_hash_state.get_block_size();
        let global_match_blocks = preble_global_match_blocks(&observations, prefix);
        let threshold = PREBLE_MATCH_RATIO_T.get().copied().unwrap_or(0.5) as f64;
        filter_then(
            &root_target(&observations),
            |_o| (global_match_blocks * block_size) as f64
                / req.input_tokens.len().max(1) as f64
                > threshold,
            |t| select_max_by(
                t,
                |o| (
                    preble_owned_match_blocks(o),
                    preble_tps_count(o),
                ),
            ),
            |t| select_max_by(t, preble_tps_count),
        )
    },
}
