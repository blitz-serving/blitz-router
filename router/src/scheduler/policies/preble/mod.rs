//! Preble scheduling policy — DSL form using `policy!` macro.
//!
//! Per-replica state lives inside [`block_hash::PrebleBlockHash`] —
//! the richer [`BlockHash`](radixtree::BlockHash) flavour swapped in
//! via the `PrefixBlockHash` alias under `--features preble-q`. The
//! tree itself is the existing engine-driven `RadixTreeBlockHash`;
//! Preble adds a 3-min `SlidingWindow<f64, Sum>` whose `len()` gives
//! `pod_load` and whose aggregate gives `pod_cost`.
//!
//! There is no `PrebleGCtx` — Preble carries no cross-request global
//! state. The DSL `gctx` is `()`. Cost-model coefficients live as
//! constants on `PrebleBlockHash` itself.
//!
//! See `docs/preble-design.md` for the abstract spec
//! (Definitions / INSERT / EXPIRE / QUERY) and `docs/dsl/policies.md`
//! §2 for the canonical DSL listing.

pub(crate) mod block_hash;
pub(crate) mod cost_model;

pub(crate) use block_hash::PrebleBlockHash;

use policy_dsl::policy;

// =========================================================================
// PrebleQ — DSL form
// =========================================================================
//
// Spec-form DSL:
//
//   Filter (preble_global_match_blocks * block_size / |req| > 0.5)
//     (Select max by (preble_owned_match_blocks(o), -preble_load(o)))
//     (Select min by preble_cost(o))
//   after default; preble_update_after(entry, chosen, all_sctx)
//
// `preble_owned_match_blocks(o)` is `o.hit_blocks` — the per-replica
// Patricia longest-match against this replica's engine-driven tree.
// `preble_global_match_blocks(observations)` is the max over replicas
// of the same; bijective with Go's `len(matchedTokens) / len(tokens)`
// gating decision since each replica's tree faithfully reflects what
// is currently cached on that replica.
//
// `preble_load` and `preble_cost` are O(1) reads of the per-replica
// `pod_load` / `pod_cost` snapshots populated in `Observation` at
// `capture_observations` time.

policy! {
    name: PrebleQ,
    gctx: (),
    body: {
        let prefix = entry.block_hash_state.get_hashes();
        let block_size = entry.block_hash_state.get_block_size();
        let global_match_blocks = preble_global_match_blocks(&observations, prefix);
        filter_then(
            &root_target(&observations),
            |_o| (global_match_blocks * block_size) as f64
                / req.input_tokens.len().max(1) as f64
                > 0.5,
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
