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
// bs-sum-min: pick the replica whose accumulated BS samples over
// the past 3 min are smallest, i.e. has been least loaded.
//
// The KV$-aware-branch tie-break (negated bs_sum) keeps the same
// load metric in play across both branches, so that two replicas
// with equal longest-match still resolve to the less-loaded one.
//
// **Push-both window** (see `PrebleBsBlockHash` docs): admission
// events from this `after_extra` hook ensure the metric is responsive
// before the next routing decision; SSE forward-step events from the
// colocation loop capture runtime BS occupancy. The combination
// approximates ∫ BS(t) dt × event_rate. The earlier admission-only
// variant of this fix addressed the burst-collapse from the original
// step-only design but had semantic drift toward "arrival-weighted
// queue pressure"; adding SSE samples back restores runtime occupancy
// fidelity without re-opening the lag gap (admission push happens
// before the next decision regardless of SSE timing).

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
    after_extra: {
        preble_bs_update_after(entry, &observations[chosen], all_sctx).await;
    }
}

// =========================================================================
// PrebleTpsQ — load-balancing branch = max forward-step count over window
// =========================================================================
//
// Same KV$-aware branch shape as PrebleQ — filter by match ratio,
// then max owned match blocks with the load-balancing-branch metric
// as the tie-break. Load-balancing branch picks the replica with the
// most forward steps in the window — i.e. highest throughput, lowest
// load. The window is engine-step-driven.
//
// Design 1 (idle-engine sentinel): idle engines (bs=0) get
// `preble_tps_count = usize::MAX`, so they categorically win over
// any busy engine. This prevents the cold-start mode-collapse where
// the first chosen replica would otherwise lock in every subsequent
// admission (only it has nonzero tps_count; idle peers stay at 0).
//
// Design 2 (idle-period compensation): when an admission lifts a
// previously-idle engine to busy, the `after_extra` hook
// retroactively credits the idle gap at decode_fps rate (default
// 120). This keeps a recently-idle engine competitive with
// continuously-busy peers in subsequent `select_max_by` rounds,
// otherwise the engine would lose admissions until it briefly went
// idle again (yo-yo dynamic).
//
// Design 3 (fallback tie-break on -bs): even with Designs 1+2, a
// cold-burst cascade leaves engines at `bs > 0` with empty windows
// (compensation is a no-op when `last_busy_at == None`, i.e. an
// engine that has *never* ticked). Their `tps_count` reads
// `Some(0)`, which under `max` selector's deterministic last-wins
// re-concentrates admissions to the last engine. Appending
// `-(o.bs as i64)` as a secondary score key keeps tps_count as the
// primary signal in steady state and routes the cold-burst "all
// Some(0)" window to whichever engine currently carries the fewest
// running requests — converting the cascade lock-in into round-
// robin fan-out. Has no effect once real tps_counts diverge.
//
// The KV$-aware-branch tie-break is `+tps_count, -bs` (same
// rationale), keeping the same load metric chain across both
// branches.

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
                    -(o.bs as i64),
                ),
            ),
            |t| select_max_by(
                t,
                |o| (
                    preble_tps_count(o),
                    -(o.bs as i64),
                ),
            ),
        )
    },
    after_extra: {
        preble_tps_compensate_idle(entry, &observations[chosen], all_sctx).await;
    }
}

// =========================================================================
// Policy-level regression tests
// =========================================================================

#[cfg(all(test, feature = "preble-bs-q"))]
mod bs_q_policy_tests {
    //! Locks Codex regression #2: the original 1094-burst on engine 13
    //! (ali-h20 `_1p` campaign) lived in the policy ordering + after_extra
    //! wiring, not just in the window primitive. These tests construct
    //! real Entry values and call `PrebleBsQ::schedule` to assert that
    //! the burst pattern does not return.

    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::{mpsc, Mutex};
    use tokio::time::Instant;
    use tracing::Span;

    use radixtree::BlockHash;

    use super::PrebleBsQ;
    use crate::gateway::validation::ValidGenerateRequest;
    use crate::scheduler::kvcache::{BlockHashState, PrefixBlockHash};
    use crate::scheduler::policies::Entry;
    use crate::scheduler::policies::policy_trait::Policy;
    use crate::scheduler::state::{LMetric, PREBLE_MATCH_RATIO_T, PREBLE_WINDOW_SECS};
    use crate::types::GenerationParams;
    use crate::ScheduleContext;

    fn make_entry(request_id: u64, input_tokens: Vec<u32>, block_size: usize) -> Entry {
        let (response_tx, _rx) = mpsc::unbounded_channel();
        let block_hash_state = BlockHashState::new(&input_tokens, block_size);
        let input_length = input_tokens.len() as u32;
        let request = ValidGenerateRequest {
            request_id,
            messages: vec![],
            inputs: String::new(),
            input_length,
            truncate: 0,
            decoder_input_details: false,
            params: GenerationParams {
                temperature: 1.0,
                repetition_penalty: 1.0,
                top_k: 0,
                top_p: 1.0,
                seed: 0,
                max_new_tokens: 1,
                stop_sequences: vec![],
                ignore_eos_token: false,
            },
            top_n_tokens: 0,
            input_tokens,
        };
        Entry {
            request,
            block_hash_state,
            response_tx,
            span: Span::none(),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
            generated_token_cnt: 0,
            prev_token_time: None,
            time_of_per_token: None,
            max_time_between_tokens: Duration::from_micros(0),
        }
    }

    fn make_pool(n: usize) -> Vec<Arc<Mutex<ScheduleContext>>> {
        (0..n)
            .map(|_| {
                Arc::new(Mutex::new(ScheduleContext {
                    lmetric: LMetric::default(),
                    block_hash: PrefixBlockHash::new(1024),
                }))
            })
            .collect()
    }

    fn max_consecutive_run(seq: &[usize]) -> usize {
        let mut max_run = 1usize;
        let mut run = 1usize;
        for w in seq.windows(2) {
            if w[0] == w[1] {
                run += 1;
                if run > max_run { max_run = run; }
            } else {
                run = 1;
            }
        }
        max_run
    }

    /// THE regression test for ali-h20 `_1p` 1094-burst. THRESHOLD=1.0
    /// masks the KV-aware branch entirely (filter never passes — the
    /// strict `> 1.0` cannot be saturated; see kvcache.rs:185 only
    /// hashes complete blocks). All admissions go through the
    /// load-balancing branch `select_min_by(t, preble_bs_sum)`. Each
    /// admission distinct (no shared prefix) so coincidental cache
    /// state is irrelevant.
    ///
    /// Expected behavior under the push-both design: max consecutive
    /// same-engine admissions is small (≤ 3, matching preble-q's
    /// observed baseline of 3). The pre-fix engine-step-only design
    /// produced runs up to 1094 on the same workload shape.
    #[tokio::test]
    async fn bs_q_no_burst_under_threshold_one_load_balancing_only() {
        // Set policy hyperparameters. `let _ = ...set(...)` because
        // OnceLock can only be set once per process; tests in the same
        // binary share state. We're the only setter under preble-bs-q.
        let _ = PREBLE_MATCH_RATIO_T.set(1.0);
        let _ = PREBLE_WINDOW_SECS.set(180);

        let pool = make_pool(16);
        let mut gctx = ();
        let mut chosen_seq = Vec::with_capacity(100);

        for i in 0..100u64 {
            // Disjoint token prefixes per request — no cache overlap,
            // so even with threshold < 1.0 the KV-aware branch would
            // not trigger here. With threshold = 1.0 it's masked
            // unconditionally.
            let base = (i as u32) * 10_000;
            let tokens: Vec<u32> = (0..64).map(|j| base + j as u32).collect();
            let entry = make_entry(i, tokens, 16);
            let chosen = PrebleBsQ::schedule(&entry, &pool, &mut gctx)
                .await
                .expect("scheduler must admit (lossless contract)");
            chosen_seq.push(chosen);
        }

        let max_run = max_consecutive_run(&chosen_seq);
        assert!(
            max_run <= 3,
            "BURST REGRESSION: max consecutive same-engine admissions = {max_run}, \
             expected ≤ 3 (preble-q baseline). The 1094-burst on engine 13 \
             from ali-h20 _1p has returned. Sequence: {chosen_seq:?}"
        );

        // Stronger: per-engine admission counts should be balanced.
        let mut counts = [0usize; 16];
        for &c in &chosen_seq {
            counts[c] += 1;
        }
        let min = *counts.iter().min().unwrap();
        let max = *counts.iter().max().unwrap();
        assert!(
            max - min <= 1,
            "fan-out should be near-perfect under load-balancing-only \
             (threshold=1.0, no cache overlap); got per-engine counts {counts:?}"
        );
    }
}
