//! `least-ttft-q` — replica selection by simulator-projected TTFT.
//!
//! For each candidate request, the policy queries the latency simulator
//! for every replica and picks the one whose `RolloutGist.ttft_ms` is
//! minimum. This is the first prediction-based policy that consumes
//! `simulator::query` (the rest of the policies in this codebase score
//! replicas from `ScheduleContext` data only).
//!
//! Hand-written `impl Policy` — the `policy!` proc macro can't express
//! a per-replica predictor call, only DSL-shaped scalar scoring over
//! the existing `Observation` helpers.
//!
//! Build requirement: `--features simulator,least-ttft-q`. The policy
//! is also compilable without `simulator`, but in that mode `query`
//! returns `None` and the fallback (replica 0) kicks in for every
//! request — which is a degenerate routing strategy. The router CLI
//! still needs `--enable-simulator` at runtime so the predictor grids
//! actually load.

use std::future::Future;
use std::sync::Arc;
use tokio::sync::Mutex;

use radixtree::BlockHash;

use crate::scheduler::state::{LMetricInc, ScheduleContext};

use super::policy_trait::Policy;
use super::Entry;

#[derive(Default, Debug)]
pub(crate) struct LeastTtftQ;

impl Policy for LeastTtftQ {
    type GlobalContext = ();

    fn schedule<'a>(
        entry: &'a Entry,
        all_sctx: &'a [Arc<Mutex<ScheduleContext>>],
        _gctx: &'a mut Self::GlobalContext,
    ) -> impl Future<Output = Option<usize>> + Send + 'a {
        async move {
            if all_sctx.is_empty() {
                return None;
            }

            let candidate_id = entry.request.request_id;
            let input_length = entry.request.input_length;
            let hashes = entry.block_hash_state.get_hashes();

            // Per-replica: snapshot SCtx prefix-hit count, then query the
            // simulator. The simulator's A2 max-merge composes this with
            // its own L1 mirror's view of in-flight (pre-SSE) hashes.
            let mut best: Option<(usize, f32)> = None;
            let mut any_gist = false;
            for (idx, sctx) in all_sctx.iter().enumerate() {
                let sctx_hits = {
                    let g = sctx.lock().await;
                    g.block_hash.get(hashes)
                };

                #[cfg(feature = "simulator")]
                let ttft = crate::scheduler::simulator::query(
                    idx,
                    candidate_id,
                    input_length,
                    hashes,
                    sctx_hits,
                )
                .and_then(|g| g.ttft_ms);

                #[cfg(not(feature = "simulator"))]
                let ttft: Option<f32> = None;

                if let Some(t) = ttft {
                    any_gist = true;
                    if best.map_or(true, |(_, prev)| t < prev) {
                        best = Some((idx, t));
                    }
                }
            }

            // Fallback: if no replica returned a usable gist (cold cache or
            // simulator inactive), pick replica 0 so the request still
            // routes — the framework's lossless-admission contract.
            let chosen = if let Some((idx, _)) = best {
                idx
            } else {
                if !any_gist {
                    tracing::warn!(
                        target: "policy.least-ttft-q",
                        replicas = all_sctx.len(),
                        "no replica returned a gist; falling back to replica 0"
                    );
                }
                0
            };

            // Apply the chosen-replica bookkeeping that DSL policies get
            // for free via `apply_default_after`:
            //   1. set_pred_block_hits(hit_nblks) — clears NONE_SENTINEL
            //      so the engine completion path's
            //      `set_real_token_hits_get_diff` assertion passes.
            //   2. set_decision_epoch(current_epoch) — for CORRECTION log.
            //   3. lmetric += LMetricInc — so subsequent policy calls and
            //      simulator queries see the admission.
            {
                let mut g = all_sctx[chosen].lock().await;
                let current_epoch = g.block_hash.epoch();
                let hit_nblks = g.block_hash.get(hashes);
                entry.block_hash_state.set_pred_block_hits(hit_nblks);
                entry.block_hash_state.set_decision_epoch(current_epoch);
                let new_ntkns = entry.request.input_tokens.len()
                    .saturating_sub(hit_nblks * entry.block_hash_state.get_block_size());
                g.lmetric += LMetricInc {
                    bs_inc: 1,
                    waiting_reqs_inc: 1,
                    prefill_tokens_inc: new_ntkns,
                    all_tokens_inc: entry.request.input_tokens.len(),
                };
            }

            Some(chosen)
        }
    }
}
