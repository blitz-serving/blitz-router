//! `polyserve-q` — choose the replica with the largest predicted TPOT
//! among those that satisfy both TTFT and TPOT SLOs.
//!
//! The current implementation uses simulator `RolloutGist.in_decode_tbt_ms`
//! as a TPOT approximation. If no replica satisfies both SLOs, the policy
//! falls back to uniformly random assignment.

use std::future::Future;
use std::sync::Arc;

use rand::Rng;
use radixtree::BlockHash;
use tokio::sync::Mutex;

use crate::scheduler::state::{
    LMetricInc, POLYSERVE_TPOT_SLO_MS, POLYSERVE_TTFT_SLO_MS, ScheduleContext,
};

use super::policy_trait::Policy;
use super::Entry;

#[derive(Default, Debug)]
pub(crate) struct PolyserveQ;

impl Policy for PolyserveQ {
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

            let ttft_slo_ms = POLYSERVE_TTFT_SLO_MS.get().copied().unwrap_or(5000.0);
            let tpot_slo_ms = POLYSERVE_TPOT_SLO_MS.get().copied().unwrap_or(40.0);

            let candidate_id = entry.request.request_id;
            let input_length = entry.request.input_length;
            let hashes = entry.block_hash_state.get_hashes();

            let mut best_slo_ok: Option<(usize, f32)> = None;

            for (idx, sctx) in all_sctx.iter().enumerate() {
                let sctx_hits = {
                    let g = sctx.lock().await;
                    g.block_hash.get(hashes)
                };

                let gist = crate::scheduler::simulator::query(
                    idx,
                    candidate_id,
                    input_length,
                    hashes,
                    sctx_hits,
                );

                let ttft_ms = gist.and_then(|g| g.ttft_ms);
                let tpot_ms = gist.and_then(|g| g.in_decode_tbt_ms);
                let slo_ok = matches!(
                    (ttft_ms, tpot_ms),
                    (Some(ttft), Some(tpot)) if ttft <= ttft_slo_ms && tpot <= tpot_slo_ms
                );
                tracing::info!(
                    target: "policy.polyserve-q",
                    request_id = candidate_id,
                    replica = idx,
                    input_length,
                    sctx_prefix_hits = sctx_hits,
                    ttft_ms = ?ttft_ms,
                    tpot_ms = ?tpot_ms,
                    ttft_slo_ms,
                    tpot_slo_ms,
                    slo_ok,
                    "simulator query result"
                );

                let (Some(_ttft_ms), Some(tpot_ms)) = (ttft_ms, tpot_ms) else {
                    continue;
                };

                if slo_ok {
                    if best_slo_ok.map_or(true, |(_, prev_tpot)| tpot_ms > prev_tpot) {
                        best_slo_ok = Some((idx, tpot_ms));
                    }
                }
            }

            let chosen = if let Some((idx, _)) = best_slo_ok {
                idx
            } else {
                let idx = rand::thread_rng().gen_range(0..all_sctx.len());
                tracing::warn!(
                    target: "policy.polyserve-q",
                    replicas = all_sctx.len(),
                    ttft_slo_ms,
                    tpot_slo_ms,
                    chosen = idx,
                    "no replica satisfied PolyServe SLOs; falling back to random replica"
                );
                idx
            };

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
