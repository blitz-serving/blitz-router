//! `dynamo-q` and `dynamo-po-q` — Dynamo's PD-disaggregated routing logits
//! ported as ablation baselines in our PD-colocated environment.
//! See `dsl-schema.md` §8 and `.claude/memory/dynamo_pd_colocated_terms.md`.
//!
//! Dynamo upstream is PD-**disaggregated**: separate Prefill nodes and Decode
//! nodes, each running a different routing logit (`selector.rs:150`). We
//! port both formulas to our PD-**colocated** engines (where prefill and
//! decode share an instance) as two independent baselines.
//!
//! `dynamo-q`     — **Decode-node** formula. A Decode node sees no queued
//!                  prefill (those tokens were prefilled on a Prefill node),
//!                  so the per-request prefill term is `new_tokens(req, sctx)`
//!                  only. The decode term is the real
//!                  `new_blocks(req, sctx) + decode_blocks(sctx)`.
//!
//! `dynamo-po-q`  — **Prefill-node** formula ("po" = the routing logit a
//!                  prefill-only node would compute). A Prefill node sees no
//!                  active decode (decoding happens on a Decode node), so
//!                  the per-request prefill term is `prefill_tokens(req, sctx)`
//!                  = queued + new. The "decode" term degenerates to
//!                  `floor(prefill_blocks)` as a fallback. NOT to be read as
//!                  "uses new_tokens only" — that is `dynamo-q`.
//!
//! Both bijectively match Dynamo's `selector.rs:150` formula in their
//! respective node-role specialization.

use policy_dsl::policy;

const DYNAMO_OVERLAP_WEIGHT: f64 = 1.0;

// dynamo-q := Dynamo Decode-node logit.
// No queued prefill at a Decode node, so the per-request prefill term uses
// new_tokens only; the decode term is the real (new_blocks + decode_blocks).
policy! {
    name: DynamoQ,
    gctx: (),
    body: {
        select_min_by(&root_target(&observations), |o| {
            let block_size = o.block_size.max(1);
            let new_prefill_tokens = new_tokens(req, o);
            let potential_prefill_block = new_prefill_tokens as f64 / block_size as f64;
            let decode_block = (new_blocks(req, o) + decode_blocks(o)) as f64;
            let logit = DYNAMO_OVERLAP_WEIGHT * potential_prefill_block + decode_block;
            (logit * 1000.0) as u64
        })
    },
}

// dynamo-po-q := Dynamo Prefill-node logit.
// No active decode at a Prefill node, so the decode term degenerates to a
// floor(prefill_block) fallback; per-request prefill term is the full
// prefill_tokens = queued + new.
policy! {
    name: DynamoPoQ,
    gctx: (),
    body: {
        select_min_by(&root_target(&observations), |o| {
            let block_size = o.block_size.max(1);
            let prefill_token = prefill_tokens(req, o);
            let potential_prefill_block = prefill_token as f64 / block_size as f64;
            let decode_block = potential_prefill_block.floor();
            let logit = DYNAMO_OVERLAP_WEIGHT * potential_prefill_block + decode_block;
            (logit * 1000.0) as u64
        })
    },
}
