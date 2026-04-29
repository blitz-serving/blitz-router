// AI-Dynamo scheduling policies.
//
// Dynamo is a PD-disaggregated system. Its single logit formula:
//
//   logit = overlap_weight * potential_prefill_block + decode_block
//
// is applied to ALL workers (both prefill and decode). The two terms
// carry different semantics depending on the worker type:
//
// ── For a PREFILL instance (no decode workload) ──────────────────────
//   prefill_token  = new_tokens + active_tokens   (prefill queue present)
//   decode_blocks  = ABSENT → fallback to floor(potential_prefill_block)
//   Result: logit ≈ (w + 1) * prefill_block  (both terms ≈ same value)
//
// ── For a DECODE instance (no prefill queue) ─────────────────────────
//   prefill_token  = new_tokens only              (active_tokens ≈ 0)
//   decode_blocks  = new_blocks + active_blocks   (real KV block data)
//   Result: logit = w * per_request_prefill_block + decode_block
//
// We provide both perspectives for ablation in PD-colocated:
//
// - T1 (prefill.rs): Prefill-instance perspective.
//     Uses full prefill queue (new + queued), decode_block = fallback.
//     Ignores decode overload → would route to a decode-heavy instance
//     if it has good cache.
//
// - T2 (decode.rs): Decode-instance perspective.
//     Uses per-request prefill only (no queue), decode_block = real.
//     Considers decode load → avoids overloading instances with many
//     active decode requests.
//
// Both preserve the same code structure:
//   logit = overlap_weight * potential_prefill_block + decode_block
// matching Dynamo's selector.rs:150 bijectively.

pub(crate) mod decode;
pub(crate) mod prefill;

// Default: T1 (prefill-instance perspective)
pub(crate) use prefill::DynamoPrefillQ as DynamoQ;
pub(crate) use decode::DynamoDecodeQ;
