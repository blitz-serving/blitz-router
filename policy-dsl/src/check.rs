//! Static checks on the parsed AST.
//!
//! Currently enforces the §10 `after:` well-formedness rule: the body
//! must either begin with the `default` keyword (and may extend it with
//! additional statements) OR enumerate all six canonical mutations
//! syntactically. Phase 2 ships the option-1 (default-shorthand) check;
//! the option-2 (full enumeration) check lands when a policy first
//! exercises it.

use syn::Result;

use crate::ast::Policy;

/// Reject any policy whose `after:` body is malformed per §10.
pub fn after_clause_well_formed(policy: &Policy) -> Result<()> {
    if policy.after.uses_default {
        return Ok(());
    }
    // Option 2: full enumeration check. Lands when needed.
    Err(syn::Error::new(
        policy.name.span(),
        "after: clause must reference `default` or syntactically enumerate \
         the six canonical mutations (sctx.lmetric.bs += 1, waiting_reqs += 1, \
         prefill_tokens += new_tokens(...), all_tokens += req.tokens, \
         entry.pred_hits := hit_blocks(...), entry.epoch := sctx.block_hash.epoch). \
         See docs/dsl-schema.md §10.",
    ))
}
