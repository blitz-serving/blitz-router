//! Allowlist lint for the `policy!` body and after-extra.
//!
//! See `docs/dsl-schema.md` §13.2. The lint enforces that the macro body
//! is built only from the closed allowlist of helper calls + ordinary
//! Rust expressions (closures, lets, arith, field access). This is what
//! makes the impl → DSL reverse direction trustworthy: any legal body
//! mechanically maps back to a paper-form DSL expression via §13.1.

use std::collections::HashSet;

use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Expr, ExprCall, ExprForLoop, ExprLoop, ExprMatch, ExprUnsafe, ExprWhile};

use crate::ast::PolicyInput;

/// Allowed function names. Any call to a function not in this set is rejected.
const ALLOWED_FNS: &[&str] = &[
    // Combinators (rewrite table rows 1–4)
    "filter_then",
    "select_min_by",
    "select_max_by",
    "select_rand_by",
    // Reducers (rewrite table rows 6–10)
    "mean_of_usize",
    "std_of_usize",
    "sum_of_usize",
    "min_of_usize",
    "max_of_usize",
    "mean_of_f32",
    "std_of_f32",
    "min_of_f32",
    "max_of_f32",
    // Named pure fns (docs/dsl-schema.md §5)
    "new_tokens",
    "new_blocks",
    "queued_tokens",
    "prefill_tokens",
    "hit_blocks",
    "match_blocks",
    "hit_pct",
    "decode_blocks",
    "preble_cost",
    "preble_update_after",
    // Helpers
    "root_target",
    "weighted_pick",
];

pub fn lint_policy(policy: &PolicyInput) -> Result<(), syn::Error> {
    let allowlist: HashSet<&str> = ALLOWED_FNS.iter().copied().collect();
    let mut linter = Linter { allowlist: &allowlist, errors: vec![] };

    linter.visit_expr(&policy.body);
    if let Some(ref after) = policy.after_extra {
        // The after-extra is allowed slightly more (e.g., gctx mutation methods).
        // The allowlist is the same for *function* calls; arbitrary method calls
        // on `gctx` or its fields are permitted (they go through the type system).
        linter.visit_block(after);
    }

    if linter.errors.is_empty() {
        Ok(())
    } else {
        let mut iter = linter.errors.into_iter();
        let mut combined = iter.next().unwrap();
        for e in iter {
            combined.combine(e);
        }
        Err(combined)
    }
}

struct Linter<'a> {
    allowlist: &'a HashSet<&'a str>,
    errors: Vec<syn::Error>,
}

impl<'ast, 'a> Visit<'ast> for Linter<'a> {
    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if let Expr::Path(p) = &*call.func {
            // Reject only single-segment unqualified calls (`foo(...)`). Qualified
            // calls like `crate::policies::dsl_runtime::foo(...)` are also checked
            // by their last segment, since the `use` at function top brings them
            // into scope.
            if let Some(seg) = p.path.segments.last() {
                let name = seg.ident.to_string();
                if !self.allowlist.contains(name.as_str()) {
                    self.errors.push(syn::Error::new(
                        seg.ident.span(),
                        format!(
                            "function `{name}` is not in the policy DSL allowlist. \
                             Allowed: {} (see docs/dsl-schema.md §13.2)",
                            ALLOWED_FNS.join(", ")
                        ),
                    ));
                }
            }
        } else {
            self.errors.push(syn::Error::new(
                call.func.span(),
                "policy bodies may only call named functions from the allowlist; \
                 indirect calls are not permitted (docs/dsl-schema.md §13.2)",
            ));
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_for_loop(&mut self, n: &'ast ExprForLoop) {
        self.errors.push(syn::Error::new(
            n.for_token.span(),
            "`for` loops are not allowed in policy bodies; \
             iteration belongs in dsl_runtime helpers, not in policies",
        ));
        syn::visit::visit_expr_for_loop(self, n);
    }

    fn visit_expr_while(&mut self, n: &'ast ExprWhile) {
        self.errors.push(syn::Error::new(
            n.while_token.span(),
            "`while` loops are not allowed in policy bodies",
        ));
        syn::visit::visit_expr_while(self, n);
    }

    fn visit_expr_loop(&mut self, n: &'ast ExprLoop) {
        self.errors.push(syn::Error::new(
            n.loop_token.span(),
            "`loop` constructs are not allowed in policy bodies",
        ));
        syn::visit::visit_expr_loop(self, n);
    }

    fn visit_expr_unsafe(&mut self, n: &'ast ExprUnsafe) {
        self.errors.push(syn::Error::new(
            n.unsafe_token.span(),
            "`unsafe` blocks are not allowed in policy bodies",
        ));
        syn::visit::visit_expr_unsafe(self, n);
    }

    fn visit_expr_match(&mut self, n: &'ast ExprMatch) {
        self.errors.push(syn::Error::new(
            n.match_token.span(),
            "`match` is not allowed in policy bodies; use `if`/`else` chains \
             or push the dispatch into the score function",
        ));
        syn::visit::visit_expr_match(self, n);
    }

    // Macros, return/break/continue: rare in expression position, easy to add
    // if we see them in the wild. Method calls and field accesses are deliberately
    // not restricted — the type system constrains them via Observation/Entry/etc.
}

// Override the (unused) default check stub with the real linter.
#[allow(dead_code)]
pub fn after_clause_well_formed(_p: &PolicyInput) -> Result<(), syn::Error> {
    Ok(())
}
