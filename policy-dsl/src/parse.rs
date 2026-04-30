//! DSL parser.
//!
//! Phase 2 status: skeleton. Accepts any input as an opaque token blob,
//! returns a placeholder AST. Real parsing lands in Phase 3 as each
//! migrated policy exercises a slice of the surface (random-q first,
//! aibrix-q the stress test).

use proc_macro2::TokenStream;
use syn::Result;

use crate::ast::{After, Expr, Policy, ScoreFn, SelectMode};

/// Parse a `policy! { ... }` invocation into a [`Policy`] AST node.
///
/// TODO(Phase 3): swap this stub for real syn-based parsing of the
/// grammar in `docs/dsl-schema.md` §2.
pub fn parse_policy(_input: TokenStream) -> Result<Policy> {
    // Placeholder so the crate compiles. Each policy migration in Phase 3
    // will incrementally add real parse paths and remove this fallback.
    Ok(Policy {
        name: syn::parse_quote!(StubQ),
        gctx_ty: None,
        body: Expr::Select {
            mode: SelectMode::Rand,
            score: ScoreFn(syn::parse_quote!(1)),
        },
        after: After { uses_default: true, extra_stmts: vec![] },
    })
}
