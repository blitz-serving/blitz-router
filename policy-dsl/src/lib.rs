//! Scheduling-policy DSL — proc-macro crate.
//!
//! See `docs/dsl-schema.md` for the full specification. This crate exports
//! a single proc macro, `policy!`, that takes a DSL expression and emits an
//! `impl Policy for <Name> { fn schedule(...) { ... } }` block plus the
//! associated `GlobalContext` type when needed.
//!
//! Architecture:
//!
//! ```text
//!   #[policy(name = "..."-q, gctx = ...)]
//!   policy! {
//!       <DSL expression>
//!       after: <after-clause>
//!   }
//!         │
//!         ▼  parse.rs
//!   AST  ─── ast.rs
//!         │
//!         ▼  check.rs (after-clause static check, §10)
//!   AST'
//!         │
//!         ▼  lower.rs
//!   TokenStream emitting `impl Policy for ...`
//! ```
//!
//! Phase 2 status: skeleton only. The macro accepts any input and emits an
//! empty-body stub. Per-construct parsing and lowering land in Phase 3 as
//! each policy migration exercises a slice of the surface.

use proc_macro::TokenStream;

mod ast;
mod check;
mod lower;
mod parse;

/// `policy! { ... }` — primary DSL entry point.
///
/// Takes a DSL expression (see `docs/dsl-schema.md` §2) and emits an
/// `impl Policy for <Name>Q { ... }` block. Skeleton implementation
/// for now: parses to a stub AST and emits a placeholder.
#[proc_macro]
pub fn policy(input: TokenStream) -> TokenStream {
    let parsed = match parse::parse_policy(input.into()) {
        Ok(ast) => ast,
        Err(err) => return err.to_compile_error().into(),
    };
    if let Err(err) = check::after_clause_well_formed(&parsed) {
        return err.to_compile_error().into();
    }
    lower::lower(&parsed).into()
}
