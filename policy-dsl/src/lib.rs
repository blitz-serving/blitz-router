//! Scheduling-policy DSL — proc-macro crate.
//!
//! See `docs/dsl/implementation.md` for the rewrite table (§2.1) and lint
//! allowlist (§2.2). For the DSL surface and field schema see
//! `docs/dsl/schema.md`; for canonical per-policy listings see
//! `docs/dsl/policies.md`.
//!
//! Exports a single `policy!` macro:
//!
//! ```ignore
//! policy! {
//!     name: RandomQ,
//!     gctx: (),
//!     body: { select_rand_by(&root_target(&observations), |_o| 1.0_f32) },
//! }
//! ```
//!
//! Optional `after_extra: { stmts; ... }` appends extra mutations after
//! the canonical `apply_default_after` call (with `chosen: usize` in scope).

use proc_macro::TokenStream;

mod ast;
mod check;
mod lower;
mod parse;

#[proc_macro]
pub fn policy(input: TokenStream) -> TokenStream {
    let parsed = match parse::parse_policy(input.into()) {
        Ok(ast) => ast,
        Err(err) => return err.to_compile_error().into(),
    };
    if let Err(err) = check::lint_policy(&parsed) {
        return err.to_compile_error().into();
    }
    lower::lower(&parsed).into()
}
