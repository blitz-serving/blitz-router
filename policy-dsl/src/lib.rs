//! Scheduling-policy DSL — proc-macro crate.
//!
//! See `docs/dsl-schema.md` for the full specification, especially §13
//! (rewrite table + lint).
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
