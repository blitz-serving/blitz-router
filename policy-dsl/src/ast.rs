//! AST types for the `policy!` macro input.

use syn::{Block, Expr, Ident, Type};

/// Input to `policy! { name: ..., gctx: ..., body: ..., after_extra: ... }`.
pub(crate) struct PolicyInput {
    pub name: Ident,
    pub gctx: Type,
    pub body: Expr,
    /// Extra statements appended after `apply_default_after`. Within these,
    /// `chosen: usize` is in scope (the selected replica index). The lint
    /// allows mutations to `gctx` and method calls.
    pub after_extra: Option<Block>,
}
