//! AST types for the policy DSL.
//!
//! Mirrors the BNF in `docs/dsl-schema.md` §2. Phase 2 ships only the
//! shapes; per-variant parsing/lowering lands in Phase 3 alongside each
//! policy migration.

use syn::Ident;

/// Top-level policy declaration: name, optional GlobalContext type, body, after clause.
#[allow(dead_code)] // body / after_clause read by lower.rs in Phase 3
pub struct Policy {
    pub name: Ident,
    pub gctx_ty: Option<syn::Type>,
    pub body: Expr,
    pub after: After,
}

/// Core DSL expression form.
#[allow(dead_code)] // variants land in Phase 3 as parsing/lowering is implemented
pub enum Expr {
    /// `Filter <pred> <expr_in> <expr_out>` — lossless three-arg.
    Filter {
        pred: Pred,
        on_pass: Box<Expr>,
        on_fail: Box<Expr>,
    },
    /// `Select min|max|rand by <fn>`.
    Select { mode: SelectMode, score: ScoreFn },
    /// `With <name> = <reducer> [, <name> = <reducer>]* in <expr>`.
    With { bindings: Vec<(Ident, ReducerExpr)>, body: Box<Expr> },
}

#[allow(dead_code)]
pub enum SelectMode {
    Min,
    Max,
    Rand,
}

/// `<pred>` — boolean expression over sctx.f, gctx.f, req.f, named-fn calls,
/// bound τ̄. Body parsed as a `syn::Expr` and constrained at lowering time
/// to the algebra in §7.
#[allow(dead_code)]
pub struct Pred(pub syn::Expr);

/// `<fn>` — score-producing expression with the same algebra constraints
/// as `<pred>`, but typed in whatever the surrounding `Select` consumes
/// (numeric for min/max, weight for rand, tuples for lex-cmp).
#[allow(dead_code)]
pub struct ScoreFn(pub syn::Expr);

/// `<reducer>` — pure aggregation over `[ScheduleContext]`.
#[allow(dead_code)]
pub enum ReducerExpr {
    Mean(Ident),  // .field
    Std(Ident),
    Sum(Ident),
    Min(Ident),
    Max(Ident),
    Count,
    /// Arithmetic combinations of reducers and constants (`Mean .bs + 1.0 · Std .bs`).
    Arith(syn::Expr),
}

/// `after:` clause body. The §10 static check enforces that this either
/// starts with `default` (option 1) or syntactically enumerates the six
/// canonical mutations (option 2).
#[allow(dead_code)] // extra_stmts emitted by lower.rs in Phase 3
pub struct After {
    pub uses_default: bool,
    pub extra_stmts: Vec<syn::Stmt>,
}
