//! Parser for `policy!` macro input.
//!
//! Surface form:
//!
//! ```ignore
//! policy! {
//!     name: RandomQ,
//!     gctx: (),
//!     body: { select_rand_by(&root_target(&observations), |_o| 1.0) },
//! }
//! ```
//!
//! Optional fourth field `after_extra: { ... }` appends extra statements
//! to the post-decision after-block. `chosen: usize` is in scope there.

use syn::parse::{Parse, ParseStream};
use syn::{Block, Expr, Ident, Result, Token, Type};

use crate::ast::PolicyInput;

impl Parse for PolicyInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut name: Option<Ident> = None;
        let mut gctx: Option<Type> = None;
        let mut body: Option<Expr> = None;
        let mut after_extra: Option<Block> = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let key_str = key.to_string();
            match key_str.as_str() {
                "name" => name = Some(input.parse()?),
                "gctx" => gctx = Some(input.parse()?),
                "body" => body = Some(input.parse()?),
                "after_extra" => after_extra = Some(input.parse()?),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown policy! field `{other}`; expected one of: \
                             name, gctx, body, after_extra"
                        ),
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(PolicyInput {
            name: name.ok_or_else(|| syn::Error::new(input.span(), "missing `name:`"))?,
            gctx: gctx.ok_or_else(|| syn::Error::new(input.span(), "missing `gctx:`"))?,
            body: body.ok_or_else(|| syn::Error::new(input.span(), "missing `body:`"))?,
            after_extra,
        })
    }
}

/// Entry point used by `lib.rs::policy`.
pub fn parse_policy(input: proc_macro2::TokenStream) -> Result<PolicyInput> {
    syn::parse2(input)
}
