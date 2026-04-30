//! AST → TokenStream lowering for `policy!`.
//!
//! Emits:
//!
//! ```ignore
//! pub(crate) struct {Name};
//! impl crate::policies::policy_trait::Policy for {Name} {
//!     type GlobalContext = {GctxTy};
//!     fn schedule<'a>(...) -> impl Future<...> + 'a {
//!         async move {
//!             use crate::policies::dsl_runtime::*;
//!             let req = &entry.request;
//!             let observations = capture_observations(entry, all_sctx).await;
//!             let chosen: Option<usize> = { #body };
//!             if let Some(idx) = chosen {
//!                 apply_default_after(entry, all_sctx, idx, &observations[idx]).await;
//!                 let chosen = idx;
//!                 #after_extra
//!             }
//!             chosen
//!         }
//!     }
//! }
//! ```

use proc_macro2::TokenStream;
use quote::quote;

use crate::ast::PolicyInput;

pub fn lower(policy: &PolicyInput) -> TokenStream {
    let name = &policy.name;
    let gctx = &policy.gctx;
    let body = &policy.body;

    let after_extra = policy
        .after_extra
        .as_ref()
        .map(|block| {
            let stmts = &block.stmts;
            quote! { #(#stmts)* }
        })
        .unwrap_or_default();

    quote! {
        pub(crate) struct #name;

        impl crate::policies::policy_trait::Policy for #name {
            type GlobalContext = #gctx;

            #[allow(unused_variables, clippy::let_and_return)]
            fn schedule<'a>(
                entry: &'a crate::policies::Entry,
                all_sctx: &'a [std::sync::Arc<tokio::sync::Mutex<crate::ScheduleContext>>],
                gctx: &'a mut Self::GlobalContext,
            ) -> impl std::future::Future<Output = Option<usize>> + Send + 'a {
                async move {
                    use crate::policies::dsl_runtime::*;
                    let req = &entry.request;
                    let observations = capture_observations(entry, all_sctx).await;
                    let chosen: Option<usize> = { #body };
                    if let Some(idx) = chosen {
                        apply_default_after(entry, all_sctx, idx, &observations[idx]).await;
                        #[allow(unused_variables)]
                        let chosen: usize = idx;
                        #after_extra
                    }
                    chosen
                }
            }
        }
    }
}
