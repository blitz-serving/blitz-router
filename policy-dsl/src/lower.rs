//! AST → TokenStream lowering.
//!
//! Emits an `impl router::policies::Policy for <Name>` block. Phase 2
//! status: emits a stub body that compiles but always returns
//! `Some(0)` (route everything to replica 0). Each policy migration in
//! Phase 3 implements one branch of the AST visitor and the
//! corresponding TokenStream emission.

use proc_macro2::TokenStream;
use quote::quote;

use crate::ast::Policy;

pub fn lower(policy: &Policy) -> TokenStream {
    let name = &policy.name;
    let gctx_ty = policy
        .gctx_ty
        .as_ref()
        .map(|t| quote! { #t })
        .unwrap_or_else(|| quote! { () });

    // Stub body. Real lowering walks `policy.body` and emits the
    // partition / iter().min_by_key() / categorical-sample chain plus
    // the `after:` block. Lands in Phase 3 per policy.
    quote! {
        pub(crate) struct #name;

        impl crate::policies::policy_trait::Policy for #name {
            type GlobalContext = #gctx_ty;

            #[allow(unused_variables)]
            fn schedule<'a>(
                req: &'a crate::validation::ValidGenerateRequest,
                sctxs: &'a [std::sync::Arc<tokio::sync::Mutex<crate::ScheduleContext>>],
                gctx: &'a mut Self::GlobalContext,
            ) -> impl std::future::Future<Output = Option<usize>> + Send + 'a {
                async move {
                    // Phase 2 stub. Real codegen lands per policy in Phase 3.
                    if sctxs.is_empty() { None } else { Some(0) }
                }
            }
        }
    }
}
