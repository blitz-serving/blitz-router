//! The single trait every codegen-produced policy implements.
//!
//! See `docs/dsl-schema.md` §9. This file is the entire post-Phase-4
//! lowering target; it intentionally has zero dependencies on the
//! existing `QueuePlusPlus` / `AssignScore` / sampler scaffolding,
//! which is destined for retirement.

use std::future::Future;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::validation::ValidGenerateRequest;
use crate::ScheduleContext;

/// A scheduling policy. Picks one replica from `sctxs` for `req`,
/// possibly reading and updating `gctx`.
///
/// Returning `None` signals cluster-overload (no admissible replica);
/// the framework's lossless-admission contract guarantees this happens
/// only if every replica is in a state that disqualifies it (rare; in
/// the DSL surface this corresponds to a `Filter` whose fallback is
/// also empty, which the §10 static check is designed to rule out).
///
/// Implementations are emitted by the `policy-dsl` proc macro from a
/// DSL expression — hand-implementing this trait is not the intended
/// path.
pub trait Policy {
    type GlobalContext: Default + Send + Sync + 'static;

    fn schedule<'a>(
        req: &'a ValidGenerateRequest,
        sctxs: &'a [Arc<Mutex<ScheduleContext>>],
        gctx: &'a mut Self::GlobalContext,
    ) -> impl Future<Output = Option<usize>> + Send + 'a;
}
