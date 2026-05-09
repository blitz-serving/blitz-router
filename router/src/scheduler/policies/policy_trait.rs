//! The single trait every codegen-produced policy implements.
//!
//! See `docs/dsl/implementation.md` §1.

use std::future::Future;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::Entry;
use crate::ScheduleContext;

/// A scheduling policy. Picks one replica from `all_sctx` for `entry`,
/// possibly reading and updating `gctx`.
///
/// Returning `None` signals cluster-overload (no admissible replica);
/// the framework's lossless-admission contract guarantees this happens
/// only if every replica is in a state that disqualifies it (rare; in
/// the DSL surface this corresponds to a `Filter` whose fallback is
/// also empty, which the `docs/dsl/schema.md` §8 static check is designed to rule out).
///
/// Implementations are emitted by the `policy-dsl` proc macro from a
/// DSL expression — hand-implementing this trait is not the intended
/// path.
pub trait Policy {
    type GlobalContext: Default + Send + Sync + 'static;

    fn schedule<'a>(
        entry: &'a Entry,
        all_sctx: &'a [Arc<Mutex<ScheduleContext>>],
        gctx: &'a mut Self::GlobalContext,
    ) -> impl Future<Output = Option<usize>> + Send + 'a;
}
