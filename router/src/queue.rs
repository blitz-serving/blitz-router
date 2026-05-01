// Request queue module — thin re-export shim.
//
// Originally housed both the policy-based scheduling types and a legacy
// TGI `Queue` for the `blitzllm-backend` code path. The legacy backend
// has been removed; this file is kept only because `infer.rs` and other
// callers import via `crate::queue::*`.

pub(crate) use crate::policies::{Entry, QueuePro, TaskAssigner};
