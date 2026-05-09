// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// Derived from text-generation-inference by Hugging Face Inc. (Apache-2.0).

#[cfg(not(any(feature = "vllm-backend", feature = "zmq-backend")))]
compile_error!("You must enable either `vllm-backend` or `zmq-backend`!");

mod engine;
mod gateway;
mod scheduler;

pub mod error;
pub mod types;

#[allow(unused_imports)]
pub use engine::*;
#[allow(unused_imports)]
pub use scheduler::*;
// Re-export gateway items previously living at the crate root so external
// consumers (e.g., main.rs) continue to resolve them unchanged.
pub use gateway::{ChatRenderer, load_chat_template};
pub use gateway::server;
pub use gateway::model_config;
pub use gateway::chat_template;
// API DTOs are visible crate-internally for handler/validation/server code.
pub(crate) use gateway::api_types::*;
// `HubModelInfo`, `Info`, and `TokenizerRender` are part of the public
// crate-root surface (consumed by `main.rs` and any future embedders).
pub use gateway::api_types::{HubModelInfo, Info, TokenizerRender};
// Crate-root aliases for items addressed as `crate::Validation` /
// `crate::Entry` / `crate::Infer` from sibling modules. Kept here so the
// short paths in `gateway/server.rs` and `scheduler/infer.rs` resolve.
pub(crate) use gateway::validation::Validation;
pub(crate) use scheduler::Infer;
pub(crate) use scheduler::policies::Entry;
