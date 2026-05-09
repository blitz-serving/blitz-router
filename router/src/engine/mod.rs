// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// BACK layer: engine clients and the colocation event loop that drives them.

pub(crate) mod client;
pub(crate) mod colocation;
pub(crate) mod vllm_http;
#[cfg(feature = "zmq-backend")]
pub(crate) mod zmq;

pub use client::{
    EngineClient, EngineClientError, EngineStepOutput, EngineStepReceiver, RequestStepOutput,
};
#[cfg(feature = "vllm-backend")]
pub use client::VllmEngineClient;
#[cfg(feature = "zmq-backend")]
pub use client::ZmqEngineClientAdapter;
pub(crate) use colocation::{start_vllm_colocation_event_loop, ColocationController, ExtExcept};
pub use vllm_http::VllmClient;
#[allow(unused_imports)]
pub(crate) use vllm_http::{VllmClientError, VllmMetric};
#[cfg(feature = "zmq-backend")]
pub use zmq::ZmqEngineClient;
