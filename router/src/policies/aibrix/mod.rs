// AIBrix scheduling policies.
//
// AIBrix offers multiple routing algorithms. This module implements
// those applicable to PD-colocated KV-cache-aware routing.

pub(crate) mod prefix_cache;

pub(crate) use prefix_cache::AibrixQ;
