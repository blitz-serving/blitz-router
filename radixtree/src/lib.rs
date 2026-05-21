//! Prefix-matching data structures used by the lmetric router.
//!
//! - [`BlockHash`] — interface for the KV-cache prefix matcher.
//! - [`RadixTreeBlockHash`] — production radix-tree impl (V=`Bids`, concurrent,
//!   epoch-tracked). The concurrent + multi-bid hardening on top of a
//!   compressed Patricia trie; lowered from the Verus-verified L0 spec
//!   in [`verified`] (gated by the `verify` feature) through the bench
//!   ladder under `benches/lowering_levels.rs`.
//! - [`RadixTreeReqIdHash`] — simulator-mirror specialization (V=`ReqId`,
//!   simple Box-based, single-threaded — different requirements,
//!   different impl, intentionally not unified).
//! - [`types`] — generic primitives the router/policy layer composes
//!   ([`SlidingWindow`] etc.).

mod core;
mod block_hash;
mod req_id_hash;
pub mod types;

#[cfg(feature = "verify")]
pub mod verified;

pub use block_hash::{Bids, BlockHash, RadixTreeBlockHash};
pub use req_id_hash::RadixTreeReqIdHash;
pub use types::{Aggregate, Count, SlidingWindow, Sum};
