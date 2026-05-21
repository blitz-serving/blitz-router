//! Reusable generic data structures composed by the router and policy
//! layer. Currently:
//!
//! - [`SlidingWindow`] — time-windowed aggregate over a
//!   `VecDeque<(Instant, T)>` with an [`Aggregate`] trait for
//!   incremental maintenance.

mod sliding_window;

pub use sliding_window::{Aggregate, SlidingWindow, Sum};
