//! Generic Patricia trie primitives parameterized over `<K, V>`.
//!
//! These are the L3 lowering of the verified L0 spec (see [`super::verified`])
//! after the bijective steps documented in `benches/lowering_levels.rs`:
//!
//! - L0 → L1: linear child scan → sorted `Vec` + binary search.
//! - L1 → L2: `Vec<Box<Node>>` → `Vec<*mut Node>` (raw pointers).
//! - L2 → L3: degree-aware `Children::{Small, Large}` split (HashMap
//!   upgrade at [`SMALL_MAX`]) — the form retained here.
//!
//! This module exposes only the data layout. The unsafe traversal /
//! mutation logic lives in the specialization that owns the tree
//! (e.g. [`super::block_hash::RadixTreeBlockHash`]).

use nohash_hasher::{self, BuildNoHashHasher};
use std::collections::HashMap;

/// Threshold at which `Children::Small` (sorted `Vec`) upgrades to
/// `Children::Large` (`HashMap`). Tuned for KV-cache workloads where the
/// average node fan-out stays well under 32 but pathological cases can spike.
pub(crate) const SMALL_MAX: usize = 32;

pub(crate) enum Children<K: nohash_hasher::IsEnabled, V, S = BuildNoHashHasher<K>> {
    Small(Vec<*mut Node<K, V>>),
    Large(HashMap<K, *mut Node<K, V>, S>), // Large keyed only by first element of label
}

impl<K: nohash_hasher::IsEnabled, V, S> Children<K, V, S> {
    /// Only clears the container; DOES NOT drop pointed nodes.
    pub(crate) fn clear(&mut self) {
        match self {
            Children::Large(m) => {
                m.clear();
            }
            Children::Small(v) => {
                v.clear();
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Children::Large(m) => m.is_empty(),
            Children::Small(v) => v.is_empty(),
        }
    }
}

impl<K: nohash_hasher::IsEnabled, V, S> Default for Children<K, V, S> {
    fn default() -> Self {
        Children::Small(Vec::new())
    }
}

pub(crate) struct Node<K: nohash_hasher::IsEnabled, V> {
    pub(crate) children: Children<K, V>,
    pub(crate) value: Vec<K>,   // Hash values of this kvcache block segment
    pub(crate) payload: Vec<V>, // Block ids corresponding to hash values
}

impl<K: nohash_hasher::IsEnabled, V> Node<K, V> {
    pub(crate) fn is_empty(&self) -> bool {
        let child_is_empty = match &self.children {
            Children::Small(v) => v.is_empty(),
            Children::Large(m) => m.is_empty(),
        };
        child_is_empty && self.value.is_empty() && self.payload.is_empty()
    }

    pub(crate) fn new() -> *mut Node<K, V> {
        let boxed = Box::new(Node {
            children: Children::Small(Vec::new()),
            value: Vec::default(),
            payload: Vec::default(),
        });
        Box::into_raw(boxed)
    }
}

pub(crate) enum CommonPrefixInner<K: nohash_hasher::IsEnabled, V> {
    NoMatch(*mut Node<K, V>),
    PartialMatch(*mut Node<K, V>, usize),
    FullMatch(*mut Node<K, V>),
}
