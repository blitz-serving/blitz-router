// Sibling of `block_hash::RadixTreeBlockHash`,
// purpose-built for the simulator's L1 incremental mirror.
//
// Why a fresh structure (rather than reusing `RadixTreeBlockHash`):
//
//   * `RadixTreeBlockHash` encodes the engine's exact KV-cache (V=Bids =
//     `SmallVec<[u64;1]>`, indexed by block id, must respond to
//     `evicted_block_ids`). Mirror has different responsibilities:
//     it tracks per-request prefix membership (V=ReqId), arbitrates
//     evictions by checking against the in-flight set (Group B
//     Point 2), and tolerates drift via the redo path (A2).
//   * Decoupling lets each evolve independently. The block-id
//     reverse-lookup table that `RadixTreeBlockHash` carries
//     (`block_to_node` / `block_to_hash`) is unnecessary here.
//
// Operations:
//
//   * `insert_hashes(hashes, rid)` — register that `rid` claims this
//     prefix sequence (admit time).
//   * `remove_by_hashes(hashes, rid)` — drop `rid`'s claim and prune
//     subtrees that no other rid claims (finish/abort/preempt time).
//   * `get(hashes) -> usize` — longest matched prefix length.
//   * `owners_of(hash) -> impl Iterator<u64>` — all rids that include
//     `hash` in their sequence (evict path).
//
// Implementation choice: a simple Rust-idiomatic trie with one hash
// per edge. Mirror cardinality is ~64 in-flight × ~100 hashes each =
// 6400 nodes — deep optimisation (radix path-compression, the unsafe
// `Node<K,V>` style of `block_hash.rs`) buys nothing here.

use std::collections::{HashMap, HashSet};

#[derive(Default)]
struct TrieNode {
    children: HashMap<u64, TrieNode>,
    /// Rids whose insert sequence passes through this node (i.e.
    /// every node on the path from root to a sequence's leaf has the
    /// rid in its `owners`). Empty `owners` + empty `children` → node
    /// can be pruned.
    owners: HashSet<u64>,
}

#[derive(Default)]
pub struct RadixTreeReqIdHash {
    root: TrieNode,
    /// Auxiliary global index: hash → set of rids whose sequence
    /// contains this hash. Lets `owners_of(h)` answer in O(1) without
    /// walking the trie. Maintained eagerly alongside the trie.
    hash_owners: HashMap<u64, HashSet<u64>>,
    epoch: u64,
}

impl RadixTreeReqIdHash {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `rid`'s claim over the prefix sequence `hashes`.
    /// Idempotent: re-inserting the same `(rid, hashes)` is a no-op
    /// at the set level (HashSet::insert returns false).
    pub fn insert_hashes(&mut self, hashes: &[u64], rid: u64) {
        if hashes.is_empty() {
            return;
        }
        let mut node = &mut self.root;
        for &h in hashes {
            node = node.children.entry(h).or_default();
            node.owners.insert(rid);
        }
        for &h in hashes {
            self.hash_owners.entry(h).or_default().insert(rid);
        }
        self.epoch += 1;
    }

    /// Drop `rid`'s claim from this exact prefix sequence and prune
    /// nodes that nobody else claims. Safe to call with a sequence
    /// `rid` never inserted (no-op).
    pub fn remove_by_hashes(&mut self, hashes: &[u64], rid: u64) {
        if hashes.is_empty() {
            return;
        }
        Self::strip_owner_along_path(&mut self.root, hashes, rid);
        Self::prune_empty(&mut self.root, hashes);
        for &h in hashes {
            if let Some(set) = self.hash_owners.get_mut(&h) {
                set.remove(&rid);
                if set.is_empty() {
                    self.hash_owners.remove(&h);
                }
            }
        }
        self.epoch += 1;
    }

    /// Longest prefix length of `hashes` that exists in the trie.
    pub fn get(&self, hashes: &[u64]) -> usize {
        let mut node = &self.root;
        let mut len = 0usize;
        for &h in hashes {
            match node.children.get(&h) {
                Some(child) => {
                    node = child;
                    len += 1;
                }
                None => break,
            }
        }
        len
    }

    /// All rids that have `hash` in their sequence. Used by the evict
    /// path to decide whether an evicted hash can be safely removed
    /// (no in-flight owner) or must be kept (owner still in flight).
    pub fn owners_of(&self, hash: u64) -> impl Iterator<Item = u64> + '_ {
        self.hash_owners.get(&hash).into_iter().flat_map(|s| s.iter().copied())
    }

    /// Forcibly drop `hash` from the trie regardless of which rids
    /// claim it. Used by the evict path after the caller has
    /// confirmed no in-flight owner remains. Sub-tree under `hash`
    /// is dropped wholesale (since the hash itself is gone, anything
    /// that depended on a path through it is also gone).
    pub fn evict_orphan_hash(&mut self, hash: u64) {
        if self.root.children.remove(&hash).is_some() {
            // Anything in `hash_owners` that referenced rids whose
            // sequence began with this hash is conservatively cleaned
            // by the next remove_by_hashes / insert_hashes from those
            // rids. We don't proactively walk the dropped subtree
            // here; orphan eviction is rare per Group B Point 1.
            self.hash_owners.remove(&hash);
            self.epoch += 1;
        }
    }

    /// Monotonic epoch counter, useful as a cache-key for downstream
    /// memoisation. Bumped on every mutation.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Observability only: number of distinct hashes currently in the
    /// global index.
    pub fn distinct_hashes(&self) -> usize {
        self.hash_owners.len()
    }

    // ---------- helpers ----------

    fn strip_owner_along_path(root: &mut TrieNode, hashes: &[u64], rid: u64) {
        let mut node = root;
        for &h in hashes {
            match node.children.get_mut(&h) {
                Some(child) => {
                    child.owners.remove(&rid);
                    node = child;
                }
                None => return,
            }
        }
    }

    fn prune_empty(node: &mut TrieNode, hashes: &[u64]) {
        if hashes.is_empty() {
            return;
        }
        let h = hashes[0];
        let mut should_remove = false;
        if let Some(child) = node.children.get_mut(&h) {
            Self::prune_empty(child, &hashes[1..]);
            if child.owners.is_empty() && child.children.is_empty() {
                should_remove = true;
            }
        }
        if should_remove {
            node.children.remove(&h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn owners_set(t: &RadixTreeReqIdHash, hash: u64) -> HashSet<u64> {
        t.owners_of(hash).collect()
    }

    #[test]
    fn insert_and_get_basic() {
        let mut t = RadixTreeReqIdHash::new();
        t.insert_hashes(&[10, 20, 30], 1);
        assert_eq!(t.get(&[10, 20, 30]), 3);
        assert_eq!(t.get(&[10, 20]), 2);
        assert_eq!(t.get(&[10, 20, 99]), 2);
        assert_eq!(t.get(&[40]), 0);
    }

    #[test]
    fn shared_prefix_two_rids() {
        let mut t = RadixTreeReqIdHash::new();
        t.insert_hashes(&[10, 20, 30], 1);
        t.insert_hashes(&[10, 20, 40], 2);
        assert_eq!(t.get(&[10, 20]), 2);
        assert_eq!(t.get(&[10, 20, 30]), 3);
        assert_eq!(t.get(&[10, 20, 40]), 3);
        assert_eq!(owners_set(&t, 10), [1, 2].iter().copied().collect());
        assert_eq!(owners_set(&t, 20), [1, 2].iter().copied().collect());
        assert_eq!(owners_set(&t, 30), [1].iter().copied().collect());
        assert_eq!(owners_set(&t, 40), [2].iter().copied().collect());
    }

    #[test]
    fn remove_one_rid_keeps_shared_prefix() {
        let mut t = RadixTreeReqIdHash::new();
        t.insert_hashes(&[10, 20, 30], 1);
        t.insert_hashes(&[10, 20, 40], 2);
        t.remove_by_hashes(&[10, 20, 30], 1);
        // Prefix 10,20 still served by rid 2.
        assert_eq!(t.get(&[10, 20]), 2);
        // Branch to 30 pruned.
        assert_eq!(t.get(&[10, 20, 30]), 2);
        // Branch to 40 still there.
        assert_eq!(t.get(&[10, 20, 40]), 3);
        assert_eq!(owners_set(&t, 30), HashSet::new());
        assert_eq!(owners_set(&t, 40), [2].iter().copied().collect());
    }

    #[test]
    fn remove_last_owner_drops_path_completely() {
        let mut t = RadixTreeReqIdHash::new();
        t.insert_hashes(&[10, 20, 30], 1);
        t.remove_by_hashes(&[10, 20, 30], 1);
        assert_eq!(t.get(&[10]), 0);
        assert_eq!(t.distinct_hashes(), 0);
    }

    #[test]
    fn remove_unknown_rid_is_noop() {
        let mut t = RadixTreeReqIdHash::new();
        t.insert_hashes(&[10, 20], 1);
        let epoch_before = t.epoch();
        t.remove_by_hashes(&[10, 20], 99);
        // epoch still bumps (we count every call), but tree unchanged.
        assert_eq!(t.get(&[10, 20]), 2);
        assert!(t.epoch() > epoch_before);
    }

    #[test]
    fn duplicate_insert_idempotent() {
        let mut t = RadixTreeReqIdHash::new();
        t.insert_hashes(&[10, 20], 1);
        t.insert_hashes(&[10, 20], 1);
        assert_eq!(t.get(&[10, 20]), 2);
        assert_eq!(owners_set(&t, 10), [1].iter().copied().collect());
    }

    #[test]
    fn empty_insert_is_noop() {
        let mut t = RadixTreeReqIdHash::new();
        t.insert_hashes(&[], 1);
        assert_eq!(t.get(&[10]), 0);
        assert_eq!(t.distinct_hashes(), 0);
    }

    #[test]
    fn evict_orphan_drops_subtree() {
        let mut t = RadixTreeReqIdHash::new();
        t.insert_hashes(&[10, 20, 30], 1);
        t.evict_orphan_hash(10);
        assert_eq!(t.get(&[10]), 0);
        assert_eq!(t.get(&[10, 20]), 0);
    }

    #[test]
    fn epoch_bumps_on_mutation() {
        let mut t = RadixTreeReqIdHash::new();
        let e0 = t.epoch();
        t.insert_hashes(&[10], 1);
        let e1 = t.epoch();
        assert!(e1 > e0);
        t.remove_by_hashes(&[10], 1);
        assert!(t.epoch() > e1);
    }
}
