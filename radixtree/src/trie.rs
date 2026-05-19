// Generic safe trie keyed by `K` with per-node payload `V`.
//
// One key per edge, HashMap children, no compression — the same shape as
// `req_id_hash::RadixTreeReqIdHash` but with the per-node ownership set
// abstracted into a generic `V`. Specializations layer their own struct
// on top by choosing `V`:
//
//   * Preble's owner-tracking specialization picks `V = ReplicaSetNode<...>`
//     so each node carries an owner bitset plus per-node statistics.
//   * `RadixTreeReqIdHash` could later be retrofitted as `Trie<u64, ReqIdSet>`
//     (out of scope for this PR; API supports it).
//
// Trade-off vs the unsafe `core::Node<K, V>` used by `RadixTreeBlockHash`:
// no path compression, no concurrency, no block-id reverse-index. The
// targets here have hundreds–thousands of nodes (single-digit MB) and are
// touched from one task; the simplicity is worth it.

use std::collections::HashMap;
use std::hash::Hash;

/// A safe, generic trie over `K` (edge keys) with per-node `V` payload.
///
/// Layout invariants:
///   * Every internal node holds children keyed by `K`.
///   * Every node carries a `V` payload, default-initialised at creation.
///   * `remove_path` prunes nodes whose payload becomes "empty" per the
///     caller-supplied predicate (see `remove_path_with`).
pub struct Trie<K, V> {
    root: TrieNode<K, V>,
    epoch: u64,
    n_nodes: usize,
}

struct TrieNode<K, V> {
    children: HashMap<K, TrieNode<K, V>>,
    payload: V,
}

impl<K, V> Default for TrieNode<K, V>
where
    V: Default,
{
    fn default() -> Self {
        Self {
            children: HashMap::new(),
            payload: V::default(),
        }
    }
}

impl<K, V> Default for Trie<K, V>
where
    V: Default,
{
    fn default() -> Self {
        Self {
            root: TrieNode::default(),
            epoch: 0,
            n_nodes: 0,
        }
    }
}

impl<K, V> Trie<K, V>
where
    K: Hash + Eq + Clone,
    V: Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    /// Walk to the node at `path`, creating intermediate nodes as needed.
    /// Returns a mutable reference to the leaf node's payload. Bumps
    /// `epoch` if any new node was created.
    ///
    /// `path` is the full key sequence; an empty path returns the root
    /// payload.
    pub fn ensure_path(&mut self, path: &[K]) -> &mut V {
        self.ensure_path_with(path, |_, _| {})
    }

    /// Like `ensure_path`, but invokes `on_each(depth, payload)` on
    /// every node visited along the path (including the final leaf).
    /// `depth` is 1-indexed from root (i.e., the leaf payload is
    /// visited with `depth = path.len()`).
    ///
    /// Used by Preble to add the chosen replica as an owner on every
    /// ancestor of the matched leaf — bijective with AIBrix Go's
    /// `AddOrUpdatePodForModel` walk after target-pod selection
    /// (`prefix_cache_preble.go:555-560`).
    pub fn ensure_path_with<F>(&mut self, path: &[K], mut on_each: F) -> &mut V
    where
        F: FnMut(usize, &mut V),
    {
        let mut created_any = false;
        let mut node = &mut self.root;
        for (i, k) in path.iter().enumerate() {
            let child = node.children.entry(k.clone());
            match child {
                std::collections::hash_map::Entry::Occupied(o) => {
                    node = o.into_mut();
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    created_any = true;
                    self.n_nodes += 1;
                    node = v.insert(TrieNode::default());
                }
            }
            on_each(i + 1, &mut node.payload);
        }
        if created_any {
            self.epoch += 1;
        }
        &mut node.payload
    }

    /// Walk down `path` until either the path is exhausted or we hit a
    /// missing child. Returns the payload at the deepest node we
    /// reached, plus the depth (number of edges traversed). When `path`
    /// is fully matched, depth equals `path.len()`. Returns `None` only
    /// for the empty path (where the deepest node is the root and the
    /// caller likely doesn't care).
    pub fn longest_match(&self, path: &[K]) -> Option<(usize, &V)> {
        if path.is_empty() {
            return None;
        }
        let mut node = &self.root;
        let mut depth = 0usize;
        for k in path {
            match node.children.get(k) {
                Some(c) => {
                    node = c;
                    depth += 1;
                }
                None => break,
            }
        }
        if depth == 0 {
            None
        } else {
            Some((depth, &node.payload))
        }
    }

    /// `mut` variant of `longest_match`.
    pub fn longest_match_mut(&mut self, path: &[K]) -> Option<(usize, &mut V)> {
        if path.is_empty() {
            return None;
        }
        let mut node: *mut TrieNode<K, V> = &mut self.root;
        let mut depth = 0usize;
        // Walk via raw pointer to avoid borrowck rejecting the loop-then-
        // return-mut-of-the-final-node pattern. Safe: we never alias and
        // the tree is single-threaded.
        unsafe {
            for k in path {
                match (*node).children.get_mut(k) {
                    Some(c) => {
                        node = c as *mut _;
                        depth += 1;
                    }
                    None => break,
                }
            }
            if depth == 0 {
                None
            } else {
                Some((depth, &mut (*node).payload))
            }
        }
    }

    /// Length-resolved exact lookup. Same as `longest_match` but only
    /// returns `Some` when the entire `path` exists in the trie.
    pub fn exact(&self, path: &[K]) -> Option<&V> {
        match self.longest_match(path) {
            Some((d, v)) if d == path.len() => Some(v),
            _ => None,
        }
    }

    /// Walk down `path` and return the deepest depth `d ∈ 1..=path.len()`
    /// where `pred(payload)` is true at the node reached at depth `d`.
    /// Returns `None` if no such ancestor exists. Stops walking when
    /// `path` runs out OR a missing child is encountered.
    ///
    /// Used by Preble's Stage 1 to find the deepest ancestor of the
    /// request prefix that the candidate replica owns — bijective with
    /// AIBrix Go's walk-up loop in `Route` (`prefix_cache_preble.go:484-505`).
    pub fn longest_match_with<F>(&self, path: &[K], pred: F) -> Option<(usize, &V)>
    where
        F: Fn(&V) -> bool,
    {
        if path.is_empty() {
            return None;
        }
        let mut node = &self.root;
        let mut best: Option<(usize, &V)> = None;
        for (i, k) in path.iter().enumerate() {
            match node.children.get(k) {
                Some(c) => {
                    node = c;
                    if pred(&node.payload) {
                        best = Some((i + 1, &node.payload));
                    }
                }
                None => break,
            }
        }
        best
    }

    /// Evict subtrees rooted at any descendant whose payload satisfies
    /// `is_stale(payload)`. The entire subtree (the matching node + all
    /// descendants) is dropped in one shot. `is_stale` is checked at
    /// every non-root node top-down; a node is only descended into if
    /// it is itself NOT stale.
    ///
    /// Returns the number of nodes removed. Bumps `epoch` once if any
    /// nodes were removed.
    ///
    /// Bijective with AIBrix Go's `LPRadixCache.Evict` +
    /// `collectNodeAndChildren` (`tree.go:499-538`): a node is removed
    /// iff it itself is stale; descendants are removed transitively
    /// because Go updates `lastAccess` on every traversed node, so a
    /// stale node implies its descendants are at least as stale.
    pub fn evict_subtrees_where<F>(&mut self, is_stale: F) -> usize
    where
        F: Fn(&V) -> bool,
    {
        let mut removed = 0usize;
        Self::evict_recurse(&mut self.root, &is_stale, &mut removed, &mut self.n_nodes);
        if removed > 0 {
            self.epoch += 1;
        }
        removed
    }

    fn evict_recurse<F>(
        node: &mut TrieNode<K, V>,
        is_stale: &F,
        removed: &mut usize,
        n_nodes: &mut usize,
    ) where
        F: Fn(&V) -> bool,
    {
        node.children.retain(|_k, child| {
            if is_stale(&child.payload) {
                let n = Self::count_subtree(child);
                *removed += n;
                *n_nodes -= n;
                false
            } else {
                true
            }
        });
        for child in node.children.values_mut() {
            Self::evict_recurse(child, is_stale, removed, n_nodes);
        }
    }

    fn count_subtree(node: &TrieNode<K, V>) -> usize {
        let mut n = 1; // self
        for c in node.children.values() {
            n += Self::count_subtree(c);
        }
        n
    }

    /// Remove the exact node at `path` AND prune any ancestor whose
    /// `is_empty(payload)` returns true and which has no remaining
    /// children. The caller decides what "empty" means for `V` — Preble
    /// uses "owners empty AND node_to_count is 0".
    ///
    /// Returns `true` iff the leaf at `path` existed and was removed.
    /// Bumps `epoch` only when a node is actually dropped.
    pub fn remove_path_with<F>(&mut self, path: &[K], is_empty: F) -> bool
    where
        F: Fn(&V) -> bool,
    {
        if path.is_empty() {
            return false;
        }
        let removed_any = Self::remove_recurse(&mut self.root, path, &is_empty, &mut self.n_nodes);
        if removed_any {
            self.epoch += 1;
        }
        removed_any
    }

    fn remove_recurse<F>(
        node: &mut TrieNode<K, V>,
        path: &[K],
        is_empty: &F,
        n_nodes: &mut usize,
    ) -> bool
    where
        F: Fn(&V) -> bool,
    {
        if path.is_empty() {
            return false;
        }
        let head = &path[0];
        let tail = &path[1..];
        let mut should_drop_child = false;
        let mut leaf_removed = false;
        if let Some(child) = node.children.get_mut(head) {
            if tail.is_empty() {
                // The leaf-level node. The caller asked us to remove
                // it; we drop it iff its payload is "empty" AND it has
                // no children. Otherwise the caller's intent (drop this
                // node) cannot be satisfied without losing other state,
                // so we leave it alone and report not-removed.
                if child.children.is_empty() && is_empty(&child.payload) {
                    should_drop_child = true;
                    leaf_removed = true;
                }
            } else {
                let sub = Self::remove_recurse(child, tail, is_empty, n_nodes);
                leaf_removed = sub;
                if child.children.is_empty() && is_empty(&child.payload) {
                    should_drop_child = true;
                }
            }
        }
        if should_drop_child {
            node.children.remove(head);
            *n_nodes -= 1;
        }
        leaf_removed
    }

    /// Walk every non-root node, in unspecified order. Callback receives
    /// the full path from root to that node and a shared reference to
    /// the payload.
    pub fn for_each_path<F>(&self, mut f: F)
    where
        F: FnMut(&[K], &V),
    {
        let mut buf: Vec<K> = Vec::new();
        Self::walk(&self.root, &mut buf, &mut f);
    }

    fn walk<F>(node: &TrieNode<K, V>, buf: &mut Vec<K>, f: &mut F)
    where
        F: FnMut(&[K], &V),
    {
        for (k, child) in &node.children {
            buf.push(k.clone());
            f(buf.as_slice(), &child.payload);
            Self::walk(child, buf, f);
            buf.pop();
        }
    }

    /// `mut` variant of `for_each_path`. Children of a node are visited
    /// after its payload, so callers can mutate the payload without
    /// affecting traversal.
    pub fn for_each_path_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&[K], &mut V),
    {
        let mut buf: Vec<K> = Vec::new();
        Self::walk_mut(&mut self.root, &mut buf, &mut f);
    }

    fn walk_mut<F>(node: &mut TrieNode<K, V>, buf: &mut Vec<K>, f: &mut F)
    where
        F: FnMut(&[K], &mut V),
    {
        for (k, child) in node.children.iter_mut() {
            buf.push(k.clone());
            f(buf.as_slice(), &mut child.payload);
            Self::walk_mut(child, buf, f);
            buf.pop();
        }
    }

    /// Monotonic mutation counter. Bumps on path creation and on path
    /// removal; not on payload mutation alone (callers track that
    /// themselves if needed).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Number of non-root nodes currently in the trie.
    pub fn len(&self) -> usize {
        self.n_nodes
    }

    pub fn is_empty(&self) -> bool {
        self.n_nodes == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default, Debug, PartialEq)]
    struct CountPayload {
        count: usize,
    }

    fn is_empty(p: &CountPayload) -> bool {
        p.count == 0
    }

    #[test]
    fn ensure_path_creates_nodes() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        let e0 = t.epoch();
        t.ensure_path(&[10, 20, 30]).count = 1;
        assert!(t.epoch() > e0);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn longest_match_partial_path() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        t.ensure_path(&[10, 20, 30]).count = 1;
        let (d, p) = t.longest_match(&[10, 20, 99]).unwrap();
        assert_eq!(d, 2);
        assert_eq!(p.count, 0); // node at [10, 20] was created but never written
    }

    #[test]
    fn exact_lookup() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        t.ensure_path(&[10, 20]).count = 7;
        assert_eq!(t.exact(&[10, 20]).unwrap().count, 7);
        assert!(t.exact(&[10, 20, 30]).is_none());
    }

    #[test]
    fn remove_prunes_only_empty_chain() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        t.ensure_path(&[10, 20, 30]).count = 0;
        t.ensure_path(&[10, 20, 40]).count = 5; // sibling keeps [10, 20] alive
        let removed = t.remove_path_with(&[10, 20, 30], is_empty);
        assert!(removed);
        assert_eq!(t.exact(&[10, 20]).unwrap().count, 0);
        assert_eq!(t.exact(&[10, 20, 40]).unwrap().count, 5);
        assert!(t.exact(&[10, 20, 30]).is_none());
    }

    #[test]
    fn remove_does_not_prune_nonempty_leaf() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        t.ensure_path(&[10, 20]).count = 3; // non-empty leaf
        let removed = t.remove_path_with(&[10, 20], is_empty);
        assert!(!removed);
        assert_eq!(t.exact(&[10, 20]).unwrap().count, 3);
    }

    #[test]
    fn for_each_path_visits_all() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        t.ensure_path(&[10, 20]).count = 1;
        t.ensure_path(&[10, 30]).count = 2;
        t.ensure_path(&[40]).count = 3;
        let mut sum = 0usize;
        let mut n = 0usize;
        t.for_each_path(|_path, p| {
            sum += p.count;
            n += 1;
        });
        // visits all 4 nodes ([10], [10,20], [10,30], [40]); only 3 have non-zero counts.
        assert_eq!(n, 4);
        assert_eq!(sum, 6);
    }

    #[test]
    fn longest_match_mut_writeback() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        t.ensure_path(&[10, 20]).count = 1;
        let (d, p) = t.longest_match_mut(&[10, 20, 30]).unwrap();
        assert_eq!(d, 2);
        p.count = 99;
        assert_eq!(t.exact(&[10, 20]).unwrap().count, 99);
    }

    #[test]
    fn empty_path_is_noop() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        // ensure_path on empty path returns the root payload (which is fine).
        t.ensure_path(&[]).count = 7;
        assert_eq!(t.len(), 0);
        assert!(t.longest_match(&[] as &[u64]).is_none());
        assert!(!t.remove_path_with(&[] as &[u64], is_empty));
    }

    #[test]
    fn ensure_path_with_visits_every_ancestor() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        let mut visits = Vec::<usize>::new();
        t.ensure_path_with(&[10, 20, 30], |d, p| {
            visits.push(d);
            p.count += 1; // every ancestor's count gets incremented
        });
        assert_eq!(visits, vec![1, 2, 3]);
        assert_eq!(t.exact(&[10]).unwrap().count, 1);
        assert_eq!(t.exact(&[10, 20]).unwrap().count, 1);
        assert_eq!(t.exact(&[10, 20, 30]).unwrap().count, 1);
    }

    #[test]
    fn longest_match_with_picks_deepest_truthy() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        // Build [10, 20, 30] with counts 5, 0, 7
        t.ensure_path_with(&[10, 20, 30], |d, p| {
            p.count = match d {
                1 => 5,
                2 => 0,
                3 => 7,
                _ => 0,
            };
        });
        // Predicate: count > 0. Deepest match along [10, 20, 30] should
        // be at depth 3 (count = 7).
        let (d, p) = t.longest_match_with(&[10, 20, 30], |p| p.count > 0).unwrap();
        assert_eq!(d, 3);
        assert_eq!(p.count, 7);
        // Walk only down [10, 20]: deepest truthy is at depth 1 (count=5);
        // depth 2 has count=0.
        let (d, _) = t.longest_match_with(&[10, 20], |p| p.count > 0).unwrap();
        assert_eq!(d, 1);
        // Path [10, 20, 99] dead-ends after [10, 20]; deepest truthy
        // along the matched portion is depth 1.
        let (d, _) = t.longest_match_with(&[10, 20, 99], |p| p.count > 0).unwrap();
        assert_eq!(d, 1);
    }

    #[test]
    fn longest_match_with_no_match_returns_none() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        t.ensure_path(&[10]).count = 0;
        // Predicate: count > 0 — no node satisfies it.
        assert!(t.longest_match_with(&[10, 20], |p| p.count > 0).is_none());
        // Path that doesn't exist at all.
        assert!(t.longest_match_with(&[99], |_| true).is_none());
    }

    #[test]
    fn evict_subtrees_drops_entire_subtree() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        // Tree: [10] (stale) → [10, 20] → [10, 20, 30]
        //       [40] (fresh)
        t.ensure_path_with(&[10, 20, 30], |_, p| p.count = 0);
        t.ensure_path(&[40]).count = 5;
        let removed = t.evict_subtrees_where(|p| p.count == 0);
        // [10], [10,20], [10,20,30] all evicted (count=0 at each).
        // The eviction at depth 1 drops the whole [10] subtree (3 nodes).
        // [40] survives.
        assert_eq!(removed, 3);
        assert_eq!(t.len(), 1);
        assert_eq!(t.exact(&[40]).unwrap().count, 5);
        assert!(t.exact(&[10]).is_none());
    }

    #[test]
    fn evict_subtrees_descends_past_fresh_nodes() {
        let mut t: Trie<u64, CountPayload> = Trie::new();
        // [10] (fresh) → [10, 20] (stale) → [10, 20, 30] (anything)
        // [10, 40] (fresh)
        t.ensure_path(&[10]).count = 1;
        t.ensure_path(&[10, 20]).count = 0;
        t.ensure_path(&[10, 20, 30]).count = 1;
        t.ensure_path(&[10, 40]).count = 1;
        let removed = t.evict_subtrees_where(|p| p.count == 0);
        // [10] is fresh (count=1), descend; [10,20] is stale → evict
        // subtree (2 nodes). [10] and [10,40] survive.
        assert_eq!(removed, 2);
        assert!(t.exact(&[10]).is_some());
        assert!(t.exact(&[10, 40]).is_some());
        assert!(t.exact(&[10, 20]).is_none());
        assert!(t.exact(&[10, 20, 30]).is_none());
    }
}
