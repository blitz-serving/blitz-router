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
        let mut created_any = false;
        let mut node = &mut self.root;
        for k in path {
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
}
