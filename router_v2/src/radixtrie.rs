// NOTE: This file is currently unused — it is not declared as a module in lib.rs.
// It contains a generic RadixTreeMap<K,V> that is independent of the RadixTreeBlockHash
// in kvcache.rs. Kept for potential future use.

use nohash_hasher::{BuildNoHashHasher, IntMap, IsEnabled};
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::xxh3_64;

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{Hash, RandomState};
use std::ops::Add;

use std::mem::{self};
use std::ptr::null_mut;

/// A high-performance RadixTree over sequences of `u64` symbols with **path compression**.
/// Each key is a `&[u64]`.
/// Payload is generic `T: Copy + Default` (e.g., `i64`, `u128`).
///
/// # Features
/// - **Path compression (Patricia)**: nodes may store a *label slice* of multiple `u64`s
///   instead of single-step edges, reducing depth.
/// - Children stored in small sorted Vec (binary search). For high degree, upgrades to HashMap.
/// - Unsafe raw pointers for max speed, manual memory management.
///
/// # Safety
/// - Nodes allocated with `Box::into_raw`, freed in `Drop`.
/// - `Box::from_raw` used exactly once for each node.
/// - Caller must not alias mutable/immutable references.

const SMALL_MAX: usize = 64;

enum Children<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy, S = BuildNoHashHasher<K>>
{
    Small(Vec<*mut Node<K, V>>),
    Large(HashMap<K, *mut Node<K, V>, S>), // Large keyed only by first element of label
}

impl Children<u64, u64, BuildNoHashHasher<u64>> {
    fn clear(&mut self) {
        match self {
            Children::Large(m) => {
                m.clear();
            }
            Children::Small(v) => {
                v.clear();
            }
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Children::Large(m) => m.is_empty(),
            Children::Small(v) => v.is_empty(),
        }
    }
}

impl Default for Children<u64, u64, BuildNoHashHasher<u64>> {
    fn default() -> Self {
        Children::Small(Vec::new())
    }
}

struct Node<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy> {
    children: Children<K, V>,
    value: Vec<K>,   // Hash values of this kvcache block segment
    payload: Vec<V>, // Block ids corresponding to hash values
}

impl<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy> Node<K, V> {
    pub fn is_empty(&self) -> bool {
        let child_is_empty = match &self.children {
            Children::Small(v) => v.is_empty(),
            Children::Large(m) => m.is_empty(),
        };
        child_is_empty && self.value.is_empty() && self.payload.is_empty()
    }
}

impl<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy> Node<K, V> {
    fn new() -> *mut Node<K, V> {
        let boxed = Box::new(Node {
            children: Children::Small(Vec::new()),
            value: Vec::default(),
            payload: Vec::default(),
        });
        Box::into_raw(boxed)
    }
}

pub struct RadixTreeMap<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy> {
    root: *mut Node<K, V>,
    nodes: usize,
    size: usize,
}

unsafe impl<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy> Send
    for RadixTreeMap<K, V>
{
}
unsafe impl<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy> Sync
    for RadixTreeMap<K, V>
{
}

impl<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy> Default for RadixTreeMap<K, V> {
    fn default() -> Self {
        let root = Node::new();
        Self { root, nodes: 1, size: 0 }
    }
}

enum CommonPrefixInner<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy> {
    NoMatch(*mut Node<K, V>),
    PartialMatch(*mut Node<K, V>, usize),
    FullMatch(*mut Node<K, V>),
}

impl<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy> RadixTreeMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.size
    }
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    pub fn node_count(&self) -> usize {
        self.nodes
    }

    /// Splits current node at insert time, returns newly inserted node
    unsafe fn split_node_at_insert(
        &mut self,
        node: *mut Node<K, V>,
        common: usize,
        key: &[K],
        mut value: Vec<V>,
    ) -> *mut Node<K, V> {
        // common := common_prefix(node.value, key)
        // precond: 0 < common < len(node.value)

        // split the node and reattach to itself
        let suffix_value = (*node).value.split_off(common);
        let suffix_payload = (*node).payload.split_off(common);
        let split_new_node = Node::new();
        self.nodes += 1;
        (*split_new_node).value = suffix_value;
        (*split_new_node).payload = suffix_payload;
        mem::swap(&mut (*split_new_node).children, &mut (*node).children);

        // construct new node
        let insert_new_node = Node::new();
        self.nodes += 1;
        (*insert_new_node).value = key[common..].to_vec();
        (*insert_new_node).payload = value.split_off(common);

        // update the node to valid state
        // (*node).value has been split by `split_off`
        // (*node).payload has been split by `split_off`
        (*node).children = Children::Small(vec![split_new_node, insert_new_node]);

        insert_new_node
    }

    /// Continuously traverse FullMatch nodes
    ///
    /// # Precondition
    ///   - NOT empty(key) => subroot is fully matched
    ///
    /// # Postcondition
    ///   - FullMatch => empty(key) OR all children are NoMatch
    unsafe fn common_prefix_full_combo(
        subroot: *const Node<K, V>,
        key: &[K],
    ) -> (CommonPrefixInner<K, V>, &[K]) {
        if key.is_empty() {
            return (CommonPrefixInner::FullMatch(subroot as *mut Node<K, V>), key);
        }
        // postcond: NOT empty(key)
        match &(*subroot).children {
            Children::Small(v) => {
                for &child in v.iter() {
                    match Self::common_prefix(child, key) {
                        CommonPrefixInner::FullMatch(_) => {
                            return Self::common_prefix_full_combo(
                                child,
                                &key[(*child).value.len()..],
                            );
                        }
                        CommonPrefixInner::PartialMatch(_, common) => {
                            return (CommonPrefixInner::PartialMatch(child, common), key);
                        }
                        CommonPrefixInner::NoMatch(_) => {}
                    }
                }
                // postcond: no match with any child
                return (CommonPrefixInner::FullMatch(subroot as *mut Node<K, V>), key);
            }
            Children::Large(m) => {
                if let Some(&child) = m.get(&key[0]) {
                    // postcond: child contains first u64 of key
                    match Self::common_prefix(child, key) {
                        CommonPrefixInner::FullMatch(_) => {
                            return Self::common_prefix_full_combo(
                                child,
                                &key[(*child).value.len()..],
                            );
                        }
                        CommonPrefixInner::PartialMatch(_, common) => {
                            return (CommonPrefixInner::PartialMatch(child, common), key);
                        }
                        CommonPrefixInner::NoMatch(_) => {
                            unreachable!("The first u64 must occur in some Node!")
                        }
                    }
                } else {
                    // postcond: no match with any child
                    return (CommonPrefixInner::FullMatch(subroot as *mut Node<K, V>), key);
                }
            }
        }
    }

    /// Insert a new node as direct child
    unsafe fn insert_as_child(
        &mut self,
        node: *mut Node<K, V>,
        key: &[K],
        value: Vec<V>,
    ) -> *mut Node<K, V> {
        // create new node
        let new_node = Node::new();
        self.nodes += 1;
        (*new_node).value = key.to_vec();
        (*new_node).payload = value;
        // add sequence to the children of current node
        match &mut (*node).children {
            Children::Small(v) => {
                v.push(new_node);
                if v.len() > SMALL_MAX {
                    // (NOTE) precond: NOT empty((*n).value)
                    //   + this is beacuse there are 16 children, and empty children can't exist,
                    //     it will be absorbed by its parent
                    let tmp: HashMap<K, *mut Node<K, V>, BuildNoHashHasher<K>> =
                        v.iter().map(|&n| ((*n).value[0], n)).collect();
                    let _ = mem::replace(&mut (*node).children, Children::Large(tmp));
                }
            }
            Children::Large(m) => {
                let exist_node = m.insert(key[0], new_node);
                assert!(
                    exist_node == None,
                    "Block<hash={:?}> exists in Node<{:?}>",
                    key[0],
                    (*exist_node.unwrap()).value
                );
            }
        }
        new_node
    }

    unsafe fn insert_inner(&mut self, key: &[K], mut value: Vec<V>) -> usize {
        if key.is_empty() {
            return 0;
        }
        // postcond: key is not empty
        // skip the root node
        let root_match = match &(*self.root).children {
            Children::Small(v) => {
                let mut res = CommonPrefixInner::NoMatch(self.root);
                for &child in v.iter() {
                    if (*child).value[0] == key[0] {
                        res = Self::common_prefix(child, key);
                    }
                }
                res
            }
            Children::Large(m) => {
                let res;
                if let Some(&child) = m.get(&key[0]) {
                    res = Self::common_prefix(child, key);
                } else {
                    res = CommonPrefixInner::NoMatch(self.root);
                }
                res
            }
        };
        //
        match root_match {
            CommonPrefixInner::FullMatch(node) => {
                // postcond: a first level node fully matches key
                match Self::common_prefix_full_combo(node, &key[(*node).value.len()..]) {
                    (CommonPrefixInner::FullMatch(node), key1) => {
                        if key1.is_empty() {
                            return key.len();
                        }
                        // postcond: unmatched key sequence
                        let full_match_len = key.len() - key1.len();
                        let remain_value = value.split_off(full_match_len);
                        // invariant: key1 |-> remain_value
                        if match &(*node).children {
                            Children::Small(v) => v.is_empty(),
                            Children::Large(m) => m.is_empty(),
                        } {
                            (*node).value.extend_from_slice(key1);
                            (*node).payload.extend(remain_value);
                        } else {
                            self.insert_as_child(node, key1, remain_value);
                        }
                        return full_match_len;
                    }
                    (CommonPrefixInner::PartialMatch(node, common), key1) => {
                        if key1.len() == common {
                            // postcond: full match
                            return key.len();
                        }
                        let full_match_len = key.len() - key1.len();
                        let remain_value = value.split_off(full_match_len);
                        // invariant: key1 |-> remain_value
                        self.split_node_at_insert(node, common, key1, remain_value);
                        return full_match_len + common;
                    }
                    (CommonPrefixInner::NoMatch(_), _) => {
                        unreachable!()
                    }
                }
            }
            CommonPrefixInner::PartialMatch(node, common) => {
                // postcond: a first level node partially matches key
                if key.len() == common {
                    // postcond: full match
                    return key.len();
                }
                let _ = self.split_node_at_insert(node, common, key, value);
                return common;
            }
            CommonPrefixInner::NoMatch(node) => {
                // postcond: zero matched first level node
                assert_eq!(node, self.root);
                let _ = self.insert_as_child(self.root, key, value);
                return 0;
            }
        }
    }

    /// Insert a sequence into this RadixTree, returns matched prefix length
    ///
    /// # Returns
    /// Matched prefix length `usize`
    pub fn insert(&mut self, key: &[K], value: Vec<V>) -> usize {
        unsafe { self.insert_inner(key, value) }
    }

    /// Returns matched prefix length
    pub fn prefix_len(&self, key: &[K]) -> usize {
        unsafe { self.prefix_len_inner(key) }
    }

    unsafe fn prefix_len_inner(&self, key: &[K]) -> usize {
        let mut prefix_len = 0;
        let mut cur = self.root;

        'level_down: loop {
            match &(*cur).children {
                Children::Small(v) => {
                    if prefix_len == key.len() {
                        // finish match: terminate
                        break 'level_down;
                    }
                    'next_child: for &child in v.iter() {
                        let mut i = 0;
                        while i < (*child).value.len()
                            && prefix_len + i < key.len()
                            && (*child).value[i] == key[prefix_len + i]
                        {
                            i += 1;
                        }
                        if i == 0 {
                            // no match to this child, try next
                            continue 'next_child;
                        }
                        // postcond: matched
                        prefix_len += i;
                        if i < (*child).value.len() {
                            // partial match: stop
                            break 'level_down;
                        }
                        // postcond: full match
                        // full match: descend
                        cur = child;
                        continue 'level_down;
                    }
                    // no child matches next symbol
                    break 'level_down;
                }
                Children::Large(m) => {
                    if prefix_len == key.len() {
                        // finish match: terminate
                        break;
                    }
                    // invariant: key[prefix_len] |-> the first of some node
                    if let Some(&child) = m.get(&key[prefix_len]) {
                        let mut i = 0;
                        // precond: matched
                        while i < (*child).value.len()
                            && prefix_len + i < key.len()
                            && (*child).value[i] == key[prefix_len + i]
                        {
                            i += 1;
                        }
                        // postcond: i > 0
                        prefix_len += i;
                        if i < (*child).value.len() {
                            // partial match: stop
                            break;
                        }
                        // full match: descend
                        cur = child;
                        // invariant: key[prefix_len] |-> the first of some node
                    } else {
                        // no child matches next symbol
                        break;
                    }
                }
            }
        }

        prefix_len
    }

    fn common_prefix(node: *const Node<K, V>, key: &[K]) -> CommonPrefixInner<K, V> {
        let mut i = 0;
        unsafe {
            while i < (*node).value.len() && i < key.len() && (*node).value[i] == key[i] {
                i += 1;
            }
            if i == 0 {
                return CommonPrefixInner::NoMatch(node as *mut Node<K, V>);
            } else if i == (*node).value.len() {
                return CommonPrefixInner::FullMatch(node as *mut Node<K, V>);
            } else {
                return CommonPrefixInner::PartialMatch(node as *mut Node<K, V>, i);
            }
        }
    }

    // pub fn for_each<F: FnMut(&[u64], T)>(&self, mut f: F) {
    //     unsafe {
    //         let mut buf = Vec::new();
    //         self.dfs(self.root, &mut buf, &mut f);
    //     }
    // }

    // unsafe fn dfs<F: FnMut(&[u64], T)>(&self, node: *mut Node<T>, buf: &mut Vec<u64>, f: &mut F) {
    //     if (*node).has_value {
    //         f(buf.as_slice(), (*node).value);
    //     }
    //     match &(*node).children {
    //         Children::Small(v) => {
    //             for e in v {
    //                 let start = buf.len();
    //                 buf.extend_from_slice(&e.label);
    //                 self.dfs(e.child, buf, f);
    //                 buf.truncate(start);
    //             }
    //         }
    //         Children::Large(m) => {
    //             for (&label, &child) in m.iter() {
    //                 buf.push(label);
    //                 self.dfs(child, buf, f);
    //                 buf.pop();
    //             }
    //         }
    //     }
    // }
}

impl<K: Copy + Default + Eq + Hash + Debug + IsEnabled, V: Copy> Drop for RadixTreeMap<K, V> {
    fn drop(&mut self) {
        unsafe {
            let mut stack = vec![self.root];
            let mut post = Vec::with_capacity(self.nodes);
            while let Some(n) = stack.pop() {
                post.push(n);
                match &(*n).children {
                    Children::Small(v) => {
                        for &c in v {
                            stack.push(c);
                        }
                    }
                    Children::Large(m) => {
                        for &c in m.values() {
                            stack.push(c);
                        }
                    }
                }
            }
            while let Some(n) = post.pop() {
                drop(Box::from_raw(n));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RadixTreeMap;

    #[test]
    fn test_prefix_len_simple() {
        let mut t: RadixTreeMap<u64, u32> = RadixTreeMap::new();
        t.insert(&[1, 2, 3], vec![10, 20, 30]);
        t.insert(&[1, 2, 4], vec![40, 50]);
        t.insert(&[2, 5], vec![60, 70]);

        // exact match
        assert_eq!(t.prefix_len(&[1, 2, 3]), 3);
        // match prefix [1,2]
        assert_eq!(t.prefix_len(&[1, 2, 5, 6]), 2);
        // match prefix [2]
        assert_eq!(t.prefix_len(&[2, 7, 8]), 1);
        // no match at all
        assert_eq!(t.prefix_len(&[9, 9, 9]), 0);
    }

    #[test]
    fn test_prefix_len_partial_split() {
        let mut t: RadixTreeMap<u64, u32> = RadixTreeMap::new();
        t.insert(&[1, 2, 3, 4], vec![1, 2, 3, 4]);
        // This forces a split at [1,2,5]
        t.insert(&[1, 2, 5], vec![5, 5, 5]);

        assert_eq!(t.prefix_len(&[1, 2, 5, 6]), 3);
        assert_eq!(t.prefix_len(&[1, 2, 3, 4, 9]), 4);
        assert_eq!(t.prefix_len(&[1, 2, 9]), 2);
    }
}

struct RadixTreeBlockHash {
    root: *mut Node<u64, u64>,
    size: usize,
    nodes: usize,
    block_to_node: Vec<*mut Node<u64, u64>>,
}

unsafe impl Send for RadixTreeBlockHash {}
unsafe impl Sync for RadixTreeBlockHash {}

impl RadixTreeBlockHash {
    pub fn new(num_blocks: usize) -> Self {
        let root = Node::<u64, u64>::new();
        Self { root, size: 0, nodes: 1, block_to_node: Vec::with_capacity(num_blocks) }
    }
    pub fn len(&self) -> usize {
        self.size
    }
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    pub fn node_count(&self) -> usize {
        self.nodes
    }

    /// Splits current node at insert time, returns newly inserted node
    unsafe fn split_node_at_insert(
        &mut self,
        node: *mut Node<u64, u64>,
        common: usize,
        block_hashes: &[u64],
        mut block_indices: Vec<u64>,
    ) {
        // common := common_prefix(node.value, key)
        // precond: 0 < common < len(node.value)

        // split the node and reattach to itself
        let suffix_value = (*node).value.split_off(common);
        let suffix_payload = (*node).payload.split_off(common);
        let split_new_node = Node::new();
        self.nodes += 1;
        (*split_new_node).value = suffix_value;
        (*split_new_node).payload = suffix_payload;
        mem::swap(&mut (*split_new_node).children, &mut (*node).children);

        // construct new node
        let insert_new_node = Node::new();
        self.nodes += 1;
        (*insert_new_node).value = block_hashes[common..].to_vec();
        (*insert_new_node).payload = block_indices.split_off(common);

        // update the node to valid state
        // (*node).value has been split by `split_off`
        // (*node).payload has been split by `split_off`
        (*node).children = Children::Small(vec![split_new_node, insert_new_node]);

        // validate block to node
        for &block_id in (*split_new_node).payload.iter() {
            assert_eq!(self.block_to_node[block_id as usize], node);
            self.block_to_node[block_id as usize] = split_new_node;
        }
        for &block_id in (*insert_new_node).payload.iter() {
            assert_eq!(self.block_to_node[block_id as usize], null_mut());
            self.block_to_node[block_id as usize] = insert_new_node;
        }
    }

    /// Continuously traverse FullMatch nodes
    ///
    /// # Precondition
    ///   - NOT empty(key) => subroot is fully matched
    ///
    /// # Postcondition
    ///   - FullMatch => empty(key) OR all children are NoMatch
    unsafe fn common_prefix_full_combo<'a, 'b>(
        &'a self,
        subroot: *const Node<u64, u64>,
        key: &'b [u64],
    ) -> (CommonPrefixInner<u64, u64>, &'b [u64]) {
        if key.is_empty() {
            return (CommonPrefixInner::FullMatch(subroot as *mut Node<u64, u64>), key);
        }
        // postcond: NOT empty(key)
        match &(*subroot).children {
            Children::Small(v) => {
                for &child in v.iter() {
                    match self.common_prefix(child, key) {
                        CommonPrefixInner::FullMatch(_) => {
                            return self
                                .common_prefix_full_combo(child, &key[(*child).value.len()..]);
                        }
                        CommonPrefixInner::PartialMatch(_, common) => {
                            return (CommonPrefixInner::PartialMatch(child, common), key);
                        }
                        CommonPrefixInner::NoMatch(_) => {}
                    }
                }
                // postcond: no match with any child
                return (CommonPrefixInner::FullMatch(subroot as *mut Node<u64, u64>), key);
            }
            Children::Large(m) => {
                if let Some(&child) = m.get(&key[0]) {
                    // postcond: child contains first u64 of key
                    match self.common_prefix(child, key) {
                        CommonPrefixInner::FullMatch(_) => {
                            return self
                                .common_prefix_full_combo(child, &key[(*child).value.len()..]);
                        }
                        CommonPrefixInner::PartialMatch(_, common) => {
                            return (CommonPrefixInner::PartialMatch(child, common), key);
                        }
                        CommonPrefixInner::NoMatch(_) => {
                            unreachable!("The first u64 must occur in some Node!")
                        }
                    }
                } else {
                    // postcond: no match with any child
                    return (CommonPrefixInner::FullMatch(subroot as *mut Node<u64, u64>), key);
                }
            }
        }
    }

    /// Insert a new node as direct child
    unsafe fn insert_as_child(
        &mut self,
        node: *mut Node<u64, u64>,
        block_hashes: &[u64],
        block_indices: Vec<u64>,
    ) {
        // create new node
        let new_node = Node::new();
        self.nodes += 1;
        (*new_node).value = block_hashes.to_vec();
        (*new_node).payload = block_indices;
        // add sequence to the children of current node
        match &mut (*node).children {
            Children::Small(v) => {
                v.push(new_node);
                if v.len() > SMALL_MAX {
                    // (NOTE) precond: NOT empty((*n).value)
                    //   + this is beacuse there are 16 children, and empty children can't exist,
                    //     it will be absorbed by its parent
                    let tmp: IntMap<u64, *mut Node<u64, u64>> =
                        v.iter().map(|&n| ((*n).value[0], n)).collect();
                    let _ = mem::replace(&mut (*node).children, Children::<u64, u64>::Large(tmp));
                }
            }
            Children::Large(m) => {
                let exist_node = m.insert(block_hashes[0], new_node);
                assert!(
                    exist_node == None,
                    "Block<hash={:?}> exists in Node<{:?}>",
                    block_hashes[0],
                    (*exist_node.unwrap()).value
                );
            }
        }

        // validate block to node map
        for &block_id in (*new_node).payload.iter() {
            assert_ne!(self.block_to_node[block_id as usize], null_mut());
            self.block_to_node[block_id as usize] = new_node;
        }
    }

    unsafe fn insert_inner(&mut self, key: &[u64], mut value: Vec<u64>) -> usize {
        if key.is_empty() {
            return 0;
        }
        // postcond: key is not empty
        // skip the root node
        let root_match = match &(*self.root).children {
            Children::Small(v) => {
                let mut res = CommonPrefixInner::NoMatch(self.root);
                for &child in v.iter() {
                    if (*child).value[0] == key[0] {
                        res = self.common_prefix(child, key);
                    }
                }
                res
            }
            Children::Large(m) => {
                let res;
                if let Some(&child) = m.get(&key[0]) {
                    res = self.common_prefix(child, key);
                } else {
                    res = CommonPrefixInner::NoMatch(self.root);
                }
                res
            }
        };
        //
        match root_match {
            CommonPrefixInner::FullMatch(node) => {
                // postcond: a first level node fully matches key
                match self.common_prefix_full_combo(node, &key[(*node).value.len()..]) {
                    (CommonPrefixInner::FullMatch(node), key1) => {
                        if key1.is_empty() {
                            return key.len();
                        }
                        // postcond: unmatched key sequence
                        let full_match_len = key.len() - key1.len();
                        let remain_value = value.split_off(full_match_len);
                        // invariant: key1 |-> remain_value
                        if match &(*node).children {
                            Children::Small(v) => v.is_empty(),
                            Children::Large(m) => m.is_empty(),
                        } {
                            (*node).value.extend_from_slice(key1);
                            (*node).payload.extend(remain_value);
                            // validate block to node map
                            for &block_id in (*node).payload.iter() {
                                assert!(
                                    self.block_to_node[block_id as usize].is_null()
                                        || self.block_to_node[block_id as usize] == node
                                );
                                self.block_to_node[block_id as usize] = node;
                            }
                        } else {
                            self.insert_as_child(node, key1, remain_value);
                        }
                        return full_match_len;
                    }
                    (CommonPrefixInner::PartialMatch(node, common), key1) => {
                        if key1.len() == common {
                            // postcond: full match
                            return key.len();
                        }
                        let full_match_len = key.len() - key1.len();
                        let remain_value = value.split_off(full_match_len);
                        // invariant: key1 |-> remain_value
                        self.split_node_at_insert(node, common, key1, remain_value);
                        return full_match_len + common;
                    }
                    (CommonPrefixInner::NoMatch(_), _) => {
                        unreachable!()
                    }
                }
            }
            CommonPrefixInner::PartialMatch(node, common) => {
                // postcond: a first level node partially matches key
                if key.len() == common {
                    // postcond: full match
                    return key.len();
                }
                let _ = self.split_node_at_insert(node, common, key, value);
                return common;
            }
            CommonPrefixInner::NoMatch(node) => {
                // postcond: zero matched first level node
                assert_eq!(node, self.root);
                let _ = self.insert_as_child(self.root, key, value);
                return 0;
            }
        }
    }

    /// Insert a sequence into this RadixTree, returns matched prefix length
    ///
    /// # Returns
    /// Matched prefix length `usize`
    pub fn insert(&mut self, block_hashes: &[u64], block_indices: Vec<u64>) -> usize {
        unsafe { self.insert_inner(block_hashes, block_indices) }
    }

    /// Returns matched prefix length
    pub fn prefix_len(&self, block_hashes: &[u64]) -> usize {
        unsafe { self.prefix_len_inner(block_hashes) }
    }

    unsafe fn prefix_len_inner(&self, block_hashes: &[u64]) -> usize {
        let mut prefix_len = 0;
        let mut cur = self.root;

        'level_down: loop {
            match &(*cur).children {
                Children::Small(v) => {
                    if prefix_len == block_hashes.len() {
                        // finish match: terminate
                        break 'level_down;
                    }
                    'next_child: for &child in v.iter() {
                        let mut i = 0;
                        while i < (*child).value.len()
                            && prefix_len + i < block_hashes.len()
                            && (*child).value[i] == block_hashes[prefix_len + i]
                        {
                            i += 1;
                        }
                        if i == 0 {
                            if (*child).is_empty() {
                                // lazy GC
                                drop(Box::from_raw(child));
                            }
                            // no match to this child, try next
                            continue 'next_child;
                        }
                        // postcond: matched
                        prefix_len += i;
                        if i < (*child).value.len() {
                            // partial match: stop
                            break 'level_down;
                        }
                        // postcond: full match
                        // full match: descend
                        cur = child;
                        continue 'level_down;
                    }
                    // no child matches next symbol
                    break 'level_down;
                }
                Children::Large(m) => {
                    if prefix_len == block_hashes.len() {
                        // finish match: terminate
                        break;
                    }
                    // invariant: key[prefix_len] |-> the first of some node
                    if let Some(&child) = m.get(&block_hashes[prefix_len]) {
                        if (*child).is_empty() {
                            // lazy GC
                            drop(Box::from_raw(child));
                            // no child matches next symbol
                            break;
                        }
                        // postcond: matched
                        let mut i = 0;
                        // precond: matched
                        while i < (*child).value.len()
                            && prefix_len + i < block_hashes.len()
                            && (*child).value[i] == block_hashes[prefix_len + i]
                        {
                            i += 1;
                        }
                        // postcond: i > 0
                        prefix_len += i;
                        if i < (*child).value.len() {
                            // partial match: stop
                            break;
                        }
                        // full match: descend
                        cur = child;
                        // invariant: key[prefix_len] |-> the first of some node
                    } else {
                        // no child matches next symbol
                        break;
                    }
                }
            }
        }

        prefix_len
    }

    fn common_prefix(
        &self,
        node: *const Node<u64, u64>,
        key: &[u64],
    ) -> CommonPrefixInner<u64, u64> {
        let mut i = 0;
        unsafe {
            while i < (*node).value.len() && i < key.len() && (*node).value[i] == key[i] {
                debug_assert_eq!(
                    self.block_to_node[(*node).payload[i] as usize],
                    node as *mut Node<u64, u64>
                );
                i += 1;
            }
            if i == 0 {
                return CommonPrefixInner::NoMatch(node as *mut Node<u64, u64>);
            } else if i == (*node).value.len() {
                return CommonPrefixInner::FullMatch(node as *mut Node<u64, u64>);
            } else {
                return CommonPrefixInner::PartialMatch(node as *mut Node<u64, u64>, i);
            }
        }
    }

    unsafe fn clear_inner(&mut self, node: *mut Node<u64, u64>, to_drop: bool) -> usize {
        let mut n = 0;

        for &block_id in (*node).payload.iter() {
            self.block_to_node[block_id as usize] = null_mut();
            n += 1;
        }
        (*node).value.clear();
        (*node).payload.clear();
        match &(*node).children {
            Children::Small(v) => {
                for &child in v {
                    n += self.clear_inner(child, to_drop);
                }
            }
            Children::Large(m) => {
                for &child in m.values() {
                    n += self.clear_inner(child, to_drop);
                }
            }
        }
        if to_drop {
            drop(Box::from_raw(node));
        }

        n
    }

    /// block id
    unsafe fn remove_inner(&mut self, block_indices: &Vec<u64>) {
        let mut nblock_canary: usize = 0;

        for &block_id in block_indices.iter().rev() {
            let node = self.block_to_node[block_id as usize];
            if node.is_null() {
                continue;
            }
            // precond: `node` is not nil
            // precond: `block_id` must exist in `node`
            let mut pos = 0;
            while pos < (*node).payload.len() && (*node).payload[pos] != block_id {
                pos += 1;
            }
            // update `node` to valid state
            (*node).value.truncate(pos);
            let block_id_to_rm = (*node).payload.split_off(pos);
            for bid in block_id_to_rm {
                self.block_to_node[bid as usize] = null_mut();
                nblock_canary += 1;
            }
            if !(*node).children.is_empty() {
                match &(*node).children {
                    Children::Small(v) => {
                        for &child in v {
                            nblock_canary += self.clear_inner(child, true);
                        }
                    }
                    Children::Large(m) => {
                        for &child in m.values() {
                            nblock_canary += self.clear_inner(child, true);
                        }
                    }
                }
                (*node).children.clear();
            }
            // now `node` is in valid state
            // NOTE: always keep this node, defer GC to next scan
        }

        assert_eq!(nblock_canary, block_indices.len());
    }

    pub fn remove(&mut self, block_indices: Vec<u64>) {
        unsafe {
            self.remove_inner(&block_indices);
        }
    }

    // pub fn for_each<F: FnMut(&[u64], T)>(&self, mut f: F) {
    //     unsafe {
    //         let mut buf = Vec::new();
    //         self.dfs(self.root, &mut buf, &mut f);
    //     }
    // }

    // unsafe fn dfs<F: FnMut(&[u64], T)>(&self, node: *mut Node<T>, buf: &mut Vec<u64>, f: &mut F) {
    //     if (*node).has_value {
    //         f(buf.as_slice(), (*node).value);
    //     }
    //     match &(*node).children {
    //         Children::Small(v) => {
    //             for e in v {
    //                 let start = buf.len();
    //                 buf.extend_from_slice(&e.label);
    //                 self.dfs(e.child, buf, f);
    //                 buf.truncate(start);
    //             }
    //         }
    //         Children::Large(m) => {
    //             for (&label, &child) in m.iter() {
    //                 buf.push(label);
    //                 self.dfs(child, buf, f);
    //                 buf.pop();
    //             }
    //         }
    //     }
    // }
}

impl Drop for RadixTreeBlockHash {
    fn drop(&mut self) {
        unsafe {
            let mut stack = vec![self.root];
            let mut post = Vec::with_capacity(self.nodes);
            while let Some(n) = stack.pop() {
                post.push(n);
                match &(*n).children {
                    Children::Small(v) => {
                        for &c in v {
                            stack.push(c);
                        }
                    }
                    Children::Large(m) => {
                        for &c in m.values() {
                            stack.push(c);
                        }
                    }
                }
            }
            while let Some(n) = post.pop() {
                drop(Box::from_raw(n));
            }
        }
    }
}

/// PrefixCache for single vllm instance
pub(crate) struct PrefixCache {
    inner: RadixTreeBlockHash,
}

impl PrefixCache {
    pub(crate) fn new(num_blocks: usize) -> Self {
        PrefixCache { inner: RadixTreeBlockHash::new(num_blocks) }
    }

    pub(crate) fn match_prefix(&self, hash_values: Vec<u64>) -> usize {
        self.inner.prefix_len(hash_values.as_slice())
    }

    pub(crate) fn evict_blocks(&mut self, block_ids: Vec<u64>) {
        for id in block_ids.iter() {
            self.inner.remove(id);
        }
    }
}
