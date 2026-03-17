use nohash_hasher::{self, BuildNoHashHasher, IntMap};

use std::collections::HashMap;
use std::mem::{self};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::SpinLock;

pub(crate) static DEFAULT_BLOCK_HASH: u64 = 42;
#[cfg(feature = "default-hash-algo")]
pub(crate) type BackendBlockHash = [u64; 1];
#[cfg(feature = "sha256-hash-algo")]
pub(crate) type BackendBlockHash = [u64; 4];

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

const SMALL_MAX: usize = 32;

enum Children<K: nohash_hasher::IsEnabled, V, S = BuildNoHashHasher<K>> {
    Small(Vec<*mut Node<K, V>>),
    Large(HashMap<K, *mut Node<K, V>, S>), // Large keyed only by first element of label
}

impl<K: nohash_hasher::IsEnabled, V, S> Children<K, V, S> {
    /// Only clears the container; DOES NOT drop pointed nodes.
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

impl<K: nohash_hasher::IsEnabled, V, S> Default for Children<K, V, S> {
    fn default() -> Self {
        Children::Small(Vec::new())
    }
}

struct Node<K: nohash_hasher::IsEnabled, V> {
    children: Children<K, V>,
    value: Vec<K>,   // Hash values of this kvcache block segment
    payload: Vec<V>, // Block ids corresponding to hash values
}

impl<K: nohash_hasher::IsEnabled, V> Node<K, V> {
    fn is_empty(&self) -> bool {
        let child_is_empty = match &self.children {
            Children::Small(v) => v.is_empty(),
            Children::Large(m) => m.is_empty(),
        };
        child_is_empty && self.value.is_empty() && self.payload.is_empty()
    }

    fn new() -> *mut Node<K, V> {
        let boxed = Box::new(Node {
            children: Children::Small(Vec::new()),
            value: Vec::default(),
            payload: Vec::default(),
        });
        Box::into_raw(boxed)
    }
}

enum CommonPrefixInner<K: nohash_hasher::IsEnabled, V> {
    NoMatch(*mut Node<K, V>),
    PartialMatch(*mut Node<K, V>, usize),
    FullMatch(*mut Node<K, V>),
}

pub trait BlockHash {
    /// Must specify the number of KVCache blocks at backend
    fn new(num_blocks: usize) -> Self
    where
        Self: Sized;
    /// Returns the number of cached KVCache block at backend
    #[allow(unused)]
    fn len(&self) -> usize;
    #[allow(unused)]
    fn is_empty(&self) -> bool;
    /// Inserts a sequence into this RadixTree, returns inserted number of hashes
    /// # NOTE:
    /// BlockHash is possibly an 1 -> N structure.
    /// No dedup is because vLLM wants to avoid modifying GPU related states
    fn insert(&mut self, block_hashes: &[u64], block_indices: Vec<u64>) -> usize;
    /// Returns matched prefix length
    fn get(&self, block_hashes: &[u64]) -> usize;
    /// Removes corresponding prefix cache blocks
    fn remove(&mut self, block_indices: Vec<u64>);
}

pub(crate) struct RadixTreeBlockHash {
    root: *mut Node<u64, u64>,
    size: usize,
    nodes: AtomicUsize,
    block_to_node: Vec<*mut Node<u64, u64>>,
    mtx: SpinLock,
}

unsafe impl Send for RadixTreeBlockHash {}
unsafe impl Sync for RadixTreeBlockHash {}

impl RadixTreeBlockHash {
    /// Internal helper method, for memory leak avoidance
    #[allow(unused)]
    fn node_count(&self) -> usize {
        self.nodes.load(Ordering::Relaxed)
    }

    /// Splits current node at insert time, returns newly inserted node
    ///
    /// `block_hashed` and `block_indices` are aligned at `node` boundary,
    /// inclusively contain `common` `V`
    ///
    /// # Precondition
    ///   - `common` < `block_hashes.len()` otherwise, its a perfect match
    unsafe fn split_node_at_insert(
        &mut self,
        node: *mut Node<u64, u64>,
        common: usize,
        block_hashes: &[u64],
        mut block_indices: Vec<u64>,
    ) {
        assert!(common < block_hashes.len());
        // common := common_prefix(node.value, key)
        // precond: 0 < common < len(node.value)

        // split the node and reattach to itself
        let suffix_value = (*node).value.split_off(common);
        let suffix_payload = (*node).payload.split_off(common);
        let split_new_node = Node::new();
        self.nodes.fetch_add(1, Ordering::Relaxed);
        (*split_new_node).value = suffix_value;
        (*split_new_node).payload = suffix_payload;
        mem::swap(&mut (*split_new_node).children, &mut (*node).children);

        // construct new node
        let insert_new_node = Node::new();
        self.nodes.fetch_add(1, Ordering::Relaxed);
        (*insert_new_node).value = block_hashes[common..].to_vec();
        (*insert_new_node).payload = block_indices.split_off(common);

        // update the node to valid state
        // (*node).value has been split by `split_off`
        // (*node).payload has been split by `split_off`
        (*node).children = Children::Small(vec![split_new_node, insert_new_node]);

        // validate block to node
        for &block_id in (*split_new_node).payload.iter() {
            assert_eq!(
                self.block_to_node[block_id as usize],
                node,
                "block#{} |-> {:?}",
                block_id,
                (*self.block_to_node[block_id as usize]).value
            );
            self.block_to_node[block_id as usize] = split_new_node;
        }
        for &block_id in (*insert_new_node).payload.iter() {
            assert_eq!(
                self.block_to_node[block_id as usize],
                null_mut(),
                "block#{} |-> {:?}",
                block_id,
                (*self.block_to_node[block_id as usize]).value
            );
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
        subroot: *mut Node<u64, u64>,
        key: &'b [u64],
    ) -> (CommonPrefixInner<u64, u64>, &'b [u64]) {
        if key.is_empty() {
            return (CommonPrefixInner::FullMatch(subroot as *mut Node<u64, u64>), key);
        }
        // postcond: NOT empty(key)
        match &mut (*subroot).children {
            Children::Small(v) => {
                let mut i = 0;
                while i < v.len() {
                    let child = v[i];
                    if (*child).is_empty() {
                        self.mtx.lock();
                        if (*child).is_empty() {
                            v.remove(i);
                            self.atomize_manually_lazy_gc(child);
                        }
                        self.mtx.unlock();
                        continue;
                    }
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
                    i += 1;
                }
                // postcond: no match with any child
                return (CommonPrefixInner::FullMatch(subroot as *mut Node<u64, u64>), key);
            }
            Children::Large(m) => {
                if let Some(&child) = m.get(&key[0]) {
                    if (*child).is_empty() {
                        self.mtx.lock();
                        if (*child).is_empty() {
                            let _ = m.remove(&key[0]);
                            self.atomize_manually_lazy_gc(child);
                        }
                        self.mtx.unlock();
                        // postcond: no match with any child
                        return (CommonPrefixInner::FullMatch(subroot), key);
                    }
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

    /// Inserts a new node as direct child of `node`
    unsafe fn insert_as_child(
        &mut self,
        node: *mut Node<u64, u64>,
        block_hashes: &[u64],
        block_indices: Vec<u64>,
    ) {
        // create new node
        let new_node = Node::new();
        self.nodes.fetch_add(1, Ordering::Relaxed);
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
                if let Some(en) = m.insert(block_hashes[0], new_node) {
                    panic!("Block<hash={:?}> exists in Node<{:?}>", block_hashes[0], unsafe {
                        &(*en).value
                    });
                }
            }
        }

        // validate block to node map
        for &block_id in (*new_node).payload.iter() {
            assert_eq!(self.block_to_node[block_id as usize], null_mut());
            self.block_to_node[block_id as usize] = new_node;
        }
    }

    unsafe fn insert_inner(&mut self, key: &[u64], mut value: Vec<u64>) -> usize {
        if key.is_empty() {
            return 0;
        }
        // postcond: key is not empty

        let root_match = match &mut (*self.root).children {
            Children::Small(v) => {
                let mut res = CommonPrefixInner::NoMatch(self.root);
                let mut i = 0;
                while i < v.len() {
                    let child = v[i];
                    if (*child).is_empty() {
                        self.lazy_gc(child);
                        v.remove(i);
                        continue;
                    }
                    if (*child).value[0] == key[0] {
                        res = self.common_prefix(child, key);
                        break;
                    }
                    i += 1;
                }
                res
            }
            Children::Large(m) => {
                let res;
                if let Some(&child) = m.get(&key[0]) {
                    if (*child).is_empty() {
                        // GC
                        self.lazy_gc(child);
                        m.remove(&key[0]);
                        res = CommonPrefixInner::NoMatch(self.root);
                    } else {
                        res = self.common_prefix(child, key);
                    }
                } else {
                    res = CommonPrefixInner::NoMatch(self.root);
                }
                res
            }
        };

        match root_match {
            CommonPrefixInner::FullMatch(node) => {
                // postcond: a first level node fully matches key
                match self.common_prefix_full_combo(node, &key[(*node).value.len()..]) {
                    (CommonPrefixInner::FullMatch(node), key1) => {
                        if key1.is_empty() {
                            // postcond: perfect match
                            return 0;
                        }
                        // postcond: unmatched key sequence
                        let full_match_len = key.len() - key1.len();
                        let remain_value = value.split_off(full_match_len);
                        // invariant: key1 |-> remain_value
                        assert_eq!(key1.len(), remain_value.len());
                        if (*node).children.is_empty() {
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
                        return key1.len();
                    }
                    (CommonPrefixInner::PartialMatch(node, common), key1) => {
                        if key1.len() == common {
                            // postcond: perfect match
                            return 0;
                        }
                        // postcond: imperfect match
                        let full_match_len = key.len() - key1.len();
                        let remain_value = value.split_off(full_match_len);
                        // invariant: key1 |-> remain_value
                        assert_eq!(key1.len(), remain_value.len());
                        self.split_node_at_insert(node, common, key1, remain_value);
                        return key1.len() - common;
                    }
                    (CommonPrefixInner::NoMatch(_), _) => {
                        unreachable!()
                    }
                }
            }
            CommonPrefixInner::PartialMatch(node, common) => {
                // postcond: a first level node partially matches key
                if key.len() == common {
                    // postcond: perfect match
                    return 0;
                }
                self.split_node_at_insert(node, common, key, value);
                return key.len() - common;
            }
            CommonPrefixInner::NoMatch(node) => {
                // postcond: zero matched first level node
                assert_eq!(node, self.root);
                self.insert_as_child(self.root, key, value);
                return key.len();
            }
        }
    }

    unsafe fn get_inner(&self, block_hashes: &[u64]) -> usize {
        let mut prefix_len = 0;
        let mut cur = self.root;

        'level_down: loop {
            match &(*cur).children {
                Children::Small(v) => {
                    if prefix_len == block_hashes.len() {
                        // perfect match: terminate
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
                            // GC child will be filtered
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
                            // GC child will be filtered
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

    unsafe fn lazy_gc(&mut self, node: *mut Node<u64, u64>) {
        drop(Box::from_raw(node));
        self.nodes.fetch_sub(1, Ordering::Relaxed);
    }

    /// Atomicaly lazy gc `node`, nonetheless, caller must ensure
    /// that some mutex is hold before calling this method
    ///
    /// # Precondition:
    ///   + `self.mtx` is held by current thread
    unsafe fn atomize_manually_lazy_gc(&self, node: *mut Node<u64, u64>) {
        drop(Box::from_raw(node));
        self.nodes.fetch_sub(1, Ordering::Relaxed);
    }

    unsafe fn clear_inner(&mut self, node: *mut Node<u64, u64>, to_drop: bool) -> usize {
        if (*node).is_empty() {
            if to_drop {
                drop(Box::from_raw(node));
                self.nodes.fetch_sub(1, Ordering::Relaxed);
            }
            return 0;
        }

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
            self.nodes.fetch_sub(1, Ordering::Relaxed);
        }

        n
    }

    unsafe fn remove_inner(&mut self, block_indices: &Vec<u64>) {
        #[cfg(debug_assertions)]
        for &bid in block_indices {
            debug_assert!(!self.block_to_node[bid as usize].is_null());
        }

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
            debug_assert!(pos < (*node).payload.len());
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
            // postcond: `node` is in valid state
            // NOTE: always keep this node, defer GC to next scan
            // it's parent's responsibility to drop child
        }

        assert_eq!(nblock_canary, block_indices.len());
    }
}

impl Drop for RadixTreeBlockHash {
    fn drop(&mut self) {
        unsafe {
            let mut stack = vec![self.root];
            let mut post = Vec::with_capacity(self.nodes.load(Ordering::Relaxed));
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

impl BlockHash for RadixTreeBlockHash {
    fn new(num_blocks: usize) -> Self {
        let root = Node::<u64, u64>::new();
        Self {
            root,
            size: 0,
            nodes: AtomicUsize::new(1),
            block_to_node: vec![null_mut(); num_blocks],
            mtx: SpinLock::new(),
        }
    }

    fn len(&self) -> usize {
        self.size
    }

    fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn insert(&mut self, block_hashes: &[u64], block_indices: Vec<u64>) -> usize {
        unsafe {
            let n = self.insert_inner(block_hashes, block_indices);
            self.size += n;
            n
        }
    }

    fn get(&self, block_hashes: &[u64]) -> usize {
        unsafe { self.get_inner(block_hashes) }
    }

    fn remove(&mut self, block_indices: Vec<u64>) {
        debug_assert!(self.size >= block_indices.len());
        unsafe {
            self.size -= block_indices.len();
            self.remove_inner(&block_indices);
        }
    }
}

mod hashtable_block_hash {
    use super::*;
    use smallvec::{smallvec, SmallVec};
    use xxhash_rust::xxh3::xxh3_64_with_seed;

    pub struct HashTableBlockHash {
        map: IntMap<u64, SmallVec<[u64; 1]>>, // hash -> block_id
        block_to_hash: Vec<Option<u64>>,      // block_id -> hash
    }

    unsafe impl Send for HashTableBlockHash {}
    unsafe impl Sync for HashTableBlockHash {}

    impl BlockHash for HashTableBlockHash {
        fn new(num_blocks: usize) -> Self {
            Self {
                map: IntMap::with_capacity_and_hasher(num_blocks, BuildNoHashHasher::default()),
                block_to_hash: vec![None; num_blocks],
            }
        }

        fn len(&self) -> usize {
            self.map.len()
        }

        fn is_empty(&self) -> bool {
            self.map.is_empty()
        }

        fn insert(&mut self, block_hashes: &[u64], block_indices: Vec<u64>) -> usize {
            debug_assert_eq!(block_hashes.len(), block_indices.len());
            let mut n = 0;
            for (h, &bid) in block_hashes.iter().zip(block_indices.iter()) {
                // precond: `bid` must be in domain
                debug_assert!((bid as usize) < self.block_to_hash.len());
                // invariant: idempotent insertion
                self.map
                    .entry(*h)
                    .and_modify(|bids| {
                        if bids.contains(&bid) {
                            debug_assert_eq!(self.block_to_hash[bid as usize].unwrap(), *h);
                        } else {
                            bids.push(bid);
                        }
                    })
                    .or_insert_with(|| {
                        #[cfg(debug_assertions)]
                        {
                            // bid ❌ old hash ✅
                            if let Some(old_hash) = self.block_to_hash[bid as usize] {
                                eprintln!(
                                    "Rust hashes: {:?}, backend bids: {:?}",
                                    block_hashes, block_indices
                                );
                                panic!(
                                    "Bid [{bid}:>({old_hash})] has been mapping to new hash ({})",
                                    *h
                                );
                            }
                        }
                        self.block_to_hash[bid as usize] = Some(*h);
                        n += 1;
                        smallvec![bid]
                    });
            }
            n
        }

        fn get(&self, block_hashes: &[u64]) -> usize {
            let mut matched = 0;
            for &h in block_hashes {
                if self.map.contains_key(&h) {
                    matched += 1;
                } else {
                    break;
                }
            }
            matched
        }

        fn remove(&mut self, block_indices: Vec<u64>) {
            for bid in block_indices {
                let idx = bid as usize;
                debug_assert!(idx < self.block_to_hash.len());
                if let Some(h) = self.block_to_hash[idx].take() {
                    // invariant: block id must be occupied
                    if let Some(bids) = self.map.get_mut(&h) {
                        // invariant: hash value must exist
                        if let Some(pos) = bids.iter().position(|&x| x == bid) {
                            bids.remove(pos);
                        } else {
                            unreachable!("bid not found in map for hash {h}");
                        }
                        if bids.is_empty() {
                            self.map.remove(&h);
                        }
                    } else {
                        debug_assert!(false);
                    }
                }
                // TODO: add some checking for "last_block_evict"
            }
        }
    }

    #[derive(Debug, Default)]
    pub(crate) struct BlockHashState {
        /// Calculated block hash values
        pub block_hashes: Vec<u64>,
        /// Materialized block index at backend
        /// NOTE: since this request is still active, the backend should preserve these blocks
        block_indices: Vec<u64>,
        /// Number of blocks that occupied at backend, and synchronised with scheduler
        sync_nblks: usize,
        pred_hit_nblks: AtomicUsize,
        real_hit_nblks: Option<usize>,
        block_size: usize,
        prev_hash: u64,
        token_in_last_block: Vec<u32>,
    }

    // Private sentinel for uninitialized `pred_hit_nblks`,
    // i.e., a MSB mask 0b1_000...000
    const NONE_SENTINEL: usize = isize::MIN as usize;

    impl BlockHashState {
        pub fn new(input_tokens: &Vec<u32>, block_size: usize) -> Self {
            let mut prev_hash = DEFAULT_BLOCK_HASH;
            let nblk = input_tokens.len().div_ceil(block_size);
            let mut block_hashes = Vec::with_capacity(nblk.next_power_of_two());
            let block_indices = Vec::with_capacity(nblk.next_power_of_two());
            let mut token_in_last_block = Vec::with_capacity(block_size);

            for block in input_tokens.chunks(block_size) {
                if block.len() < 16 {
                    token_in_last_block.extend_from_slice(block);
                    break;
                }
                let len = block.len() * std::mem::size_of::<u32>();
                let ptr = block.as_ptr() as *const u8;
                let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr, len) };
                prev_hash = xxh3_64_with_seed(bytes, prev_hash);
                block_hashes.push(prev_hash);
            }

            BlockHashState {
                block_hashes,
                block_indices,
                pred_hit_nblks: AtomicUsize::new(NONE_SENTINEL),
                real_hit_nblks: None,
                sync_nblks: 0,
                block_size,
                prev_hash,
                token_in_last_block,
            }
        }

        /// This function must be called by scheduler
        pub fn set_pred_block_hits(&self, hit_nblks: usize) {
            let x = self.pred_hit_nblks.swap(hit_nblks & !NONE_SENTINEL, Ordering::AcqRel);
            assert_eq!(x, NONE_SENTINEL);
        }

        /// This function must be called exactly once at return from Prefill
        pub fn set_real_token_hits_get_diff(&mut self, hit_token_cnt: u64) -> isize {
            assert!(hit_token_cnt as usize % self.block_size == 0);
            let block_hits = (hit_token_cnt as usize) / self.block_size;
            // NOTE: hit blocks has been synchronized by previous entries
            self.sync_nblks = self.sync_nblks.max(block_hits);
            self.real_hit_nblks = Some(block_hits);
            // `self.pred_hit_nblks.unwrap()`, a set MSB indicates uninitialized state
            let hit_nblks = self.pred_hit_nblks.load(Ordering::Acquire);
            assert!(hit_nblks & NONE_SENTINEL == 0);
            self.real_hit_nblks.unwrap() as isize - hit_nblks as isize
        }

        /// Equivalent to `append_tokens ; get_hashes_onto_indices`
        /// # Returns
        /// slice of hash values for caller to further manipulate `PrefixBlockHash`
        #[allow(unused)]
        pub fn append(
            &mut self,
            new_tokens: &[u32],
            cur_backend_bids: Vec<u64>,
            new_backend_bids: &Vec<u64>,
        ) -> Result<&[u64], &[u64]> {
            self.append_tokens(new_tokens);
            self.set_bids(cur_backend_bids);
            self.get_onto_hashes(new_backend_bids)
        }

        pub fn append_tokens(&mut self, new_tokens: &[u32]) -> Option<u64> {
            self.token_in_last_block.extend_from_slice(new_tokens);
            if self.token_in_last_block.len() >= self.block_size {
                let block = self.token_in_last_block.drain(..self.block_size).collect::<Vec<_>>();
                let len = block.len() * std::mem::size_of::<u32>();
                let ptr = block.as_ptr() as *const u8;
                let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr, len) };
                self.prev_hash = xxh3_64_with_seed(bytes, self.prev_hash);
                self.block_hashes.push(self.prev_hash);
                return Some(self.prev_hash);
            }
            None
        }

        /// Set block indices occupied at backends as backup
        /// # Arguments:
        /// `backend_bids` currently occupied blocks at backend, including prefix
        pub fn set_bids(&mut self, backend_bids: Vec<u64>) {
            assert!(backend_bids.is_empty() == false);
            // `0` is an invalid bid at backend
            let old = self.block_indices.get(self.sync_nblks).cloned().unwrap_or(0);
            let new = backend_bids.get(self.sync_nblks).cloned().unwrap_or(0);
            if (old != new) || old == 0 {
                self.sync_nblks = 0;
            }
            self.block_indices = backend_bids;
        }

        /// Get block hashes mapped onto newly occupied blocks at backend
        ///
        /// # Returns:
        /// `Ok` => onto hashes at Router
        /// `Err` => all marked occupied bids at Router
        pub fn get_onto_hashes(&mut self, new_backend_bids: &Vec<u64>) -> Result<&[u64], &[u64]> {
            // Common (fast) path
            // 0 is an invalid bid, vLLM's BlockHash uses 1-base indexing
            if self.block_indices[self.sync_nblks] == new_backend_bids.first().cloned().unwrap_or(0)
            {
                let begin = self.sync_nblks;
                let end = begin + new_backend_bids.len();
                if &self.block_indices[begin..end] == new_backend_bids.as_slice() {
                    self.sync_nblks = end;
                    return Ok(&self.block_hashes[begin..end]);
                }
            }
            // Corner (slow) path
            if let Some(begin) = self
                .block_indices
                .windows(new_backend_bids.len())
                .position(|backend_bids| backend_bids == new_backend_bids.as_slice())
            {
                let end = begin + new_backend_bids.len();
                self.sync_nblks = end;
                return Ok(&self.block_hashes[begin..end]);
            }

            Err(&self.block_indices)
        }

        pub fn get_hashes(&self) -> &[u64] {
            &self.block_hashes
        }

        pub fn get_block_size(&self) -> usize {
            self.block_size
        }
    }
}

#[cfg(feature = "hashtable-blockhash")]
pub(crate) use hashtable_block_hash::{BlockHashState, HashTableBlockHash as PrefixBlockHash};
#[cfg(feature = "radixtree-blockhash")]
pub(crate) use RadixTreeBlockHash as PrefixBlockHash;

#[cfg(test)]
mod tests {
    use super::*;
    use hashtable_block_hash::HashTableBlockHash;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    // ---------- 全局规模 ----------
    const SMALL_BLOCKS: usize = 10_000;
    const MEDIUM_BLOCKS: usize = 100_000;
    const FUZZ_BLOCKS: usize = 400_000;

    // ---------- block_id 池，双重保证 block_id < num_blocks ----------
    #[derive(Clone, Debug)]
    struct IndexPool {
        base: u64,
        cap: u64,
        cur: u64,
        limit: u64,
    }
    impl IndexPool {
        fn new(base: u64, cap: u64, limit: u64) -> Self {
            debug_assert!(base + cap <= limit, "pool must be within [0, num_blocks)");
            Self { base, cap, cur: 0, limit }
        }
        fn alloc_run(&mut self, n: usize) -> Vec<u64> {
            let n_u = n as u64;
            debug_assert!(
                self.cur + n_u <= self.cap,
                "pool exhausted; enlarge cap or reduce test size"
            );
            let start = self.base + self.cur;
            self.cur += n_u;
            let out: Vec<u64> = (0..n_u).map(|i| start + i).collect();
            for &bid in &out {
                assert!((bid as u64) < self.limit);
            }
            out
        }
    }

    // ---------- 轨道（前缀序列）模型 ----------
    #[derive(Clone, Debug)]
    struct Track {
        hashes_master: Vec<u64>, // 固定 hash 序列（某条“语义链路”）
        cur_len: usize,          // 已提交前缀长度
        cur_indices: Vec<u64>,   // 与 master 等长，前缀 [0..cur_len) 有效
        pool: IndexPool,         // 独占 block 子区间
    }
    impl Track {
        fn new(domain_base: u64, offset: u64, master_len: usize, pool: IndexPool) -> Self {
            let hashes_master: Vec<u64> =
                (0..master_len).map(|i| domain_base + offset + i as u64).collect();
            Self { hashes_master, cur_len: 0, cur_indices: vec![0; master_len], pool }
        }
        // 扩展：L -> new_len（只新增尾部 indices；插入前缀 new_len）
        fn plan_extend(&mut self, new_len: usize) -> (Vec<u64>, Vec<u64>) {
            assert!(new_len >= self.cur_len && new_len <= self.hashes_master.len());
            let add = new_len - self.cur_len;
            if add > 0 {
                let tail = self.pool.alloc_run(add);
                for (i, v) in (self.cur_len..new_len).zip(tail.into_iter()) {
                    self.cur_indices[i] = v;
                }
            }
            let ins_h = self.hashes_master[..new_len].to_vec();
            let ins_i = self.cur_indices[..new_len].to_vec();
            (ins_h, ins_i)
        }
        // 改写后缀：删 [cut..cur_len) → 设置为 new_len≥cut，重建 [cut..new_len) indices → 插入前缀 new_len
        fn plan_rewrite_suffix(
            &mut self,
            cut: usize,
            new_len: usize,
        ) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
            assert!(cut <= self.cur_len && new_len >= cut && new_len <= self.hashes_master.len());
            let remove_suffix = if self.cur_len > cut {
                self.cur_indices[cut..self.cur_len].to_vec()
            } else {
                vec![]
            };
            if new_len > cut {
                let tail = self.pool.alloc_run(new_len - cut);
                for (i, v) in (cut..new_len).zip(tail.into_iter()) {
                    self.cur_indices[i] = v;
                }
            }
            self.cur_len = new_len;
            let ins_h = self.hashes_master[..new_len].to_vec();
            let ins_i = self.cur_indices[..new_len].to_vec();
            (remove_suffix, ins_h, ins_i)
        }
        // 幂等：写已存在前缀 k
        fn plan_idem_prefix(&self, k: usize) -> (Vec<u64>, Vec<u64>) {
            assert!(k <= self.cur_len);
            (self.hashes_master[..k].to_vec(), self.cur_indices[..k].to_vec())
        }
        fn master_len(&self) -> usize {
            self.hashes_master.len()
        }
    }

    // ---------- 安全选择器，避免空范围 ----------
    fn choose_suffix_edit<R: Rng>(
        cur_len: usize,
        master_len: usize,
        delta: usize,
        rng: &mut R,
    ) -> (usize, usize) {
        debug_assert!(cur_len <= master_len);
        let cut = rng.gen_range(0..=cur_len);
        let hi = (cur_len + delta).min(master_len);
        let new_len = rng.gen_range(cut..=hi);
        (cut, new_len)
    }
    fn choose_extend_target<R: Rng>(
        cur_len: usize,
        master_len: usize,
        delta: usize,
        rng: &mut R,
    ) -> Option<usize> {
        if cur_len >= master_len {
            return None;
        }
        let lo = cur_len + 1;
        let hi = (cur_len + delta).min(master_len);
        debug_assert!(lo <= hi);
        Some(rng.gen_range(lo..=hi))
    }
    fn choose_idem_k<R: Rng>(cur_len: usize, rng: &mut R) -> Option<usize> {
        if cur_len == 0 {
            return None;
        }
        Some(rng.gen_range(1..=cur_len))
    }

    // ---------- 执行器：严格“先删后缀，再写前缀”并对拍 ----------
    fn exec_suffix_then_prefix_and_compare(
        ht: &mut HashTableBlockHash,
        rt: &mut RadixTreeBlockHash,
        remove_suffix: &[u64],
        ins_hashes: &[u64],
        ins_indices: &[u64],
    ) {
        // FIXME: `block_to_hash` is
        // for &bid in remove_suffix {
        //     assert!((bid as usize) < ht.block_to_hash.len());
        // }
        // for &bid in ins_indices {
        //     assert!((bid as usize) < ht.block_to_hash.len());
        // }

        if !remove_suffix.is_empty() {
            ht.remove(remove_suffix.to_vec());
            rt.remove(remove_suffix.to_vec());
        }
        assert_eq!(ins_hashes.len(), ins_indices.len());
        let n_ht = ht.insert(ins_hashes, ins_indices.to_vec());
        let n_rt = rt.insert(ins_hashes, ins_indices.to_vec());
        assert_eq!(n_ht, n_rt, "insert: 新增块数不一致");
        assert_eq!(ht.get(ins_hashes), rt.get(ins_hashes), "get: 前缀匹配不一致");
    }

    // ---------- 收尾：删除所有 block，并检查 RadixTree “一层空子节点 + 逻辑空” ----------
    fn teardown_clear_all_and_assert_one_layer(
        tracks: &[Track],
        ht: &mut HashTableBlockHash,
        rt: &mut RadixTreeBlockHash,
    ) {
        // 1) 删除每条 Track 已提交的整段（合法后缀：cut=0 → suffix=全部）
        for t in tracks {
            if t.cur_len > 0 {
                let all = t.cur_indices[..t.cur_len].to_vec();
                ht.remove(all.clone());
                rt.remove(all);
            }
        }
        // HashTable 的 map 应为空
        assert!(ht.is_empty(), "HashTableBlockHash should be empty after teardown");

        // 2) 逻辑空：对每条轨道的任意非空前缀，get 应为 0
        for t in tracks {
            let ml = t.master_len();
            if ml > 0 {
                let k = ml.min(5); // 检几段前缀就够
                let pref = &t.hashes_master[..k];
                assert_eq!(ht.get(pref), 0, "HT get should be 0 after teardown");
                assert_eq!(rt.get(pref), 0, "RT get should be 0 after teardown");
            }
        }

        // 3) 结构检查：RadixTree 只有一层，且 root 的每个 child 都是“空节点且无子”
        unsafe {
            // root 自身也应无 value/payload
            assert!((*rt.root).value.is_empty());
            assert!((*rt.root).payload.is_empty());

            match &(*rt.root).children {
                super::Children::Small(v) => {
                    for &c in v {
                        // child 必须是空节点
                        assert!((*c).value.is_empty(), "child.value should be empty");
                        assert!((*c).payload.is_empty(), "child.payload should be empty");
                        // 且 child 不能再有子节点
                        match &(*c).children {
                            super::Children::Small(v2) => {
                                assert!(v2.is_empty(), "grandchildren must be empty")
                            }
                            super::Children::Large(m2) => {
                                assert!(m2.is_empty(), "grandchildren must be empty")
                            }
                        }
                    }
                }
                super::Children::Large(m) => {
                    for &c in m.values() {
                        assert!((*c).value.is_empty(), "child.value should be empty");
                        assert!((*c).payload.is_empty(), "child.payload should be empty");
                        match &(*c).children {
                            super::Children::Small(v2) => {
                                assert!(v2.is_empty(), "grandchildren must be empty")
                            }
                            super::Children::Large(m2) => {
                                assert!(m2.is_empty(), "grandchildren must be empty")
                            }
                        }
                    }
                }
            }
        }
    }

    // ================== Small：手工路径 ==================
    #[test]
    fn small_prefix_suffix_suite() {
        let num_blocks = SMALL_BLOCKS as u64;
        let mut ht = HashTableBlockHash::new(SMALL_BLOCKS);
        let mut rt = RadixTreeBlockHash::new(SMALL_BLOCKS);

        // 单轨道
        let pool = IndexPool::new(2_000, 2_000, num_blocks);
        let mut t = Track::new(100_000, 0, 16, pool);

        // 1) 扩到 5
        {
            let (ins_h, ins_i) = t.plan_extend(5);
            exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &[], &ins_h, &ins_i);
            t.cur_len = 5;
        }
        // 2) 幂等前缀 3
        {
            let (ins_h, ins_i) = t.plan_idem_prefix(3);
            exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &[], &ins_h, &ins_i);
        }
        // 3) 扩到 10
        {
            let (ins_h, ins_i) = t.plan_extend(10);
            exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &[], &ins_h, &ins_i);
            t.cur_len = 10;
        }
        // 4) 改写后缀 cut=6 → new_len=10
        {
            let (rm_suf, ins_h, ins_i) = t.plan_rewrite_suffix(6, 10);
            exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &rm_suf, &ins_h, &ins_i);
        }
        // 5) 全删到空，再插 4
        {
            let rm_all = t.cur_indices[..t.cur_len].to_vec();
            exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &rm_all, &[], &[]);
            t.cur_len = 0;

            let (ins_h, ins_i) = t.plan_extend(4);
            exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &[], &ins_h, &ins_i);
            t.cur_len = 4;
        }

        // —— 收尾：删光所有 block，并检查 RadixTree 的“一层空子节点 + 逻辑空”
        teardown_clear_all_and_assert_one_layer(std::slice::from_ref(&t), &mut ht, &mut rt);
    }

    // ================== Medium：多轨道、可复现随机 ==================
    #[test]
    fn medium_batched_prefix_suffix() {
        let num_blocks = MEDIUM_BLOCKS as u64;
        let mut ht = HashTableBlockHash::new(MEDIUM_BLOCKS);
        let mut rt = RadixTreeBlockHash::new(MEDIUM_BLOCKS);

        // 8 条轨道，4 个 hash 域
        let hash_domains = [0u64, 50_000, 100_000, 150_000];
        let per_track_cap = 4_000u64;
        let mut tracks: Vec<Track> = (0..8)
            .map(|i| {
                let di = (i as usize) % hash_domains.len();
                let domain = hash_domains[di];
                let pool_base = (i as u64) * per_track_cap;
                let pool = IndexPool::new(pool_base, per_track_cap, num_blocks);
                Track::new(domain, (i as u64) * 11, 24, pool)
            })
            .collect();

        let mut rng = StdRng::seed_from_u64(0xCAFE_FEED);

        for _step in 0..120 {
            let i = rng.gen_range(0..tracks.len());
            let mut t = tracks[i].clone();

            enum Op {
                IdemPrefix(usize),
                Extend(usize),
                RewriteSuffix { cut: usize, new_len: usize },
            }

            // 构造候选，避免空范围
            let mut candidates: Vec<Op> = Vec::new();
            if let Some(k) = choose_idem_k(t.cur_len, &mut rng) {
                candidates.push(Op::IdemPrefix(k));
            }
            if let Some(target) = choose_extend_target(t.cur_len, t.master_len(), 8, &mut rng) {
                candidates.push(Op::Extend(target));
            }
            if t.cur_len > 0 {
                let (cut, new_len) = choose_suffix_edit(t.cur_len, t.master_len(), 8, &mut rng);
                candidates.push(Op::RewriteSuffix { cut, new_len });
            }

            // 至少一个候选
            let choice = rng.gen_range(0..candidates.len());
            match candidates.swap_remove(choice) {
                Op::IdemPrefix(k) => {
                    let (ins_h, ins_i) = t.plan_idem_prefix(k);
                    exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &[], &ins_h, &ins_i);
                }
                Op::Extend(target) => {
                    let (ins_h, ins_i) = t.plan_extend(target);
                    exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &[], &ins_h, &ins_i);
                    t.cur_len = target;
                }
                Op::RewriteSuffix { cut, new_len } => {
                    let (rm_suf, ins_h, ins_i) = t.plan_rewrite_suffix(cut, new_len);
                    exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &rm_suf, &ins_h, &ins_i);
                }
            }

            tracks[i] = t;
        }

        // —— 收尾：删光所有 block，并检查 RadixTree 的“一层空子节点 + 逻辑空”
        teardown_clear_all_and_assert_one_layer(&tracks, &mut ht, &mut rt);
    }

    // ================== Fuzz：更大规模，严格“删后缀/写前缀” ==================
    #[test]
    fn fuzz_prefix_suffix() {
        let num_blocks = FUZZ_BLOCKS as u64;
        let mut ht = HashTableBlockHash::new(FUZZ_BLOCKS);
        let mut rt = RadixTreeBlockHash::new(FUZZ_BLOCKS);

        let per_track_cap = 8_000u64;
        let mut tracks: Vec<Track> = (0..16)
            .map(|i| {
                let domain = (i as u64) * 200_000; // 远离，避免跨轨道 hash 冲突
                let pool_base = (i as u64) * per_track_cap;
                let pool = IndexPool::new(pool_base, per_track_cap, num_blocks);
                Track::new(domain, (i as u64) * 13, 32, pool)
            })
            .collect();

        let mut rng = StdRng::seed_from_u64(0x5EED_BEEF);

        for _ in 0..800 {
            let i = rng.gen_range(0..tracks.len());
            let mut t = tracks[i].clone();

            enum Op {
                IdemPrefix(usize),
                Extend(usize),
                RewriteSuffix { cut: usize, new_len: usize },
            }

            let mut candidates: Vec<Op> = Vec::new();
            if let Some(k) = choose_idem_k(t.cur_len, &mut rng) {
                candidates.push(Op::IdemPrefix(k));
            }
            if let Some(target) = choose_extend_target(t.cur_len, t.master_len(), 10, &mut rng) {
                candidates.push(Op::Extend(target));
            }
            if t.cur_len > 0 {
                let (cut, new_len) = choose_suffix_edit(t.cur_len, t.master_len(), 10, &mut rng);
                candidates.push(Op::RewriteSuffix { cut, new_len });
            }

            let choice = rng.gen_range(0..candidates.len());
            match candidates.swap_remove(choice) {
                Op::IdemPrefix(k) => {
                    let (ins_h, ins_i) = t.plan_idem_prefix(k);
                    exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &[], &ins_h, &ins_i);
                }
                Op::Extend(target) => {
                    let (ins_h, ins_i) = t.plan_extend(target);
                    exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &[], &ins_h, &ins_i);
                    t.cur_len = target;
                }
                Op::RewriteSuffix { cut, new_len } => {
                    let (rm_suf, ins_h, ins_i) = t.plan_rewrite_suffix(cut, new_len);
                    exec_suffix_then_prefix_and_compare(&mut ht, &mut rt, &rm_suf, &ins_h, &ins_i);
                }
            }

            tracks[i] = t;
        }

        // —— 收尾：删光所有 block，并检查 RadixTree 的“一层空子节点 + 逻辑空”
        teardown_clear_all_and_assert_one_layer(&tracks, &mut ht, &mut rt);
    }
}
