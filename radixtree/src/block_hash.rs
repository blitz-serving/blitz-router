//! Production `RadixTreeBlockHash` — the L3-lowered, concurrent, multi-bid
//! specialization used by the router's KV-cache prefix matcher.
//!
//! Why this specialization (vs. the verified L0 in [`super::verified`]):
//!
//! - `V = Bids` (`SmallVec<[u64; 1]>`) tracks multiple backend block ids
//!   that hash to the same prefix position. This handles the chunked-prefill
//!   duplicate-prefix phenomenon: two requests in the same scheduler step
//!   prefilling the same prefix hash to identical values yet receive distinct
//!   backend block_ids — both must be tracked or eviction of the primary
//!   truncates a still-live trie path (the dynamo-q under-prediction bug).
//! - `mtx: SpinLock` + `nodes: AtomicUsize` allow concurrent readers (e.g.
//!   simulator queries) alongside the writer (completion event loop).
//! - `epoch: u64` exposes a monotonic counter so external observers can
//!   detect drift between scheduler decisions and trie state.

use crate::core::{Children, CommonPrefixInner, Node, SMALL_MAX};
use nohash_hasher::{self, BuildNoHashHasher, IntMap};
use smallvec::{smallvec, SmallVec};

use std::mem::{self};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicUsize, Ordering};

use core::hint::spin_loop;
use std::sync::atomic::AtomicBool;
use std::thread::yield_now;

/// Per-position bid container in `RadixTreeBlockHash`. Inline storage for the
/// common case (one bid per hash position); spills to heap only when multiple
/// requests in the same step concurrently prefill the same prefix and end up
/// allocating distinct backend block_ids that hash to the same value (a real
/// vLLM/yaullm chunked-prefill phenomenon — see `insert` in this module).
pub type Bids = SmallVec<[u64; 1]>;

pub(crate) struct SpinLock {
    flag: AtomicBool, // false: unlocked, true: locked
}

unsafe impl Send for SpinLock {}
unsafe impl Sync for SpinLock {}

#[allow(unused)]
impl SpinLock {
    pub const fn new() -> Self {
        Self { flag: AtomicBool::new(false) }
    }

    /// Blocking spinlock
    pub fn lock(&self) {
        let mut spins = 0u32;
        loop {
            while self.flag.load(Ordering::Relaxed) {
                spins = spinlock_backoff(spins);
            }
            match self.flag.compare_exchange(
                false,
                true,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => {
                    spins = spinlock_backoff(spins);
                }
            }
        }
    }

    #[inline]
    fn unlock(&self) {
        self.flag.store(false, Ordering::Release);
    }
}

#[inline]
fn spinlock_backoff(spins: u32) -> u32 {
    if spins < 64 {
        spin_loop();
        spins + 1
    } else {
        if spins & 0xF == 0 {
            yield_now();
        } else {
            spin_loop();
        }
        spins.saturating_add(1)
    }
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
    /// Monotonic epoch counter, incremented on every insert/remove
    fn epoch(&self) -> u64;
}

pub struct RadixTreeBlockHash {
    root: *mut Node<u64, Bids>,
    size: usize,
    nodes: AtomicUsize,
    block_to_node: Vec<*mut Node<u64, Bids>>,
    mtx: SpinLock,
    epoch: u64,
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
        node: *mut Node<u64, Bids>,
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
        (*insert_new_node).payload =
            block_indices.split_off(common).into_iter().map(|b| smallvec![b]).collect();
        // After `split_off(common)`, `block_indices` retains its first `common`
        // elements — these are the bids the caller supplied for the matched
        // prefix (positions 0..common in `node`'s pre-split payload, now `node`
        // post-split). They must be aliased into `node`'s payload so a later
        // remove(bid) can find them; otherwise they leak (the dynamo-q
        // duplicate-prefix bug).
        for (i, &alias_bid) in block_indices.iter().enumerate() {
            if self.block_to_node[alias_bid as usize].is_null() {
                (*node).payload[i].push(alias_bid);
                self.block_to_node[alias_bid as usize] = node;
            }
        }

        // update the node to valid state
        // (*node).value has been split by `split_off`
        // (*node).payload has been split by `split_off`
        (*node).children = Children::Small(vec![split_new_node, insert_new_node]);

        // validate block to node
        for bids in (*split_new_node).payload.iter() {
            for &block_id in bids.iter() {
                assert_eq!(
                    self.block_to_node[block_id as usize],
                    node,
                    "block#{} |-> {:?}",
                    block_id,
                    (*self.block_to_node[block_id as usize]).value
                );
                self.block_to_node[block_id as usize] = split_new_node;
            }
        }
        for bids in (*insert_new_node).payload.iter() {
            for &block_id in bids.iter() {
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
        subroot: *mut Node<u64, Bids>,
        key: &'b [u64],
    ) -> (CommonPrefixInner<u64, Bids>, &'b [u64]) {
        if key.is_empty() {
            return (CommonPrefixInner::FullMatch(subroot as *mut Node<u64, Bids>), key);
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
                return (CommonPrefixInner::FullMatch(subroot as *mut Node<u64, Bids>), key);
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
                    return (CommonPrefixInner::FullMatch(subroot as *mut Node<u64, Bids>), key);
                }
            }
        }
    }

    /// Inserts a new node as direct child of `node`
    unsafe fn insert_as_child(
        &mut self,
        node: *mut Node<u64, Bids>,
        block_hashes: &[u64],
        block_indices: Vec<u64>,
    ) {
        // create new node
        let new_node = Node::new();
        self.nodes.fetch_add(1, Ordering::Relaxed);
        (*new_node).value = block_hashes.to_vec();
        (*new_node).payload = block_indices.into_iter().map(|b| smallvec![b]).collect();
        // add sequence to the children of current node
        match &mut (*node).children {
            Children::Small(v) => {
                v.push(new_node);
                if v.len() > SMALL_MAX {
                    // (NOTE) precond: NOT empty((*n).value)
                    //   + this is beacuse there are 16 children, and empty children can't exist,
                    //     it will be absorbed by its parent
                    let tmp: IntMap<u64, *mut Node<u64, Bids>> =
                        v.iter().map(|&n| ((&(*n).value)[0], n)).collect();
                    let _ = mem::replace(&mut (*node).children, Children::<u64, Bids>::Large(tmp));
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
        for bids in (*new_node).payload.iter() {
            for &block_id in bids.iter() {
                assert_eq!(self.block_to_node[block_id as usize], null_mut());
                self.block_to_node[block_id as usize] = new_node;
            }
        }
    }

    /// Walk the trie path corresponding to `key`, registering each
    /// `aliases[i]` as an alias bid at the matching (node, position).
    ///
    /// Precondition: caller has verified the trie already contains `key` as a
    /// complete prefix path of length `key.len()`. Aliases that are already
    /// registered (block_to_node not null) are idempotently skipped.
    ///
    /// This is the corrective path for the chunked-prefill duplicate-prefix
    /// case: when two requests in the same scheduler step both prefill the
    /// same prefix and yaullm allocates distinct backend block_ids that hash
    /// to the same (chained) values, both bids must end up tracked in the
    /// trie payload's SmallVec — otherwise eviction of the primary bid would
    /// truncate the trie even though yaullm still holds the prefix via the
    /// alias bid, causing router under-prediction.
    unsafe fn alias_path(&mut self, key: &[u64], aliases: &[u64]) {
        debug_assert_eq!(key.len(), aliases.len());
        if key.is_empty() {
            return;
        }
        let mut cur = self.root;
        let mut consumed = 0;
        while consumed < key.len() {
            let child_opt = match &(*cur).children {
                Children::Small(v) => v.iter().copied().find(|&n| {
                    !(*n).value.is_empty() && (*n).value[0] == key[consumed]
                }),
                Children::Large(m) => m.get(&key[consumed]).copied(),
            };
            let Some(child) = child_opt else {
                debug_assert!(false, "alias_path: trie path missing for key prefix");
                return;
            };
            let val_len = (*child).value.len();
            let mut i = 0;
            while i < val_len && consumed < key.len() && (*child).value[i] == key[consumed] {
                let alias_bid = aliases[consumed];
                if self.block_to_node[alias_bid as usize].is_null() {
                    (*child).payload[i].push(alias_bid);
                    self.block_to_node[alias_bid as usize] = child;
                }
                i += 1;
                consumed += 1;
            }
            if consumed >= key.len() {
                break;
            }
            cur = child;
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
                    if (&(*child).value)[0] == key[0] {
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
                    (CommonPrefixInner::FullMatch(_node), key1) => {
                        if key1.is_empty() {
                            // postcond: perfect match — register `value` as
                            // aliases at every existing position along the path
                            self.alias_path(key, &value);
                            return 0;
                        }
                        // postcond: unmatched key suffix; matched prefix
                        // (value[0..full_match_len]) must be aliased before we
                        // truncate `value` via split_off.
                        let full_match_len = key.len() - key1.len();
                        let alias_value = value[..full_match_len].to_vec();
                        let remain_value = value.split_off(full_match_len);
                        // invariant: key1 |-> remain_value
                        assert_eq!(key1.len(), remain_value.len());
                        self.alias_path(&key[..full_match_len], &alias_value);
                        // Re-resolve `node` since alias_path may have walked deeper
                        // than the originally matched outer node; we need the
                        // tail-most node where the extension attaches. The
                        // common_prefix_full_combo result `node` is the right
                        // one — we just trust it (alias_path doesn't relocate).
                        let node = _node;
                        if (*node).children.is_empty() {
                            (*node).value.extend_from_slice(key1);
                            for &b in remain_value.iter() {
                                (*node).payload.push(smallvec![b]);
                            }
                            // validate block to node map for the newly extended
                            // suffix only (existing positions were already valid)
                            let new_start = (*node).payload.len() - remain_value.len();
                            for bids in (*node).payload[new_start..].iter() {
                                for &block_id in bids.iter() {
                                    assert!(
                                        self.block_to_node[block_id as usize].is_null()
                                            || self.block_to_node[block_id as usize] == node
                                    );
                                    self.block_to_node[block_id as usize] = node;
                                }
                            }
                        } else {
                            self.insert_as_child(node, key1, remain_value);
                        }
                        return key1.len();
                    }
                    (CommonPrefixInner::PartialMatch(node, common), key1) => {
                        if key1.len() == common {
                            // postcond: perfect match — alias `value` along path
                            self.alias_path(key, &value);
                            return 0;
                        }
                        // postcond: imperfect match; the matched prefix
                        // (value[0..key.len()-key1.len()]) plus the partial
                        // match within `node` (value at positions
                        // [full_match_len .. full_match_len+common]) must be
                        // aliased before split_node_at_insert discards them.
                        let full_match_len = key.len() - key1.len();
                        let alias_value = value[..full_match_len].to_vec();
                        let remain_value = value.split_off(full_match_len);
                        // invariant: key1 |-> remain_value
                        assert_eq!(key1.len(), remain_value.len());
                        self.alias_path(&key[..full_match_len], &alias_value);
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
                    // postcond: perfect match — alias `value` along path
                    self.alias_path(key, &value);
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
                        while i < (&(*child).value).len()
                            && prefix_len + i < block_hashes.len()
                            && (&(*child).value)[i] == block_hashes[prefix_len + i]
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
                        while i < (&(*child).value).len()
                            && prefix_len + i < block_hashes.len()
                            && (&(*child).value)[i] == block_hashes[prefix_len + i]
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
        node: *const Node<u64, Bids>,
        key: &[u64],
    ) -> CommonPrefixInner<u64, Bids> {
        let mut i = 0;
        unsafe {
            while i < (&(*node).value).len() && i < key.len() && (&(*node).value)[i] == key[i] {
                debug_assert!(
                    (&(*node).payload)[i].iter().all(|&b| {
                        self.block_to_node[b as usize] == node as *mut Node<u64, Bids>
                    })
                );
                i += 1;
            }
            if i == 0 {
                return CommonPrefixInner::NoMatch(node as *mut Node<u64, Bids>);
            } else if i == (*node).value.len() {
                return CommonPrefixInner::FullMatch(node as *mut Node<u64, Bids>);
            } else {
                return CommonPrefixInner::PartialMatch(node as *mut Node<u64, Bids>, i);
            }
        }
    }

    unsafe fn lazy_gc(&mut self, node: *mut Node<u64, Bids>) {
        drop(Box::from_raw(node));
        self.nodes.fetch_sub(1, Ordering::Relaxed);
    }

    /// Atomicaly lazy gc `node`, nonetheless, caller must ensure
    /// that some mutex is hold before calling this method
    ///
    /// # Precondition:
    ///   + `self.mtx` is held by current thread
    unsafe fn atomize_manually_lazy_gc(&self, node: *mut Node<u64, Bids>) {
        drop(Box::from_raw(node));
        self.nodes.fetch_sub(1, Ordering::Relaxed);
    }

    unsafe fn clear_inner(&mut self, node: *mut Node<u64, Bids>, to_drop: bool) -> usize {
        if (*node).is_empty() {
            if to_drop {
                drop(Box::from_raw(node));
                self.nodes.fetch_sub(1, Ordering::Relaxed);
            }
            return 0;
        }

        let mut n = 0;
        for bids in (*node).payload.iter() {
            for &block_id in bids.iter() {
                self.block_to_node[block_id as usize] = null_mut();
                n += 1;
            }
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

    unsafe fn remove_inner(&mut self, block_indices: &Vec<u64>) -> usize {
        let mut nblock_canary: usize = 0;

        for &block_id in block_indices.iter().rev() {
            let node = self.block_to_node[block_id as usize];
            if node.is_null() {
                continue;
            }
            // precond: `node` is not nil
            // precond: `block_id` must exist in some `payload[pos]` of `node`
            let mut pos = 0;
            while pos < (*node).payload.len() && !(*node).payload[pos].contains(&block_id) {
                pos += 1;
            }
            debug_assert!(pos < (*node).payload.len());

            // Remove this specific bid from the SmallVec at pos.
            let bids_at_pos = &mut (*node).payload[pos];
            let idx = bids_at_pos.iter().position(|&b| b == block_id).unwrap();
            bids_at_pos.remove(idx);
            self.block_to_node[block_id as usize] = null_mut();
            nblock_canary += 1;

            // If aliases still occupy this position, keep the position alive.
            // The trie path to this position is still valid in yaullm's view
            // (since the aliased bids still cover this hash), so we must not
            // truncate. This is the fix for the dynamo-q under-prediction:
            // previously the trie had no concept of aliases and any bid eviction
            // would truncate the suffix even when concurrent-prefill duplicates
            // were still alive on the engine.
            if !bids_at_pos.is_empty() {
                continue;
            }
            // Otherwise, truncate from pos onward and cascade-evict children.
            (*node).value.truncate(pos);
            let payload_to_rm = (*node).payload.split_off(pos);
            for tail_bids in payload_to_rm {
                for tail_bid in tail_bids {
                    self.block_to_node[tail_bid as usize] = null_mut();
                    nblock_canary += 1;
                }
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

        debug_assert!(nblock_canary <= block_indices.len());
        nblock_canary
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
        let root = Node::<u64, Bids>::new();
        Self {
            root,
            size: 0,
            nodes: AtomicUsize::new(1),
            block_to_node: vec![null_mut(); num_blocks],
            mtx: SpinLock::new(),
            epoch: 0,
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
            self.epoch += 1;
            n
        }
    }

    fn get(&self, block_hashes: &[u64]) -> usize {
        unsafe { self.get_inner(block_hashes) }
    }

    fn remove(&mut self, block_indices: Vec<u64>) {
        self.epoch += 1;
        unsafe {
            let removed = self.remove_inner(&block_indices);
            self.size = self.size.saturating_sub(removed);
        }
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }
}

impl RadixTreeBlockHash {
    /// Test-only structural invariant: after fully draining all bids, the
    /// tree should consist of root + at most one layer of empty placeholder
    /// children (lazy-GC residue) with no live keys / payloads anywhere.
    /// Exposed across the crate boundary so the integration tests in
    /// `router::kvcache::tests` can keep using it after the radix code moved
    /// out of `router/`. Not part of the supported public API — hidden from
    /// rustdoc; do not call from production paths.
    #[doc(hidden)]
    pub fn assert_drained_to_one_layer(&self) {
        unsafe {
            assert!((*self.root).value.is_empty(), "root.value should be empty");
            assert!((*self.root).payload.is_empty(), "root.payload should be empty");

            let walk_child = |c: *mut Node<u64, Bids>| {
                assert!((*c).value.is_empty(), "child.value should be empty");
                assert!((*c).payload.is_empty(), "child.payload should be empty");
                match &(*c).children {
                    Children::Small(v2) => {
                        assert!(v2.is_empty(), "grandchildren must be empty")
                    }
                    Children::Large(m2) => {
                        assert!(m2.is_empty(), "grandchildren must be empty")
                    }
                }
            };

            match &(*self.root).children {
                Children::Small(v) => {
                    for &c in v {
                        walk_child(c);
                    }
                }
                Children::Large(m) => {
                    for &c in m.values() {
                        walk_child(c);
                    }
                }
            }
        }
    }
}

// Suppress unused import warning when `BuildNoHashHasher` ends up only
// referenced through `Children<K, V>`'s default type parameter.
#[allow(unused_imports)]
use BuildNoHashHasher as _;
