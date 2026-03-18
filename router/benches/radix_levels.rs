// Progressive lowering: Verified RadixTree → Optimized Rust
//
// L0: Safe Rust baseline (direct translation of verified_radix.rs)
//     Vec<Box<Node>>, linear child scan, remove+modify+insert
//
// L1: Sorted children + binary search
//     O(log n) child lookup instead of O(n)
//
// L2: Raw pointers instead of Box
//     Vec<*mut Node>, in-place mutation, no Box overhead
//
// L3: Small/Large children split
//     HashMap upgrade at 64 children (production pattern)

use std::collections::HashMap;
use std::mem;

// ============================================================================
// Common trait for benchmarking
// ============================================================================

pub trait RadixTree {
    fn new() -> Self;
    fn insert(&mut self, key: &[u64], payload: &[u64]);
    fn prefix_len(&self, query: &[u64]) -> usize;
}

// ============================================================================
// L0: Safe Rust baseline (Verus translation)
// ============================================================================

pub mod l0 {
    use super::RadixTree;

    struct Node {
        value: Vec<u64>,
        payload: Vec<u64>,
        children: Vec<Box<Node>>,
    }

    pub struct RadixTreeMap {
        roots: Vec<Box<Node>>,
    }

    impl RadixTree for RadixTreeMap {
        fn new() -> Self {
            RadixTreeMap { roots: Vec::new() }
        }

        fn insert(&mut self, key: &[u64], payload: &[u64]) {
            assert!(!key.is_empty());
            assert_eq!(key.len(), payload.len());
            forest_insert(&mut self.roots, key, payload, 0);
        }

        fn prefix_len(&self, query: &[u64]) -> usize {
            forest_prefix_len(&self.roots, query, 0)
        }
    }

    fn forest_prefix_len(forest: &[Box<Node>], query: &[u64], offset: usize) -> usize {
        if offset >= query.len() {
            return offset;
        }
        for child in forest.iter() {
            if !child.value.is_empty() && child.value[0] == query[offset] {
                return node_prefix_len(child, query, offset);
            }
        }
        offset
    }

    fn node_prefix_len(node: &Node, query: &[u64], offset: usize) -> usize {
        let vlen = node.value.len();
        let mut matched = 1; // first element matches by caller guarantee
        while matched < vlen && (offset + matched) < query.len() {
            if node.value[matched] != query[offset + matched] {
                break;
            }
            matched += 1;
        }
        if matched < vlen {
            offset + matched
        } else {
            forest_prefix_len(&node.children, query, offset + matched)
        }
    }

    fn make_leaf(key: &[u64], payload: &[u64], offset: usize) -> Node {
        Node {
            value: key[offset..].to_vec(),
            payload: payload[offset..].to_vec(),
            children: Vec::new(),
        }
    }

    fn forest_insert(forest: &mut Vec<Box<Node>>, key: &[u64], payload: &[u64], offset: usize) {
        let target = key[offset];
        let mut found = None;
        for (i, child) in forest.iter().enumerate() {
            if !child.value.is_empty() && child.value[0] == target {
                found = Some(i);
                break;
            }
        }

        match found {
            None => {
                forest.push(Box::new(make_leaf(key, payload, offset)));
            }
            Some(idx) => {
                // Remove, modify, reinsert (matches Verus pattern)
                let mut boxed = forest.remove(idx);
                node_insert(&mut boxed, key, payload, offset);
                forest.insert(idx, boxed);
            }
        }
    }

    fn node_insert(node: &mut Node, key: &[u64], payload: &[u64], offset: usize) {
        let vlen = node.value.len();
        let remaining = key.len() - offset;
        let limit = vlen.min(remaining);

        let mut common = 1;
        while common < limit {
            if node.value[common] != key[offset + common] {
                break;
            }
            common += 1;
        }

        if common == vlen && (offset + common) == key.len() {
            // Exact match — update payload
            for j in 0..vlen {
                node.payload[j] = payload[offset + j];
            }
        } else if common == vlen {
            // Full edge match — descend
            forest_insert(&mut node.children, key, payload, offset + common);
        } else {
            // Partial match — split
            let sv: Vec<u64> = node.value[common..].to_vec();
            let sp: Vec<u64> = node.payload[common..].to_vec();
            let mut orig_children = Vec::new();
            mem::swap(&mut node.children, &mut orig_children);

            let suffix = Node {
                value: sv,
                payload: sp,
                children: orig_children,
            };

            node.value.truncate(common);
            node.payload.truncate(common);
            node.children.push(Box::new(suffix));

            if (offset + common) < key.len() {
                let leaf = make_leaf(key, payload, offset + common);
                node.children.push(Box::new(leaf));
            }
        }
    }
}

// ============================================================================
// L1: Sorted children + binary search
// ============================================================================

pub mod l1 {
    use super::RadixTree;
    use std::mem;

    struct Node {
        value: Vec<u64>,
        payload: Vec<u64>,
        children: Vec<Box<Node>>, // sorted by children[i].value[0]
    }

    pub struct RadixTreeMap {
        roots: Vec<Box<Node>>, // sorted by roots[i].value[0]
    }

    // Binary search for child whose value[0] == target.
    // Returns Ok(idx) if found, Err(idx) for insertion point.
    #[inline]
    fn find_child(children: &[Box<Node>], target: u64) -> Result<usize, usize> {
        children.binary_search_by_key(&target, |c| c.value[0])
    }

    impl RadixTree for RadixTreeMap {
        fn new() -> Self {
            RadixTreeMap { roots: Vec::new() }
        }

        fn insert(&mut self, key: &[u64], payload: &[u64]) {
            assert!(!key.is_empty());
            assert_eq!(key.len(), payload.len());
            forest_insert(&mut self.roots, key, payload, 0);
        }

        fn prefix_len(&self, query: &[u64]) -> usize {
            forest_prefix_len(&self.roots, query, 0)
        }
    }

    fn forest_prefix_len(forest: &[Box<Node>], query: &[u64], offset: usize) -> usize {
        if offset >= query.len() {
            return offset;
        }
        match find_child(forest, query[offset]) {
            Ok(idx) => node_prefix_len(&forest[idx], query, offset),
            Err(_) => offset,
        }
    }

    fn node_prefix_len(node: &Node, query: &[u64], offset: usize) -> usize {
        let vlen = node.value.len();
        let mut matched = 1;
        while matched < vlen && (offset + matched) < query.len() {
            if node.value[matched] != query[offset + matched] {
                break;
            }
            matched += 1;
        }
        if matched < vlen {
            offset + matched
        } else {
            forest_prefix_len(&node.children, query, offset + matched)
        }
    }

    fn make_leaf(key: &[u64], payload: &[u64], offset: usize) -> Node {
        Node {
            value: key[offset..].to_vec(),
            payload: payload[offset..].to_vec(),
            children: Vec::new(),
        }
    }

    fn forest_insert(forest: &mut Vec<Box<Node>>, key: &[u64], payload: &[u64], offset: usize) {
        let target = key[offset];
        match find_child(forest, target) {
            Err(ins) => {
                forest.insert(ins, Box::new(make_leaf(key, payload, offset)));
            }
            Ok(idx) => {
                node_insert(&mut forest[idx], key, payload, offset);
            }
        }
    }

    fn node_insert(node: &mut Node, key: &[u64], payload: &[u64], offset: usize) {
        let vlen = node.value.len();
        let remaining = key.len() - offset;
        let limit = vlen.min(remaining);

        let mut common = 1;
        while common < limit {
            if node.value[common] != key[offset + common] {
                break;
            }
            common += 1;
        }

        if common == vlen && (offset + common) == key.len() {
            for j in 0..vlen {
                node.payload[j] = payload[offset + j];
            }
        } else if common == vlen {
            forest_insert(&mut node.children, key, payload, offset + common);
        } else {
            // Split
            let sv: Vec<u64> = node.value[common..].to_vec();
            let sp: Vec<u64> = node.payload[common..].to_vec();
            let mut orig_children = Vec::new();
            mem::swap(&mut node.children, &mut orig_children);

            let suffix = Node {
                value: sv,
                payload: sp,
                children: orig_children,
            };

            node.value.truncate(common);
            node.payload.truncate(common);

            // Insert suffix and possibly new leaf in sorted order
            let suffix_key = suffix.value[0];
            let suffix_box = Box::new(suffix);

            if (offset + common) < key.len() {
                let leaf = make_leaf(key, payload, offset + common);
                let leaf_key = leaf.value[0];
                let leaf_box = Box::new(leaf);

                // Insert both in sorted order
                if suffix_key < leaf_key {
                    node.children.push(suffix_box);
                    node.children.push(leaf_box);
                } else {
                    node.children.push(leaf_box);
                    node.children.push(suffix_box);
                }
            } else {
                node.children.push(suffix_box);
            }
        }
    }
}

// ============================================================================
// L2: Raw pointers instead of Box
// ============================================================================

pub mod l2 {
    use super::RadixTree;
    use std::mem;

    struct Node {
        value: Vec<u64>,
        payload: Vec<u64>,
        children: Vec<*mut Node>, // sorted by (*children[i]).value[0]
    }

    impl Node {
        fn alloc(value: Vec<u64>, payload: Vec<u64>) -> *mut Node {
            Box::into_raw(Box::new(Node {
                value,
                payload,
                children: Vec::new(),
            }))
        }
    }

    pub struct RadixTreeMap {
        roots: Vec<*mut Node>,
    }

    // SAFETY: Nodes are single-owner via raw pointers, no aliasing.
    unsafe impl Send for RadixTreeMap {}
    unsafe impl Sync for RadixTreeMap {}

    impl Drop for RadixTreeMap {
        fn drop(&mut self) {
            for &ptr in &self.roots {
                unsafe { free_tree(ptr) };
            }
        }
    }

    unsafe fn free_tree(node: *mut Node) {
        if node.is_null() {
            return;
        }
        for &child in &(*node).children {
            free_tree(child);
        }
        drop(Box::from_raw(node));
    }

    #[inline]
    unsafe fn find_child(children: &[*mut Node], target: u64) -> Result<usize, usize> {
        children.binary_search_by_key(&target, |&c| (*c).value[0])
    }

    impl RadixTree for RadixTreeMap {
        fn new() -> Self {
            RadixTreeMap { roots: Vec::new() }
        }

        fn insert(&mut self, key: &[u64], payload: &[u64]) {
            assert!(!key.is_empty());
            assert_eq!(key.len(), payload.len());
            unsafe { forest_insert(&mut self.roots, key, payload, 0) };
        }

        fn prefix_len(&self, query: &[u64]) -> usize {
            unsafe { forest_prefix_len(&self.roots, query, 0) }
        }
    }

    unsafe fn forest_prefix_len(forest: &[*mut Node], query: &[u64], offset: usize) -> usize {
        if offset >= query.len() {
            return offset;
        }
        match find_child(forest, query[offset]) {
            Ok(idx) => node_prefix_len(forest[idx], query, offset),
            Err(_) => offset,
        }
    }

    unsafe fn node_prefix_len(node: *mut Node, query: &[u64], offset: usize) -> usize {
        let n = &*node;
        let vlen = n.value.len();
        let mut matched = 1;
        while matched < vlen && (offset + matched) < query.len() {
            if n.value[matched] != query[offset + matched] {
                break;
            }
            matched += 1;
        }
        if matched < vlen {
            offset + matched
        } else {
            forest_prefix_len(&n.children, query, offset + matched)
        }
    }

    unsafe fn forest_insert(
        forest: &mut Vec<*mut Node>,
        key: &[u64],
        payload: &[u64],
        offset: usize,
    ) {
        let target = key[offset];
        match find_child(forest, target) {
            Err(ins) => {
                let node = Node::alloc(
                    key[offset..].to_vec(),
                    payload[offset..].to_vec(),
                );
                forest.insert(ins, node);
            }
            Ok(idx) => {
                // In-place mutation via raw pointer (no remove+insert dance)
                node_insert(forest[idx], key, payload, offset);
            }
        }
    }

    unsafe fn node_insert(
        node: *mut Node,
        key: &[u64],
        payload: &[u64],
        offset: usize,
    ) {
        let n = &mut *node;
        let vlen = n.value.len();
        let remaining = key.len() - offset;
        let limit = vlen.min(remaining);

        let mut common = 1;
        while common < limit {
            if n.value[common] != key[offset + common] {
                break;
            }
            common += 1;
        }

        if common == vlen && (offset + common) == key.len() {
            // Exact match — update payload in-place
            for j in 0..vlen {
                n.payload[j] = payload[offset + j];
            }
        } else if common == vlen {
            // Full edge match — descend
            forest_insert(&mut n.children, key, payload, offset + common);
        } else {
            // Partial match — split in-place
            let suffix_value = n.value.split_off(common);
            let suffix_payload = n.payload.split_off(common);
            let mut orig_children = Vec::new();
            mem::swap(&mut n.children, &mut orig_children);

            let suffix = Box::into_raw(Box::new(Node {
                value: suffix_value,
                payload: suffix_payload,
                children: orig_children,
            }));

            if (offset + common) < key.len() {
                let leaf = Node::alloc(
                    key[offset + common..].to_vec(),
                    payload[offset + common..].to_vec(),
                );
                // Insert both in sorted order
                let suffix_key = (*suffix).value[0];
                let leaf_key = (*leaf).value[0];
                if suffix_key < leaf_key {
                    n.children.push(suffix);
                    n.children.push(leaf);
                } else {
                    n.children.push(leaf);
                    n.children.push(suffix);
                }
            } else {
                n.children.push(suffix);
            }
        }
    }
}

// ============================================================================
// L3: Small/Large children split (production pattern)
// ============================================================================

pub mod l3 {
    use super::RadixTree;
    use std::collections::HashMap;
    use std::mem;

    const SMALL_MAX: usize = 64;

    enum Children {
        Small(Vec<*mut Node>), // sorted by value[0] when len <= SMALL_MAX
        Large(HashMap<u64, *mut Node>),
    }

    impl Children {
        fn new() -> Self {
            Children::Small(Vec::new())
        }
    }

    struct Node {
        value: Vec<u64>,
        payload: Vec<u64>,
        children: Children,
    }

    impl Node {
        fn alloc(value: Vec<u64>, payload: Vec<u64>) -> *mut Node {
            Box::into_raw(Box::new(Node {
                value,
                payload,
                children: Children::new(),
            }))
        }
    }

    pub struct RadixTreeMap {
        roots: Children,
    }

    unsafe impl Send for RadixTreeMap {}
    unsafe impl Sync for RadixTreeMap {}

    impl Drop for RadixTreeMap {
        fn drop(&mut self) {
            unsafe { free_children(&mut self.roots) };
        }
    }

    unsafe fn free_tree(node: *mut Node) {
        if node.is_null() {
            return;
        }
        free_children(&mut (*node).children);
        drop(Box::from_raw(node));
    }

    unsafe fn free_children(children: &mut Children) {
        match children {
            Children::Small(v) => {
                for &ptr in v.iter() {
                    free_tree(ptr);
                }
            }
            Children::Large(m) => {
                for (_, &ptr) in m.iter() {
                    free_tree(ptr);
                }
            }
        }
    }

    #[inline]
    unsafe fn find_child(children: &Children, target: u64) -> Option<*mut Node> {
        match children {
            Children::Small(v) => {
                match v.binary_search_by_key(&target, |&c| (*c).value[0]) {
                    Ok(idx) => Some(v[idx]),
                    Err(_) => None,
                }
            }
            Children::Large(m) => m.get(&target).copied(),
        }
    }

    #[inline]
    unsafe fn insert_child(children: &mut Children, node: *mut Node) {
        let key = (*node).value[0];
        match children {
            Children::Small(v) => {
                if v.len() >= SMALL_MAX {
                    // Upgrade to Large
                    let mut map = HashMap::with_capacity(v.len() + 1);
                    for &ptr in v.iter() {
                        map.insert((*ptr).value[0], ptr);
                    }
                    map.insert(key, node);
                    *children = Children::Large(map);
                } else {
                    match v.binary_search_by_key(&key, |&c| (*c).value[0]) {
                        Ok(idx) => v[idx] = node, // replace
                        Err(idx) => v.insert(idx, node),
                    }
                }
            }
            Children::Large(m) => {
                m.insert(key, node);
            }
        }
    }

    unsafe fn take_children(children: &mut Children) -> Children {
        let mut taken = Children::new();
        mem::swap(children, &mut taken);
        taken
    }

    impl RadixTree for RadixTreeMap {
        fn new() -> Self {
            RadixTreeMap {
                roots: Children::new(),
            }
        }

        fn insert(&mut self, key: &[u64], payload: &[u64]) {
            assert!(!key.is_empty());
            assert_eq!(key.len(), payload.len());
            unsafe { children_insert(&mut self.roots, key, payload, 0) };
        }

        fn prefix_len(&self, query: &[u64]) -> usize {
            unsafe { children_prefix_len(&self.roots, query, 0) }
        }
    }

    unsafe fn children_prefix_len(
        children: &Children,
        query: &[u64],
        offset: usize,
    ) -> usize {
        if offset >= query.len() {
            return offset;
        }
        match find_child(children, query[offset]) {
            Some(node) => node_prefix_len(node, query, offset),
            None => offset,
        }
    }

    unsafe fn node_prefix_len(node: *mut Node, query: &[u64], offset: usize) -> usize {
        let n = &*node;
        let vlen = n.value.len();
        let mut matched = 1;
        while matched < vlen && (offset + matched) < query.len() {
            if n.value[matched] != query[offset + matched] {
                break;
            }
            matched += 1;
        }
        if matched < vlen {
            offset + matched
        } else {
            children_prefix_len(&n.children, query, offset + matched)
        }
    }

    unsafe fn children_insert(
        children: &mut Children,
        key: &[u64],
        payload: &[u64],
        offset: usize,
    ) {
        let target = key[offset];
        match find_child(children, target) {
            None => {
                let node = Node::alloc(
                    key[offset..].to_vec(),
                    payload[offset..].to_vec(),
                );
                insert_child(children, node);
            }
            Some(node) => {
                node_insert(node, children, key, payload, offset);
            }
        }
    }

    unsafe fn node_insert(
        node: *mut Node,
        _parent_children: &mut Children,
        key: &[u64],
        payload: &[u64],
        offset: usize,
    ) {
        let n = &mut *node;
        let vlen = n.value.len();
        let remaining = key.len() - offset;
        let limit = vlen.min(remaining);

        let mut common = 1;
        while common < limit {
            if n.value[common] != key[offset + common] {
                break;
            }
            common += 1;
        }

        if common == vlen && (offset + common) == key.len() {
            for j in 0..vlen {
                n.payload[j] = payload[offset + j];
            }
        } else if common == vlen {
            children_insert(&mut n.children, key, payload, offset + common);
        } else {
            // Split in-place
            let suffix_value = n.value.split_off(common);
            let suffix_payload = n.payload.split_off(common);
            let orig_children = take_children(&mut n.children);

            let suffix = Box::into_raw(Box::new(Node {
                value: suffix_value,
                payload: suffix_payload,
                children: orig_children,
            }));

            insert_child(&mut n.children, suffix);

            if (offset + common) < key.len() {
                let leaf = Node::alloc(
                    key[offset + common..].to_vec(),
                    payload[offset + common..].to_vec(),
                );
                insert_child(&mut n.children, leaf);
            }
        }
    }
}
