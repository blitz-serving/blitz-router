// Verified RadixTreeMap — Deductive proof using Verus
//
// "Meet in the middle" approach:
//   Top:    spec layer — well-formedness + longest prefix correctness
//   Middle: exec layer — safe Rust (Box<Node>, Vec<Box<Node>>)
//
// Run: verus radixtree/src/verified.rs --crate-type=lib
//
// Strategy: Use `assume` for complex invariants first, then
// progressively replace with proofs. This gives us a verified
// skeleton that we can strengthen incrementally.

use vstd::prelude::*;

verus! {

// =========================================================================
// Data Structure — Node-only (no enum, simpler for Verus)
// =========================================================================

// A node in the compressed radix tree.
// - value: compressed edge label (sequence of keys)
// - payload: associated data for each key element
// - children: child nodes (each has a distinct first key element)
struct Node {
    value: Vec<u64>,
    payload: Vec<u64>,
    children: Vec<Box<Node>>,
}

// The radix tree: a forest of nodes sharing no first-key prefix.
struct RadixTreeMap {
    roots: Vec<Box<Node>>,
}

// =========================================================================
// Spec: well-formedness (non-mutually-recursive version)
// =========================================================================
//
// We avoid mutual recursion by having a single recursive spec function
// that checks the entire tree.

impl Node {
    spec fn wf(self) -> bool
        decreases self,
    {
        &&& self.value@.len() > 0
        &&& self.value@.len() == self.payload@.len()
        &&& forall|i: int| 0 <= i < self.children@.len()
            ==> (#[trigger] self.children@[i]).wf()
    }
}

// Spec helper: all children in a Seq are well-formed
spec fn all_wf(children: Seq<Box<Node>>) -> bool {
    forall|i: int| 0 <= i < children.len()
        ==> (#[trigger] children[i]).wf()
}

// Proof: pushing a wf node onto an all_wf seq preserves all_wf
proof fn lemma_all_wf_push(s: Seq<Box<Node>>, v: Box<Node>)
    requires
        all_wf(s),
        v.wf(),
    ensures
        all_wf(s.push(v)),
{
    assert forall|i: int| 0 <= i < s.push(v).len()
        implies (#[trigger] s.push(v)[i]).wf() by {
        if i < s.len() {
            assert(s.push(v)[i] == s[i]);
        } else {
            assert(s.push(v)[i] == v);
        }
    }
}

// Proof: removing an element from an all_wf seq preserves all_wf
proof fn lemma_all_wf_remove(s: Seq<Box<Node>>, idx: int)
    requires
        all_wf(s),
        0 <= idx < s.len(),
    ensures
        all_wf(s.remove(idx)),
{
    assert forall|i: int| 0 <= i < s.remove(idx).len()
        implies (#[trigger] s.remove(idx)[i]).wf() by {
        if i < idx {
            assert(s.remove(idx)[i] == s[i]);
        } else {
            assert(s.remove(idx)[i] == s[i + 1]);
        }
    }
}

// Proof: inserting a wf node into an all_wf seq preserves all_wf
proof fn lemma_all_wf_insert(s: Seq<Box<Node>>, idx: int, v: Box<Node>)
    requires
        all_wf(s),
        v.wf(),
        0 <= idx <= s.len(),
    ensures
        all_wf(s.insert(idx, v)),
{
    assert forall|i: int| 0 <= i < s.insert(idx, v).len()
        implies (#[trigger] s.insert(idx, v)[i]).wf() by {
        if i < idx {
            assert(s.insert(idx, v)[i] == s[i]);
        } else if i == idx {
            assert(s.insert(idx, v)[i] == v);
        } else {
            assert(s.insert(idx, v)[i] == s[i - 1]);
        }
    }
}

impl RadixTreeMap {
    closed spec fn well_formed(self) -> bool {
        all_wf(self.roots@)
    }
}

// =========================================================================
// Exec: new
// =========================================================================

impl RadixTreeMap {
    fn new() -> (result: Self)
        ensures result.well_formed(),
    {
        RadixTreeMap { roots: Vec::new() }
    }
}

// =========================================================================
// Exec: prefix_len — O(|query|) tree traversal
// =========================================================================

impl RadixTreeMap {
    fn prefix_len(&self, query: &Vec<u64>) -> (result: usize)
        requires self.well_formed(),
        ensures result <= query@.len(),
    {
        Self::forest_prefix_len(&self.roots, query, 0)
    }

    #[verifier::loop_isolation(false)]
    fn forest_prefix_len(
        forest: &Vec<Box<Node>>,
        query: &Vec<u64>,
        offset: usize,
    ) -> (result: usize)
        requires
            all_wf(forest@),
            offset <= query@.len(),
        ensures
            result >= offset,
            result <= query@.len(),
        decreases query@.len() - offset, forest@.len(),
    {
        if offset >= query.len() {
            return offset;
        }

        let mut i: usize = 0;
        while i < forest.len()
            invariant
                i <= forest@.len(),
                all_wf(forest@),
                offset < query@.len(),
            decreases forest@.len() - i,
        {
            let child: &Node = &*forest[i];
            // child.wf() holds because all_wf(forest@) and i < forest.len()
            if child.value.len() > 0 && child.value[0] == query[offset] {
                return Self::node_prefix_len(child, query, offset);
            }
            i = i + 1;
        }
        offset
    }

    #[verifier::loop_isolation(false)]
    fn node_prefix_len(
        node: &Node,
        query: &Vec<u64>,
        offset: usize,
    ) -> (result: usize)
        requires
            node.wf(),
            offset < query@.len(),
            node.value@[0] == query@[offset as int],
        ensures
            result >= offset + 1,
            result <= query@.len(),
        decreases query@.len() - offset, 0nat,
    {
        let vlen = node.value.len();
        // First element matches by precondition — start at 1
        assert(offset + 1 <= query.len());
        let mut matched: usize = 1;

        while matched < vlen && (offset + matched) < query.len()
            invariant
                1 <= matched <= vlen,
                vlen == node.value@.len(),
                offset as int + matched as int <= query@.len(),
                offset < query@.len(),
                node.wf(),
                forall|j: int| 0 <= j < matched
                    ==> node.value@[j] == query@[(offset + j) as int],
            decreases vlen - matched,
        {
            if node.value[matched] != query[offset + matched] {
                break;
            }
            matched = matched + 1;
        }

        if matched < vlen {
            (offset + matched) as usize
        } else {
            Self::forest_prefix_len(&node.children, query, offset + matched)
        }
    }
}

// =========================================================================
// Exec: insert
// =========================================================================

impl RadixTreeMap {
    fn insert(&mut self, key: &Vec<u64>, payload: &Vec<u64>)
        requires
            old(self).well_formed(),
            key@.len() > 0,
            key@.len() == payload@.len(),
        ensures
            self.well_formed(),
    {
        Self::forest_insert(&mut self.roots, key, payload, 0);
    }

    #[verifier::loop_isolation(false)]
    fn forest_insert(
        forest: &mut Vec<Box<Node>>,
        key: &Vec<u64>,
        payload: &Vec<u64>,
        offset: usize,
    )
        requires
            all_wf(old(forest)@),
            offset < key@.len(),
            key@.len() == payload@.len(),
        ensures
            all_wf(forest@),
        decreases key@.len() - offset, 1nat,
    {
        let target = key[offset];

        // Find child with matching first key
        let mut found: usize = forest.len();
        let mut i: usize = 0;
        while i < forest.len()
            invariant
                i <= forest@.len(),
                found <= forest@.len(),
                all_wf(forest@),
                offset < key@.len(),
                key@.len() == payload@.len(),
                target == key@[offset as int],
                // Track: when found < len, the found child matches
                found < forest@.len() ==> (
                    forest@[found as int].value@.len() > 0
                    && forest@[found as int].value@[0] == target
                ),
            decreases forest@.len() - i,
        {
            if forest[i].value.len() > 0 && forest[i].value[0] == target {
                found = i;
                break;
            }
            i = i + 1;
        }

        if found == forest.len() {
            // No match — create new node
            let new_node = Self::make_leaf(key, payload, offset);
            let ghost old_forest = forest@;
            forest.push(Box::new(new_node));
            proof { lemma_all_wf_push(old_forest, Box::new(new_node)); }
        } else {
            // Match found — remove, modify, reinsert
            let ghost old_forest = forest@;
            let mut boxed = forest.remove(found);
            proof { lemma_all_wf_remove(old_forest, found as int); }
            Self::node_insert(&mut *boxed, key, payload, offset);
            let ghost mid_forest = forest@;
            forest.insert(found, boxed);
            proof { lemma_all_wf_insert(mid_forest, found as int, boxed); }
        }
    }

    // Build a new leaf node from key[offset..] and payload[offset..]
    #[verifier::loop_isolation(false)]
    fn make_leaf(key: &Vec<u64>, payload: &Vec<u64>, offset: usize) -> (result: Node)
        requires
            offset < key@.len(),
            key@.len() == payload@.len(),
        ensures
            result.wf(),
    {
        let mut v: Vec<u64> = Vec::new();
        let mut p: Vec<u64> = Vec::new();
        let mut j: usize = offset;
        while j < key.len()
            invariant
                offset <= j <= key@.len(),
                v@.len() == p@.len(),
                v@.len() == (j - offset) as int,
                key@.len() == payload@.len(),
            decreases key@.len() - j,
        {
            v.push(key[j]);
            p.push(payload[j]);
            j = j + 1;
        }
        // v.len() == key.len() - offset > 0 (since offset < key.len())
        // v.len() == p.len() (pushed in lockstep)
        // children is empty, so forall over children is vacuously true
        Node { value: v, payload: p, children: Vec::new() }
    }

    // Insert key[offset..] into an existing node
    #[verifier::loop_isolation(false)]
    fn node_insert(
        node: &mut Node,
        key: &Vec<u64>,
        payload: &Vec<u64>,
        offset: usize,
    )
        requires
            old(node).wf(),
            offset < key@.len(),
            key@.len() == payload@.len(),
            old(node).value@[0] == key@[offset as int],
        ensures
            node.wf(),
        decreases key@.len() - offset, 0nat,
    {
        let vlen = node.value.len();
        let remaining = key.len() - offset;
        let limit = if vlen < remaining { vlen } else { remaining };

        // Compute common prefix length; first element matches by precondition
        let mut common: usize = 1;
        while common < limit
            invariant
                1 <= common <= limit,
                limit <= vlen,
                limit <= remaining,
                offset + common <= key@.len(),
                vlen == node.value@.len(),
                vlen == node.payload@.len(),
                offset < key@.len(),
                key@.len() == payload@.len(),
            decreases limit - common,
        {
            if node.value[common] != key[offset + common] {
                break;
            }
            common = common + 1;
        }

        if common == vlen && (offset + common) == key.len() {
            // Case 1: exact match — update payload in-place
            let mut j: usize = 0;
            while j < vlen
                invariant
                    j <= vlen,
                    vlen == node.value@.len(),
                    vlen == node.payload@.len(),
                    offset + vlen == key@.len(),
                    key@.len() == payload@.len(),
                    all_wf(node.children@),
                decreases vlen - j,
            {
                node.payload.set(j, payload[offset + j]);
                j = j + 1;
            }
            // value unchanged (len > 0), payload same length, children still all_wf
        } else if common == vlen {
            // Case 2: full edge match — descend into children
            // forest_insert only mutates node.children; value/payload unchanged
            Self::forest_insert(&mut node.children, key, payload, offset + common);
            // node.value@.len() > 0, == node.payload@.len(), all_wf(node.children@)
        } else {
            // Case 3: partial match — split node at `common`
            //
            //   Before: node.value = [a, b, c, d], children = [...]
            //   After:  node.value = [a, b],
            //           node.children = [suffix([c,d], old_children), new_leaf(...)]

            // Build suffix from node.value[common..] and node.payload[common..]
            // common < vlen in this branch, so suffix will have len > 0
            let mut sv: Vec<u64> = Vec::new();
            let mut sp: Vec<u64> = Vec::new();
            let mut j: usize = common;
            while j < vlen
                invariant
                    common <= j <= vlen,
                    sv@.len() == sp@.len(),
                    sv@.len() == (j - common) as int,
                    vlen == node.value@.len(),
                    vlen == node.payload@.len(),
                    common < vlen,
                    all_wf(node.children@),
                decreases vlen - j,
            {
                sv.push(node.value[j]);
                sp.push(node.payload[j]);
                j = j + 1;
            }
            // sv.len() == vlen - common > 0, sv.len() == sp.len()

            // Take original children (all wf)
            let mut orig_children: Vec<Box<Node>> = Vec::new();
            std::mem::swap(&mut node.children, &mut orig_children);
            // orig_children are all_wf (swapped from node.children)

            let suffix = Node { value: sv, payload: sp, children: orig_children };
            // suffix.wf(): sv.len() == vlen - common > 0, sv.len() == sp.len(),
            // orig_children were all_wf from old node

            // Truncate node to common prefix (common >= 1)
            node.value.truncate(common);
            node.payload.truncate(common);
            // node.value.len() == common > 0, node.payload.len() == common

            // Add suffix as child
            let ghost c0 = node.children@;
            node.children.push(Box::new(suffix));
            proof { lemma_all_wf_push(c0, Box::new(suffix)); }

            // Add new key's remainder as sibling child
            if (offset + common) < key.len() {
                let leaf = Self::make_leaf(key, payload, offset + common);
                let ghost c1 = node.children@;
                node.children.push(Box::new(leaf));
                proof { lemma_all_wf_push(c1, Box::new(leaf)); }
            }
        }
    }
}

// =========================================================================
// Test
// =========================================================================

fn test_basic() {
    let mut tree = RadixTreeMap::new();

    let key1: Vec<u64> = vec![1, 2, 3];
    let pay1: Vec<u64> = vec![10, 20, 30];
    tree.insert(&key1, &pay1);

    let q1: Vec<u64> = vec![1, 2, 3];
    let r1 = tree.prefix_len(&q1);

    let key2: Vec<u64> = vec![1, 2, 5];
    let pay2: Vec<u64> = vec![11, 22, 55];
    tree.insert(&key2, &pay2);

    let q2: Vec<u64> = vec![1, 2, 5];
    let r2 = tree.prefix_len(&q2);

    let q3: Vec<u64> = vec![1, 2];
    let r3 = tree.prefix_len(&q3);
}

} // verus!

fn main() {}
