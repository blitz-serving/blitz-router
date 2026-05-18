// Compact bitset over replica indices `0..num_replicas`.
//
// Backing: `SmallVec<[u64; 1]>`. The 1×u64 inline buffer covers up to 64
// replicas without heap allocation — the common deployment shape on a
// single host with one replica per GPU. Larger fleets (>64) spill to the
// heap. `len()` is recomputed lazily on demand from popcount; for the
// per-(node, owner) iteration patterns Preble uses, this is plenty.

use smallvec::{smallvec, SmallVec};

const BITS_PER_WORD: usize = 64;

/// Bit `i` set ⇔ replica `i` is in the set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplicaSet {
    /// Number of words = ceil(num_replicas / 64). Sized at construction
    /// to avoid out-of-bounds during `insert`.
    bits: SmallVec<[u64; 1]>,
    /// Capacity (number of distinct replica IDs accepted). Stored so
    /// `iter()` can stop early; not strictly required for correctness
    /// since out-of-range bits would simply never be set.
    cap: usize,
}

impl ReplicaSet {
    /// Empty set sized for `num_replicas` IDs.
    pub fn new(num_replicas: usize) -> Self {
        let n_words = num_replicas.div_ceil(BITS_PER_WORD).max(1);
        Self {
            bits: smallvec![0u64; n_words],
            cap: num_replicas,
        }
    }

    /// Add `id` to the set. Out-of-range IDs are silently ignored.
    pub fn insert(&mut self, id: usize) {
        if id >= self.cap {
            return;
        }
        let (w, b) = (id / BITS_PER_WORD, id % BITS_PER_WORD);
        self.bits[w] |= 1u64 << b;
    }

    /// Remove `id` from the set. Returns `true` iff `id` was present.
    pub fn remove(&mut self, id: usize) -> bool {
        if id >= self.cap {
            return false;
        }
        let (w, b) = (id / BITS_PER_WORD, id % BITS_PER_WORD);
        let mask = 1u64 << b;
        let was_set = self.bits[w] & mask != 0;
        self.bits[w] &= !mask;
        was_set
    }

    pub fn contains(&self, id: usize) -> bool {
        if id >= self.cap {
            return false;
        }
        let (w, b) = (id / BITS_PER_WORD, id % BITS_PER_WORD);
        self.bits[w] & (1u64 << b) != 0
    }

    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// Number of replicas currently in the set.
    pub fn len(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn clear(&mut self) {
        for w in self.bits.iter_mut() {
            *w = 0;
        }
    }

    /// Iterate replica IDs in ascending order.
    pub fn iter(&self) -> ReplicaSetIter<'_> {
        ReplicaSetIter {
            bits: &self.bits,
            word_idx: 0,
            cur: self.bits.first().copied().unwrap_or(0),
            cap: self.cap,
        }
    }
}

pub struct ReplicaSetIter<'a> {
    bits: &'a [u64],
    word_idx: usize,
    cur: u64,
    cap: usize,
}

impl<'a> Iterator for ReplicaSetIter<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.cur != 0 {
                let b = self.cur.trailing_zeros() as usize;
                let id = self.word_idx * BITS_PER_WORD + b;
                self.cur &= self.cur - 1; // clear lowest set bit
                if id < self.cap {
                    return Some(id);
                }
                // Out-of-range bit (shouldn't happen since insert masks);
                // keep scanning.
                continue;
            }
            self.word_idx += 1;
            if self.word_idx >= self.bits.len() {
                return None;
            }
            self.cur = self.bits[self.word_idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_set() {
        let s = ReplicaSet::new(16);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(!s.contains(0));
        assert_eq!(s.iter().count(), 0);
    }

    #[test]
    fn insert_contains_remove() {
        let mut s = ReplicaSet::new(16);
        s.insert(0);
        s.insert(7);
        s.insert(15);
        assert!(s.contains(0));
        assert!(s.contains(7));
        assert!(s.contains(15));
        assert!(!s.contains(8));
        assert_eq!(s.len(), 3);

        assert!(s.remove(7));
        assert!(!s.contains(7));
        assert!(!s.remove(7));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn iter_ordered() {
        let mut s = ReplicaSet::new(70); // spills to >1 word
        for i in [3usize, 0, 64, 65, 30, 63] {
            s.insert(i);
        }
        let collected: Vec<usize> = s.iter().collect();
        assert_eq!(collected, vec![0, 3, 30, 63, 64, 65]);
    }

    #[test]
    fn out_of_range_is_silent() {
        let mut s = ReplicaSet::new(8);
        s.insert(99);
        assert_eq!(s.len(), 0);
        assert!(!s.contains(99));
    }

    #[test]
    fn clear() {
        let mut s = ReplicaSet::new(16);
        s.insert(1);
        s.insert(5);
        s.clear();
        assert!(s.is_empty());
    }
}
