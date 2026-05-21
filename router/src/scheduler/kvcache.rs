//! Per-request KV-cache hash tracking + the `BlockHash`-typed prefix matcher
//! the router consumes. The radix-tree implementation of that matcher lives
//! in the `radixtree` crate (`radixtree::RadixTreeBlockHash`); this file
//! holds only:
//!
//! - The `HashTableBlockHash` alternative (selected via the
//!   `hashtable-blockhash` cargo feature).
//! - `BlockHashState`, the per-request struct that hashes input tokens into
//!   block-aligned `u64`s and pairs them with backend block ids.
//! - The compile-time `PrefixBlockHash` re-export that picks one of the two
//!   `BlockHash` impls based on the active feature.
//!
//! The `BlockHash` trait itself is defined in `radixtree::block_hash` (it is
//! the interface both impls satisfy) and re-exported from this module so the
//! tests and the `mod hashtable_block_hash` body keep their existing
//! `use super::*` form.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use nohash_hasher::{BuildNoHashHasher, IntMap};
use xxhash_rust::xxh3::xxh3_64_with_seed;

pub(crate) use radixtree::{BlockHash, RadixTreeBlockHash};

pub(crate) static DEFAULT_BLOCK_HASH: u64 = 42;
#[cfg(feature = "default-hash-algo")]
pub(crate) type BackendBlockHash = [u64; 1];
#[cfg(feature = "sha256-hash-algo")]
pub(crate) type BackendBlockHash = [u64; 4];

mod hashtable_block_hash {
    use super::*;
    use smallvec::{smallvec, SmallVec};

    pub struct HashTableBlockHash {
        map: IntMap<u64, SmallVec<[u64; 1]>>, // hash -> block_id
        block_to_hash: Vec<Option<u64>>,      // block_id -> hash
        epoch: u64,
    }

    unsafe impl Send for HashTableBlockHash {}
    unsafe impl Sync for HashTableBlockHash {}

    impl BlockHash for HashTableBlockHash {
        fn new(num_blocks: usize) -> Self {
            Self {
                map: IntMap::with_capacity_and_hasher(num_blocks, BuildNoHashHasher::default()),
                block_to_hash: vec![None; num_blocks],
                epoch: 0,
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
                            // Duplicate hash with a new bid (e.g. two requests in the same
                            // step both prefilling the same prefix). The new bid must be
                            // tracked in block_to_hash too so a later remove(bid) works;
                            // without this the bid becomes a phantom in `bids` and
                            // `map[h]` never empties.
                            debug_assert!(self.block_to_hash[bid as usize].is_none());
                            self.block_to_hash[bid as usize] = Some(*h);
                            bids.push(bid);
                            n += 1;
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
            self.epoch += 1;
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
            self.epoch += 1;
        }

        fn epoch(&self) -> u64 {
            self.epoch
        }
    }
}

// ---------------------------------------------------------------------------
// BlockHashState — per-request hash tracking, backend-agnostic.
// Used by both `hashtable-blockhash` and `radixtree-blockhash` feature paths.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub(crate) struct BlockHashState {
    /// Calculated block hash values
    block_hashes: Vec<u64>,
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
    decision_epoch: AtomicU64,
}

// Sentinel for uninitialized `pred_hit_nblks`, i.e., a MSB mask 0b1_000...000
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
            decision_epoch: AtomicU64::new(0),
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

    pub fn append_tokens(&mut self, new_tokens: &[u32]) {
        self.token_in_last_block.extend_from_slice(new_tokens);
        if self.token_in_last_block.len() >= self.block_size {
            let block = self.token_in_last_block.drain(..self.block_size).collect::<Vec<_>>();
            let len = block.len() * std::mem::size_of::<u32>();
            let ptr = block.as_ptr() as *const u8;
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr, len) };
            self.prev_hash = xxh3_64_with_seed(bytes, self.prev_hash);
            self.block_hashes.push(self.prev_hash);
        }
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

    pub fn set_decision_epoch(&self, epoch: u64) {
        self.decision_epoch.store(epoch, Ordering::Release);
    }

    pub fn decision_epoch(&self) -> u64 {
        self.decision_epoch.load(Ordering::Acquire)
    }

    pub fn pred_hit_tokens(&self) -> usize {
        let v = self.pred_hit_nblks.load(Ordering::Acquire);
        if v == NONE_SENTINEL { 0 } else { v * self.block_size }
    }
}

#[cfg(feature = "hashtable-blockhash")]
pub(crate) use hashtable_block_hash::HashTableBlockHash as PrefixBlockHash;
#[cfg(all(feature = "radixtree-blockhash", not(feature = "preble-q")))]
pub(crate) use RadixTreeBlockHash as PrefixBlockHash;
// Under `--features preble-q`, the prefix matcher is Preble's richer
// flavour: the existing `RadixTreeBlockHash` (engine-driven) plus a
// per-replica sliding-window aggregator for `pod_load` / `pod_cost`.
// The Preble policy reads the aggregates via inherent methods on the
// concrete `PrebleBlockHash` type — those resolve cleanly because
// `PrefixBlockHash` is a compile-time alias.
#[cfg(feature = "preble-q")]
pub(crate) use crate::scheduler::policies::preble::PrebleBlockHash as PrefixBlockHash;

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
        hashes_master: Vec<u64>, // 固定 hash 序列（某条"语义链路"）
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

    // ---------- 执行器：严格"先删后缀，再写前缀"并对拍 ----------
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

    // ---------- 收尾：删除所有 block，并检查 RadixTree "一层空子节点 + 逻辑空" ----------
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

        // 3) 结构检查：RadixTree 只有一层,且 root 的每个 child 都是"空节点且无子"
        // (white-box assertion lives inside the `radixtree` crate; the
        // helper is `#[doc(hidden)]` and only walks the tree to assert).
        rt.assert_drained_to_one_layer();
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

        // —— 收尾：删光所有 block，并检查 RadixTree 的"一层空子节点 + 逻辑空"
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

        // —— 收尾：删光所有 block，并检查 RadixTree 的"一层空子节点 + 逻辑空"
        teardown_clear_all_and_assert_one_layer(&tracks, &mut ht, &mut rt);
    }

    // ==================== Property-Based Tests (proptest) ====================
    //
    // NOTE: RadixTreeBlockHash and HashTableBlockHash have DIFFERENT semantics:
    //   - RT: prefix-path trie. `get(q)` = longest matching prefix PATH.
    //         `insert()` returns count of new path nodes.
    //   - HT: per-hash set. `get(q)` = count of consecutive hashes in the set.
    //         `insert()` returns count of genuinely new hash values.
    //
    // They only agree when all hash values in sequences are DISTINCT.
    // Tests marked [BUG] document known bugs in RadixTreeBlockHash.

    use proptest::prelude::*;

    /// Strategy: generate hash sequences with ALL DISTINCT elements.
    /// This ensures RT and HT semantics agree (no duplicate hashes).
    fn distinct_hash_seq(max_len: usize) -> impl Strategy<Value = Vec<u64>> {
        // Use large alphabet + short sequences to make collisions extremely unlikely
        prop::collection::vec(0u64..10_000, 1..=max_len)
            .prop_filter("must have distinct elements", |v| {
                let mut seen = std::collections::HashSet::new();
                v.iter().all(|x| seen.insert(*x))
            })
    }

    /// Strategy: generate hash sequences with small alphabet (allows duplicates)
    fn hash_seq(max_len: usize) -> impl Strategy<Value = Vec<u64>> {
        prop::collection::vec(0u64..8, 1..=max_len)
    }

    /// Oracle: longest matching prefix of `query` against any inserted sequence
    fn oracle_prefix_len(inserted_seqs: &[Vec<u64>], query: &[u64]) -> usize {
        let mut best = 0;
        for seq in inserted_seqs {
            let common = seq.iter().zip(query.iter()).take_while(|(a, b)| a == b).count();
            best = best.max(common);
        }
        best
    }

    // --- Prop 1: insert-then-get returns full length ---
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn prop_rtbh_insert_then_get(
            hashes in hash_seq(16),
        ) {
            let n = hashes.len();
            let num_blocks = n + 1;
            let mut rt = RadixTreeBlockHash::new(num_blocks);
            let indices: Vec<u64> = (0..n as u64).collect();

            rt.insert(&hashes, indices);
            let got = rt.get(&hashes);
            prop_assert_eq!(got, n,
                "After inserting {:?}, get should return {}", hashes, n);
        }
    }

    // --- Prop 2: RT vs HT agree when sequences use disjoint hash domains ---
    // Each sequence uses a non-overlapping hash range, so both implementations agree.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_rtbh_vs_htbh_disjoint_domains(
            seq_lens in prop::collection::vec(1usize..=6, 1..=6),
        ) {
            let total_blocks: usize = seq_lens.iter().sum::<usize>() + 1;
            let num_blocks = total_blocks.max(1);

            let mut rt = RadixTreeBlockHash::new(num_blocks);
            let mut ht = HashTableBlockHash::new(num_blocks);

            // Build sequences with disjoint hash domains (no hash overlap)
            let mut seqs: Vec<Vec<u64>> = Vec::new();
            let mut hash_base: u64 = 100;
            let mut next_bid: u64 = 0;

            for &len in &seq_lens {
                let seq: Vec<u64> = (hash_base..hash_base + len as u64).collect();
                hash_base += len as u64 + 100; // large gap ensures disjoint
                let indices: Vec<u64> = (next_bid..next_bid + len as u64).collect();
                next_bid += len as u64;

                let n_rt = rt.insert(&seq, indices.clone());
                let n_ht = ht.insert(&seq, indices);
                prop_assert_eq!(n_rt, n_ht,
                    "Insert count mismatch for disjoint seq {:?}", seq);
                seqs.push(seq);
            }

            // Query each inserted sequence
            for seq in &seqs {
                let got_rt = rt.get(seq);
                let got_ht = ht.get(seq);
                prop_assert_eq!(got_rt, got_ht,
                    "get mismatch on inserted seq {:?}", seq);
            }
        }
    }

    // --- Prop 3: After removing all blocks, get returns 0 ---
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_rtbh_remove_all_then_get_zero(
            hashes in hash_seq(12),
        ) {
            let n = hashes.len();
            let num_blocks = n + 1;
            let mut rt = RadixTreeBlockHash::new(num_blocks);
            let indices: Vec<u64> = (0..n as u64).collect();

            rt.insert(&hashes, indices.clone());
            rt.remove(indices);

            let got = rt.get(&hashes);
            prop_assert_eq!(got, 0,
                "After removing all blocks, get should return 0 for {:?}", hashes);
        }
    }

    // --- Prop 4: Partial removal truncates correctly ---
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn prop_rtbh_partial_remove(
            hashes in prop::collection::vec(0u64..8, 3..=12),
            cut in 1usize..12,
        ) {
            let n = hashes.len();
            let cut = cut.min(n - 1);
            let num_blocks = n + 1;
            let mut rt = RadixTreeBlockHash::new(num_blocks);
            let indices: Vec<u64> = (0..n as u64).collect();

            rt.insert(&hashes, indices.clone());

            // Remove suffix [cut..n)
            let to_remove: Vec<u64> = (cut as u64..n as u64).collect();
            rt.remove(to_remove);

            // get should return at most `cut`
            let got = rt.get(&hashes);
            prop_assert!(got <= cut,
                "After removing suffix from {}, get returned {} > cut {}",
                n, got, cut);
        }
    }

    // --- Prop 5: Disjoint sequences don't interfere ---
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_rtbh_disjoint_sequences(
            seq_a in prop::collection::vec(0u64..4, 1..=8),
            seq_b in prop::collection::vec(5u64..9, 1..=8),
        ) {
            let total = seq_a.len() + seq_b.len();
            let num_blocks = total + 1;
            let mut rt = RadixTreeBlockHash::new(num_blocks);

            let idx_a: Vec<u64> = (0..seq_a.len() as u64).collect();
            let idx_b: Vec<u64> = (seq_a.len() as u64..total as u64).collect();

            rt.insert(&seq_a, idx_a);
            rt.insert(&seq_b, idx_b);

            prop_assert_eq!(rt.get(&seq_a), seq_a.len());
            prop_assert_eq!(rt.get(&seq_b), seq_b.len());
        }
    }

    // --- Prop 6: Oracle agreement (RT-only, prefix-trie semantics) ---
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_rtbh_oracle_agreement(
            seqs in prop::collection::vec(hash_seq(8), 1..=6),
            queries in prop::collection::vec(hash_seq(8), 1..=5),
        ) {
            let total_blocks: usize = seqs.iter().map(|s| s.len()).sum::<usize>() + 1;
            let num_blocks = total_blocks.max(1);
            let mut rt = RadixTreeBlockHash::new(num_blocks);

            let mut next_bid: u64 = 0;
            for seq in &seqs {
                let indices: Vec<u64> = (next_bid..next_bid + seq.len() as u64).collect();
                next_bid += seq.len() as u64;
                rt.insert(seq, indices);
            }

            for query in &queries {
                let expected = oracle_prefix_len(&seqs, query);
                let actual = rt.get(query);
                prop_assert_eq!(actual, expected,
                    "Oracle mismatch: seqs={:?}, query={:?}", seqs, query);
            }
        }
    }

    // --- Prop 7: Prefix sharing with distinct hashes ---
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_rtbh_prefix_sharing(
            base in distinct_hash_seq(4),
            suffix_a in distinct_hash_seq(3),
            suffix_b in distinct_hash_seq(3),
        ) {
            let mut key_a = base.clone();
            key_a.extend_from_slice(&suffix_a);
            let mut key_b = base.clone();
            key_b.extend_from_slice(&suffix_b);

            // Ensure all block IDs are distinct across both keys
            let total = key_a.len() + key_b.len();
            let num_blocks = total + 1;
            let mut rt = RadixTreeBlockHash::new(num_blocks);

            let idx_a: Vec<u64> = (0..key_a.len() as u64).collect();
            let idx_b: Vec<u64> = (key_a.len() as u64..total as u64).collect();

            rt.insert(&key_a, idx_a);
            rt.insert(&key_b, idx_b);

            // Both queries should be findable
            let expected_a = oracle_prefix_len(&[key_a.clone(), key_b.clone()], &key_a);
            let expected_b = oracle_prefix_len(&[key_a.clone(), key_b.clone()], &key_b);
            prop_assert_eq!(rt.get(&key_a), expected_a);
            prop_assert_eq!(rt.get(&key_b), expected_b);

            // Base prefix should match at least base.len()
            prop_assert!(rt.get(&base) >= base.len(),
                "Base prefix {:?} should match at least {} but got {}",
                base, base.len(), rt.get(&base));
        }
    }

    // --- Prop 8: Single-element sequences ---
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_rtbh_single_element(
            val in 0u64..100,
        ) {
            let num_blocks = 2;
            let mut rt = RadixTreeBlockHash::new(num_blocks);
            rt.insert(&[val], vec![0]);
            prop_assert_eq!(rt.get(&[val]), 1);
            prop_assert_eq!(rt.get(&[val + 1]), 0);

            rt.remove(vec![0]);
            prop_assert_eq!(rt.get(&[val]), 0);
        }
    }

    // --- Prop 9: Insert return value with distinct hashes ---
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_rtbh_insert_returns_new_count(
            hashes in distinct_hash_seq(8),
        ) {
            let n = hashes.len();
            let num_blocks = 2 * n + 1;
            let mut rt = RadixTreeBlockHash::new(num_blocks);
            let mut ht = HashTableBlockHash::new(num_blocks);

            let idx1: Vec<u64> = (0..n as u64).collect();
            let n1_rt = rt.insert(&hashes, idx1.clone());
            let n1_ht = ht.insert(&hashes, idx1);
            prop_assert_eq!(n1_rt, n1_ht, "First insert count mismatch");
        }
    }

    // ==================== BUG-targeting property tests ====================

    // --- Prop 10: insert() returns the same count from both implementations
    //              for duplicate hashes with distinct bids ---
    //
    // Originally written as a `bug_*` test to *document* a semantic gap:
    // RadixTreeBlockHash treated each (hash, bid) pair as a new path node
    // (so `[6, 6]` with bids `[0, 1]` returned 2), while
    // HashTableBlockHash only counted the unique hash and returned 1.
    //
    // The HashTableBlockHash implementation has since been updated to
    // also track duplicate-hash-with-distinct-bid as a fresh insertion
    // (the `bids.push(bid); n += 1;` arm in its `and_modify` branch),
    // so the two implementations now agree. This test is the regression
    // guard for that agreement — flip it back to a `bug_*` documenting
    // a divergence if the impls ever disagree again.
    #[test]
    fn insert_count_duplicate_hashes_agrees() {
        let mut rt = RadixTreeBlockHash::new(3);
        let mut ht = HashTableBlockHash::new(3);

        let n_rt = rt.insert(&[6, 6], vec![0, 1]);
        let n_ht = ht.insert(&[6, 6], vec![0, 1]);

        // Both impls now count each (hash, bid) pair as a new insertion.
        assert_eq!(n_rt, 2, "RT: both positions are new path nodes");
        assert_eq!(n_ht, 2, "HT: each (hash, bid) pair counts as new");
        assert_eq!(n_rt, n_ht, "RT and HT must agree on insert() count");
    }

    // --- [BUG] Prop 11: get() semantic mismatch for non-prefix query ---
    // After inserting [0, 2], querying [2]:
    //   RT returns 0 (no prefix path match)
    //   HT returns 1 (hash 2 exists in set)
    #[test]
    fn bug_get_semantic_mismatch() {
        let mut rt = RadixTreeBlockHash::new(3);
        let mut ht = HashTableBlockHash::new(3);

        rt.insert(&[0, 2], vec![0, 1]);
        ht.insert(&[0, 2], vec![0, 1]);

        // Document the semantic difference
        assert_eq!(rt.get(&[2]), 0, "RT: [2] is not a prefix of [0,2]");
        assert_eq!(ht.get(&[2]), 1, "HT: hash 2 exists in the set");
        // NOTE: This means the `radixtree-blockhash` and `hashtable-blockhash`
        // features give DIFFERENT results. If the router switches between them
        // via feature flags expecting identical behavior, this is a bug.
    }

    // --- [BUG] Prop 12: Cascading remove assertion failure ---
    // When two sequences share a prefix and we insert them separately,
    // removing one's blocks can cascade and clear the other's blocks,
    // then removing the second sequence's blocks hits a debug_assert.
    #[test]
    fn bug_cascading_remove_shared_prefix() {
        // Two sequences sharing prefix [1]:
        //   seq_a = [1, 2] with bids [0, 1]
        //   seq_b = [1, 3] with bids [2, 3]
        // Tree structure after both inserts:
        //   root -> [1] -> [2] (bid 0, 1)
        //                -> [3] (bid 2, 3)
        //
        // Removing bid 1 (the "3" in seq_a [1,2]) truncates to [1] and clears children.
        // This cascading removal also clears bid 2 and bid 3 from seq_b.
        // Then attempting to remove bid 2 or 3 should not crash.
        let mut rt = RadixTreeBlockHash::new(10);

        rt.insert(&[1, 2], vec![0, 1]);
        rt.insert(&[1, 3], vec![2, 3]);

        // Remove the suffix of seq_a: this cascades and also clears seq_b's blocks
        rt.remove(vec![1]);

        // The tree should still be queryable without panic
        let _ = rt.get(&[1, 2]);
        let _ = rt.get(&[1, 3]);
    }

    // --- Prop 13: Interleaved insert-remove-get (RT-only, no HT comparison) ---
    #[derive(Clone, Debug)]
    enum Op {
        Insert { hashes: Vec<u64>, len: usize },
        RemoveAll { seq_idx: usize },
        Get { hashes: Vec<u64> },
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            4 => distinct_hash_seq(6).prop_map(|h| {
                let l = h.len();
                Op::Insert { hashes: h, len: l }
            }),
            2 => (0usize..100).prop_map(|i| Op::RemoveAll { seq_idx: i }),
            3 => hash_seq(6).prop_map(|h| Op::Get { hashes: h }),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_rtbh_random_op_sequence(
            ops in prop::collection::vec(op_strategy(), 5..=20),
        ) {
            let num_blocks = 2000;
            let mut rt = RadixTreeBlockHash::new(num_blocks);
            let mut next_bid: u64 = 0;

            struct SeqState { indices: Vec<u64>, hashes: Vec<u64>, removed: bool }
            let mut live_seqs: Vec<SeqState> = Vec::new();

            for op in &ops {
                match op {
                    Op::Insert { hashes, len } => {
                        if next_bid + *len as u64 >= num_blocks as u64 { continue; }
                        let indices: Vec<u64> = (next_bid..next_bid + *len as u64).collect();
                        next_bid += *len as u64;

                        rt.insert(hashes, indices.clone());
                        // Verify inserted sequence is findable
                        let got = rt.get(hashes);
                        prop_assert!(got >= hashes.len(),
                            "After insert, get({:?}) = {} < {}", hashes, got, hashes.len());

                        live_seqs.push(SeqState {
                            indices,
                            hashes: hashes.clone(),
                            removed: false,
                        });
                    }
                    Op::RemoveAll { seq_idx } => {
                        if live_seqs.is_empty() { continue; }
                        let idx = seq_idx % live_seqs.len();
                        if live_seqs[idx].removed { continue; }

                        let indices = live_seqs[idx].indices.clone();
                        rt.remove(indices);
                        live_seqs[idx].removed = true;
                    }
                    Op::Get { hashes } => {
                        // Just ensure get doesn't panic
                        let _ = rt.get(hashes);
                    }
                }
            }
        }
    }

    // ================== Fuzz：更大规模，严格"删后缀/写前缀" ==================
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

        // —— 收尾：删光所有 block，并检查 RadixTree 的"一层空子节点 + 逻辑空"
        teardown_clear_all_and_assert_one_layer(&tracks, &mut ht, &mut rt);
    }
}
