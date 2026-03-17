use nohash_hasher::IntMap;
use tracing_subscriber::field::debug;
use tracing_subscriber::fmt::format;

use crate::simulator::batch::Request;
use crate::simulator::config::SimulationConfig;
use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockHash {
    hash: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KVBlock {
    block_id: u32,
    block_hash: Option<u64>,
    ref_cnt: u32,
    token_num: usize,
    prev_free_block_id: Option<u32>,
    next_free_block_id: Option<u32>,
}

const DUMMY_ID: u32 = 0;

// should have enough information to redo the operation
#[derive(Clone, Debug)]
enum Operation {
    // request_id, Vec<(prev_block_state, new_block_hash)>
    Allocate(u64, Vec<(KVBlock, u64)>, Option<u32>, Option<u32>),
    // FreeRequest(),
    // CacheHit(Vec<KVBlock>),
}

// Copy-on-write state for simulation mode
struct CowState {
    // Modified blocks: block_id -> KVBlock
    // pub blocks: HashMap<u32, KVBlock>,
    // Modified cached_block entries: hash -> HashSet<block_id>
    pub cached_block: HashSet<u64>,
    // Modified req_to_block_ids entries: request_id -> Vec<block_id>
    // pub req_to_block_ids: HashMap<u64, Vec<u32>>,
    // // Modified req_to_hashes entries: request_id -> Vec<hash>
    // pub req_to_hashes: HashMap<u64, Vec<u64>>,
    // // Modified free list pointers
    // pub free_list_head: Option<Option<u32>>,
    // pub free_list_tail: Option<Option<u32>>,
    // // Track which cached_block entries have been removed
    // pub removed_cached_block_keys: HashSet<u64>,
    // // Track which req_to_block_ids entries have been removed
    // pub removed_req_to_block_ids: HashSet<u64>,
    // // Track which req_to_hashes entries have been removed
    // pub removed_req_to_hashes: HashSet<u64>,
}

impl CowState {
    fn new() -> Self {
        CowState {
            // blocks: HashMap::new(),
            cached_block: HashSet::new(),
            // req_to_block_ids: HashMap::new(),
            // req_to_hashes: HashMap::new(),
            // free_list_head: None,
            // free_list_tail: None,
            // removed_cached_block_keys: HashSet::new(),
            // removed_req_to_block_ids: HashSet::new(),
            // removed_req_to_hashes: HashSet::new(),
        }
    }

    fn clear(&mut self) {
        // self.blocks.clear();
        self.cached_block.clear();
        // self.req_to_block_ids.clear();
        // self.req_to_hashes.clear();
        // self.free_list_head = None;
        // self.free_list_tail = None;
        // self.removed_cached_block_keys.clear();
        // self.removed_req_to_block_ids.clear();
        // self.removed_req_to_hashes.clear();
    }
}

pub struct KVCacheManager {
    // holds the ownership of all blocks
    // pub blocks: Vec<KVBlock>,

    // trace the free list of blocks
    // pub free_list_head: Option<u32>,
    // pub free_list_tail: Option<u32>,

    // mapping from block hash to block ids
    pub cached_block: HashSet<u64>,

    pub block_id_to_hash: HashMap<u32, u64>,

    // mapping from request id to allocated block ids
    pub req_to_block_ids: HashMap<u64, Vec<u32>>,

    // store the hashes for each request
    pub req_to_hashes: HashMap<u64, Vec<u64>>,
    config: Arc<SimulationConfig>,

    evicted_blocks: Option<Vec<u64>>,
    new_hash_block_ids: Option<IntMap<u64, Vec<u64>>>,
    redo_stack: VecDeque<Operation>,
    // Copy-on-write state for simulation mode
    // cow_state: Option<CowState>,
    uncommited_kvcache: HashSet<u64>,
}

// impl PartialEq for KVCacheManager {
//     fn eq(&self, other: &Self) -> bool {
//         self.blocks == other.blocks
//             && self.free_list_head == other.free_list_head
//             && self.free_list_tail == other.free_list_tail
//             && self.cached_block == other.cached_block
//             && self.req_to_block_ids == other.req_to_block_ids
//             && self.req_to_hashes == other.req_to_hashes
//             && self.redo_stack == other.redo_stack
//         // ✅ 注意：config 故意跳过比较
//     }
// }
// impl Eq for KVCacheManager {}

impl KVCacheManager {
    pub fn new(config: Arc<SimulationConfig>) -> Self {
        let num_blocks = config.scheduler_config.num_blocks;
        // let mut blocks = Vec::with_capacity(num_blocks);
        // for i in 0..num_blocks {
        //     blocks.push(KVBlock {
        //         block_id: i as u32,
        //         block_hash: None,
        //         ref_cnt: 0,
        //         token_num: 0,
        //         prev_free_block_id: if i == 0 { None } else { Some((i - 1) as u32) },
        //         next_free_block_id: if i == num_blocks - 1 { None } else { Some((i + 1) as u32) },
        //     });
        // }
        KVCacheManager {
            // blocks,
            // free_list_head: Some(0),
            // free_list_tail: Some((num_blocks - 1) as u32),
            cached_block: HashSet::new(),
            evicted_blocks: if config.fake_backend { Some(Vec::new()) } else { None },
            new_hash_block_ids: if config.fake_backend { Some(IntMap::default()) } else { None },
            req_to_block_ids: HashMap::new(),
            req_to_hashes: HashMap::new(),
            block_id_to_hash: HashMap::new(),
            config,
            redo_stack: VecDeque::new(),
            uncommited_kvcache: HashSet::with_capacity(10_000),
        }
    }

    /// Enable copy-on-write mode for simulation
    pub fn enable_simulation_mode(&mut self) {
        // debug_assert!(self.cow_state.is_none());
        // if self.cow_state.is_none() {
        //     self.cow_state = Some(CowState::new());
        // }
        self.uncommited_kvcache.clear();
    }

    /// Disable copy-on-write mode and clear all COW state
    pub fn disable_simulation_mode(&mut self) {
        // debug_assert!(self.cow_state.is_some());
        // if let Some(ref mut cow) = self.cow_state {
        //     cow.clear();
        // }
        // self.cow_state = None;
        self.uncommited_kvcache.clear();
    }

    /// Get a block, checking COW state first
    // fn get_block(&self, block_id: u32) -> &KVBlock {
    //     // if let Some(ref cow) = self.cow_state {
    //     //     if let Some(block) = cow.blocks.get(&block_id) {
    //     //         return block;
    //     //     }
    //     // }
    //     &self.blocks[block_id as usize]
    // }

    /// Get a mutable reference to a block, creating COW copy if needed
    // fn get_block_mut(&mut self, block_id: u32) -> &mut KVBlock {
    //     // if let Some(ref mut cow) = self.cow_state {
    //     //     if !cow.blocks.contains_key(&block_id) {
    //     //         // Create COW copy
    //     //         cow.blocks.insert(block_id, self.blocks[block_id as usize].clone());
    //     //     }
    //     //     return cow.blocks.get_mut(&block_id).unwrap();
    //     // }
    //     &mut self.blocks[block_id as usize]
    // }

    /// Get free_list_head, checking COW state first
    // fn get_free_list_head(&self) -> Option<u32> {
    //     // if let Some(ref cow) = self.cow_state {
    //     //     if let Some(head) = cow.free_list_head {
    //     //         return head;
    //     //     }
    //     // }
    //     self.free_list_head
    // }

    /// Set free_list_head, creating COW copy if needed
    // fn set_free_list_head(&mut self, head: Option<u32>) {
    //     // if let Some(ref mut cow) = self.cow_state {
    //     //     if cow.free_list_head.is_none() {
    //     //         cow.free_list_head = Some(self.free_list_head);
    //     //     }
    //     //     cow.free_list_head = Some(head);
    //     // } else {
    //     self.free_list_head = head;
    //     // }
    // }

    /// Get free_list_tail, checking COW state first
    // fn get_free_list_tail(&self) -> Option<u32> {
    //     // if let Some(ref cow) = self.cow_state {
    //     //     if let Some(tail) = cow.free_list_tail {
    //     //         return tail;
    //     //     }
    //     // }
    //     self.free_list_tail
    // }

    /// Set free_list_tail, creating COW copy if needed
    // fn set_free_list_tail(&mut self, tail: Option<u32>) {
    //     // if let Some(ref mut cow) = self.cow_state {
    //     //     if cow.free_list_tail.is_none() {
    //     //         cow.free_list_tail = Some(self.free_list_tail);
    //     //     }
    //     //     cow.free_list_tail = Some(tail);
    //     // } else {
    //     self.free_list_tail = tail;
    //     // }
    // }

    /// Get cached_block entry, checking COW state first
    fn get_cached_block(&self, hash: &u64) -> bool {
        self.cached_block.contains(hash) || self.uncommited_kvcache.contains(hash)
        // if let Some(ref cow) = self.cow_state {
        //     // if cow.removed_cached_block_keys.contains(hash) {
        //     //     return false;
        //     // }
        //     if cow.cached_block.contains(&hash) {
        //         return true;
        //     }
        // }
        // self.cached_block.get(hash).map_or(false, |cached_block| !cached_block.is_empty())
    }

    /// Get mutable cached_block entry, creating COW copy if needed
    // fn get_cached_block_mut(&mut self, hash: u64) -> &mut HashSet<u32> {
    //     // assert!(
    //     //     self.cow_state.is_none(),
    //     //     "get_cached_block_mut should only be called not in cow mode"
    //     // );
    //     // if let Some(ref mut cow) = self.cow_state {
    //     //     cow.removed_cached_block_keys.remove(&hash);
    //     //     if !cow.cached_block.contains(&hash) {
    //     //         // Create COW copy
    //     //         if let Some(original) = self.cached_block.get(&hash) {
    //     //             cow.cached_block.insert(hash, original.clone());
    //     //         } else {
    //     //             cow.cached_block.insert(hash, HashSet::new());
    //     //         }
    //     //     }
    //     //     return cow.cached_block.get_mut(&hash).unwrap();
    //     // }
    //     self.cached_block.entry(hash).or_insert_with(HashSet::new)
    // }

    /// Remove cached_block entry, creating COW copy if needed
    fn remove_cached_block(&mut self, hash: &u64) {
        // if let Some(ref mut cow) = self.cow_state {
        //     cow.removed_cached_block_keys.insert(*hash);
        //     cow.cached_block.remove(hash);
        // } else {
        self.cached_block.remove(hash);
        // }
    }

    /// Get req_to_block_ids entry, checking COW state first
    fn get_req_to_block_ids(&self, request_id: &u64) -> Option<&Vec<u32>> {
        // if let Some(ref cow) = self.cow_state {
        //     if cow.removed_req_to_block_ids.contains(request_id) {
        //         return None;
        //     }
        //     if let Some(ids) = cow.req_to_block_ids.get(request_id) {
        //         return Some(ids);
        //     }
        // }
        self.req_to_block_ids.get(request_id)
    }

    /// Get mutable req_to_block_ids entry, creating COW copy if needed
    fn get_req_to_block_ids_mut(&mut self, request_id: u64) -> &mut Vec<u32> {
        // if let Some(ref mut cow) = self.cow_state {
        //     cow.removed_req_to_block_ids.remove(&request_id);
        //     if !cow.req_to_block_ids.contains_key(&request_id) {
        //         // Create COW copy
        //         if let Some(original) = self.req_to_block_ids.get(&request_id) {
        //             cow.req_to_block_ids.insert(request_id, original.clone());
        //         } else {
        //             cow.req_to_block_ids.insert(request_id, Vec::new());
        //         }
        //     }
        //     return cow.req_to_block_ids.get_mut(&request_id).unwrap();
        // }
        self.req_to_block_ids.entry(request_id).or_insert_with(Vec::new)
    }

    /// Remove req_to_block_ids entry, creating COW copy if needed
    fn remove_req_to_block_ids(&mut self, request_id: &u64) {
        // if let Some(ref mut cow) = self.cow_state {
        //     cow.removed_req_to_block_ids.insert(*request_id);
        //     cow.req_to_block_ids.remove(request_id);
        // } else {
        self.req_to_block_ids.remove(request_id);
        // }
    }

    /// Get req_to_hashes entry, checking COW state first
    fn get_req_to_hashes(&self, request_id: &u64) -> Option<&Vec<u64>> {
        // if let Some(ref cow) = self.cow_state {
        //     if cow.removed_req_to_hashes.contains(request_id) {
        //         return None;
        //     }
        //     if let Some(hashes) = cow.req_to_hashes.get(request_id) {
        //         return Some(hashes);
        //     }
        // }
        self.req_to_hashes.get(request_id)
    }

    /// Get mutable req_to_hashes entry, creating COW copy if needed
    fn get_req_to_hashes_mut(&mut self, request_id: u64) -> &mut Vec<u64> {
        // if let Some(ref mut cow) = self.cow_state {
        //     cow.removed_req_to_hashes.remove(&request_id);
        //     if !cow.req_to_hashes.contains_key(&request_id) {
        //         // Create COW copy
        //         if let Some(original) = self.req_to_hashes.get(&request_id) {
        //             cow.req_to_hashes.insert(request_id, original.clone());
        //         } else {
        //             cow.req_to_hashes.insert(request_id, Vec::new());
        //         }
        //     }
        //     return cow.req_to_hashes.get_mut(&request_id).unwrap();
        // }
        self.req_to_hashes.entry(request_id).or_insert_with(Vec::new)
    }

    /// Remove req_to_hashes entry, creating COW copy if needed
    fn remove_req_to_hashes(&mut self, request_id: &u64) {
        // if let Some(ref mut cow) = self.cow_state {
        //     cow.removed_req_to_hashes.insert(*request_id);
        //     cow.req_to_hashes.remove(request_id);
        // } else {
        self.req_to_hashes.remove(request_id);
        // }
    }

    // pub fn poll_evicted_blocks(&mut self) -> Vec<u64> {
    //     self.evicted_blocks.as_mut().unwrap().drain(..).collect()
    // }
    pub fn get_req_block_ids(&self, request_id: u64) -> Vec<u64> {
        self.get_req_to_block_ids(&request_id).unwrap().iter().map(|x| *x as u64).collect()
    }

    // pub fn is_block_free(&self, block_id: u32) -> bool {
    //     let block = self.get_block(block_id);
    //     block.ref_cnt == 0
    // }

    pub fn add_request(&mut self, request_id: u64, hashes: Vec<u64>) {
        *self.get_req_to_hashes_mut(request_id) = hashes;
        *self.get_req_to_block_ids_mut(request_id) = Vec::new();
    }

    pub fn set_req_block_ids(&mut self, request_id: u64, ids: &Vec<u64>) {
        // 1. extend the block id list
        let (old_len, new_len) = {
            let block_id_vec = self.get_req_to_block_ids_mut(request_id);
            let old_len = block_id_vec.len();
            debug_assert!(ids.len() >= old_len);
            block_id_vec.extend(ids[old_len..].iter().map(|&id| id as u32));
            let new_len = block_id_vec.len();
            (old_len, new_len)
        };

        // 2. Collect hash and block id pairs before making mutable borrows
        let hash_block_pairs: Vec<(u64, u32)> = {
            let hash_vec = self.get_req_to_hashes(&request_id).unwrap();
            let block_id_vec = self.get_req_to_block_ids(&request_id).unwrap();
            let end = hash_vec.len().min(new_len);
            hash_vec[old_len..end]
                .iter()
                .zip(block_id_vec[old_len..end].iter())
                .map(|(hash, id)| (*hash, *id))
                .collect()
        };

        // 3. update the cached_block mapping and block hashes
        for (hash, id) in hash_block_pairs {
            // tracing::info!(
            //     "set_req_block_ids: request_id {}, hash {}, block_id {}",
            //     request_id,
            //     hash,
            //     id
            // );
            self.cached_block.insert(hash);
            // self.get_cached_block_mut(hash).insert(id);
            self.block_id_to_hash.insert(id, hash);
            // self.get_block_mut(id).block_hash = Some(hash);
        }

        // 4. touch the newly added blocks
        // ids[old_len..].iter().for_each(|&id| {
        //     self.touch(id as u32);
        // });
    }

    /// Mark the block as recently used, and increment its reference count.
    /// if the block was free, remove it from the free list.
    // fn touch(&mut self, id: u32) {
    //     let block = self.get_block_mut(id);

    //     block.ref_cnt += 1;
    //     // let cloned_block = if if_redo { Some(block.clone()) } else { None };
    //     if block.ref_cnt == 1 {
    //         // Remove from free list
    //         self.evict_from_free_list(id);
    //     }
    //     // cloned_block
    // }

    pub fn evict_block_hash(&mut self, block_id: u32) {
        // Extract hash before making other mutable borrows
        // let hash = {
        //     let block = self.get_block_mut(block_id);
        //     debug_assert!(
        //         block.ref_cnt == 0,
        //         "Evicting block {} with non-zero ref_cnt {}",
        //         block_id,
        //         block.ref_cnt
        //     );
        //     block.block_hash
        // };

        // if let Some(hash) = hash {
        //     let set = self.get_cached_block_mut(hash);
        //     set.remove(&block_id);
        //     if set.is_empty() {
        //         self.remove_cached_block(&hash);
        //     }
        //     self.get_block_mut(block_id).block_hash = None;
        // }
        if let Some(hash) = self.block_id_to_hash.get(&block_id) {
            self.cached_block.remove(hash);
            self.block_id_to_hash.remove(&block_id);
        }
    }

    /// Update the hash list for the request when last block is full, and update the cached_block mapping
    pub fn req_update_hash(&mut self, request_id: u64, hash: u64) {
        // Extract bid before making other mutable borrows
        let bid = {
            let hashes = self.get_req_to_hashes_mut(request_id);
            hashes.push(hash);

            let hashes_len = hashes.len();
            let block_ids = self.get_req_to_block_ids_mut(request_id);
            assert!(block_ids.len() == hashes_len);
            *block_ids.last().unwrap()
        };

        // if let Some(new_hash_block_ids) = self.new_hash_block_ids.as_mut() {
        //     new_hash_block_ids.entry(request_id).or_insert_with(Vec::new).push(bid as u64);
        // }
        // let bid_num = bid.clone();
        // self.get_block_mut(bid).block_hash = Some(hash);
        self.block_id_to_hash.insert(bid, hash);
        self.cached_block.insert(hash);
        // self.get_cached_block_mut(hash).insert(bid);
        // self.touch(bid_num);
    }

    // pub fn poll_new_block_hash_ids(&mut self) -> IntMap<u64, Vec<u64>> {
    //     self.new_hash_block_ids.as_mut().unwrap().drain().collect()
    // }

    /// Not implemented yet
    // pub fn redo(&mut self) {
    //     // Implement logic to redo KV cache state if necessary
    //     while let Some(op) = self.redo_stack.pop_back() {
    //         match op {
    //             // Operation::CacheHit(blocks) => {
    //             //     for block in blocks {
    //             //         self.touch(block.block_id, false);
    //             //     }
    //             // }
    //             Operation::Allocate(
    //                 request_id,
    //                 mut prev_blocks,
    //                 prev_free_block_head,
    //                 prev_free_block_tail,
    //             ) => {
    //                 let hashes = self.req_to_hashes.get_mut(&request_id).unwrap();
    //                 hashes.truncate(hashes.len() - prev_blocks.len());
    //                 while let Some((prev_block, new_hash)) = prev_blocks.pop() {
    //                     let block_id = prev_block.block_id;
    //                     let block = &mut self.blocks[block_id as usize];

    //                     // Restore previous state
    //                     *block = prev_block;

    //                     // restore cached_block
    //                     if let Some(old_hash) = block.block_hash {
    //                         self.cached_block
    //                             .entry(old_hash)
    //                             .or_insert_with(HashSet::new)
    //                             .insert(block_id);
    //                     }

    //                     self.cached_block.get_mut(&new_hash).unwrap().remove(&block_id);
    //                     if self.cached_block.get(&new_hash).unwrap().is_empty() {
    //                         self.cached_block.remove(&new_hash);
    //                     }

    //                     if !block.next_free_block_id.is_none() {
    //                         let next_id = block.next_free_block_id.unwrap();
    //                         self.blocks[next_id as usize].prev_free_block_id = Some(block_id);
    //                     }
    //                 }

    //                 self.free_list_head = prev_free_block_head;
    //                 self.free_list_tail = prev_free_block_tail;
    //             }
    //         }
    //     }
    // }

    /// Evict a block from the free list, update its prev/next pointers accordingly
    // fn evict_from_free_list(&mut self, block_id: u32) {
    //     // tracing::info!("Evicting block {} from free list", block_id);

    //     // 先取出 prev/next id
    //     let (prev_id, next_id) = {
    //         let block = self.get_block_mut(block_id);
    //         let prev = block.prev_free_block_id;
    //         let next = block.next_free_block_id;

    //         // 清空自身链表指针
    //         block.prev_free_block_id = None;
    //         block.next_free_block_id = None;

    //         (prev, next)
    //     };

    //     // 然后再修改前驱节点
    //     if let Some(prev) = prev_id {
    //         self.get_block_mut(prev).next_free_block_id = next_id;
    //     } else if self.get_free_list_head() == Some(block_id) {
    //         self.set_free_list_head(next_id);
    //     }

    //     // 再修改后继节点
    //     if let Some(next) = next_id {
    //         self.get_block_mut(next).prev_free_block_id = prev_id;
    //     } else {
    //         self.set_free_list_tail(prev_id);
    //     }
    // }

    pub fn check_matched_kvcache(&self, request: &Request) -> u32 {
        let mut cache_hit_block = 0;
        if let Some(hashes) = request.hashes.as_ref() {
            for hash in hashes {
                if self.get_cached_block(hash) {
                    cache_hit_block += 1;
                }
            }
        } else {
            return 0;
        }
        if request.prompt_len as usize == cache_hit_block * self.config.scheduler_config.block_size
        {
            // full hit
            cache_hit_block -= 1;
        }
        cache_hit_block as u32
    }

    /// Get the computed KV blocks for a request as Prefix cache
    pub fn get_computed_blocks(&mut self, request: &Request) -> Option<usize> {
        let mut cache_hit_block = 0;
        if let Some(hashes) = self.get_req_to_hashes(&request.request_id) {
            for hash in hashes {
                if self.get_cached_block(hash) {
                    cache_hit_block += 1;
                }
            }
        } else {
            return None;
        }
        if request.prompt_len as usize == cache_hit_block * self.config.scheduler_config.block_size
        {
            // full hit
            cache_hit_block -= 1;
        }

        // 2️⃣ touch 所有对应块（可变借用）
        // for &id in &block_indices {
        //     self.touch(id);
        // }

        // 3️⃣ 再次只读借用 self.blocks，返回引用
        // Note: We need to return references, but with COW we can't return references to COW state
        // So we collect block indices and return them, but this changes the API
        // For now, we'll return a Vec of owned blocks or change the return type
        // Actually, since we're in a mutable context and touching blocks, we can't return references anyway
        // This method signature needs to be reconsidered, but for now we'll work around it

        Some(cache_hit_block)
    }

    /// Allocate new blocks for a request
    /// only called when need to allocate new blocks
    pub fn allocate_block(&mut self, request_id: u64, num_new_blocks: usize) -> bool {
        // let mut allocated_hashes = Vec::new();

        // Todo: Doesn't allocate new block when last block isn't full
        //    when last block is full, compute its hash and allocate a new block

        // allocate num_new_blocks blocks from free list
        // Change:
        //  1. block
        //  2. cached_block
        //  3. req_to_hashes
        //  4. free_list_head/tail
        // let prev_free_block_head = self.free_list_head;
        // let prev_free_block_tail = self.free_list_tail;

        // let mut prev_block: Vec<(KVBlock, u64)> = Vec::new();

        // // All needed logic but not quite necessary for now
        // for _ in 0..num_new_blocks {
        //     if let Some(free_block_id) = self.get_free_list_head() {
        //         // Extract all needed values before making mutable borrows
        //         let (old_hash, next_id) = {
        //             let block = self.get_block_mut(free_block_id);
        //             block.ref_cnt = 1;
        //             (block.block_hash, block.next_free_block_id)
        //         };

        //         // evict the origin cached_block mapping if exists

        //         // important but not quite necessary
        //         if let Some(old_hash) = old_hash {
        //             let set = self.get_cached_block_mut(old_hash);
        //             set.remove(&free_block_id);
        //             if set.is_empty() {
        //                 self.remove_cached_block(&old_hash);
        //             }
        //             if let Some(evicted_ids) = self.evicted_blocks.as_mut() {
        //                 evicted_ids.push(free_block_id as u64);
        //             }
        //         }

        //         // block.block_hash = Some(block_hash);
        //         // self.cached_block
        //         //     .entry(block_hash)
        //         //     .or_insert_with(HashSet::new)
        //         //     .insert(free_block_id);
        //         // allocated_hashes.push(block_hash);
        //         self.set_free_list_head(next_id);
        //         if let Some(next_id) = next_id {
        //             self.get_block_mut(next_id).prev_free_block_id = None;
        //         } else {
        //             self.set_free_list_tail(None); // List is now empty
        //         };
        //         self.get_req_to_block_ids_mut(request_id).push(free_block_id);
        //     } else {
        //         // Handle cache full scenario (e.g., evict blocks)
        //         return false;
        //     }
        // }
        // true
        let allocated = self.get_req_to_block_ids(&request_id).unwrap().len();

        let slice: Vec<u64> = {
            let mut end = allocated + num_new_blocks;

            let hashes = self.get_req_to_hashes(&request_id).unwrap();
            end = end.min(hashes.len());
            hashes[allocated..end].iter().copied().collect()
        };

        // for hash in slice {
        //     self.get_cached_block_mut(hash);
        // }
        // let cached_block = &mut self.cow_state.as_mut().unwrap().cached_block;
        slice.iter().for_each(|hash| {
            self.uncommited_kvcache.insert(*hash);
        });
        true
    }

    /// Do resource cleanup for a finished request
    /// 1. decrement ref_cnt for all blocks(reversely) associated with the request
    /// 2. remvoe id -> hash, id -> bids
    pub fn free_request(&mut self, request_id: u64) {
        // Free all blocks associated with the request

        // if if_redo {
        self.free_request_block(request_id);
        self.remove_req_to_hashes(&request_id);
        // self.redo_stack.push_back(Operation::FreeRequest(request_id));
        // }
    }

    pub fn free_request_block(&mut self, request_id: u64) {
        // let mut prev_blocks = Vec::new();
        // let ids = if let Some(ids) = self.get_req_to_block_ids(&request_id) {
        //     ids.clone()
        // } else {
        //     return;
        // };
        self.remove_req_to_block_ids(&request_id);

        // for id in ids.iter().rev() {
        //     let block = self.get_block_mut(*id);

        //     // if if_redo {
        //     //     // used for redo
        //     //     prev_blocks.push(block.clone());
        //     // }
        //     // tracing::info!("freeing request_id: {}, block: {:?}", request_id, block);
        //     block.ref_cnt -= 1;
        //     if block.ref_cnt == 0 {
        //         // Add block back to free list
        //         // if let Some(block_hash) = block.block_hash {
        //         //     let set = self.cached_block.get_mut(&block_hash).unwrap();
        //         //     set.remove(&id);
        //         //     if set.is_empty() {
        //         //         self.cached_block.remove(&block_hash);
        //         //     }
        //         //     block.block_hash = None;
        //         // }
        //         // block.block_hash = None;

        //         self.insert_block_to_freelist(*id);
        //         // tracing::info!(
        //         //     "after freeing block and added to free list: {:?}",
        //         //     self.blocks[*id as usize]
        //         // );
        //     } else {
        //         // tracing::info!("after freeing block: {:?}", block);
        //     }
        // }
        // return None;
    }

    // Insert a block back to the free list
    // fn insert_block_to_freelist(&mut self, block_id: u32) {
    //     // tracing::info!("Inserting block {} back to free list", block_id);
    //     let tail_id = self.get_free_list_tail();

    //     // Update block fields
    //     {
    //         let block = self.get_block_mut(block_id);
    //         block.ref_cnt = 0;
    //         block.prev_free_block_id = tail_id;
    //         block.next_free_block_id = None;
    //     }

    //     // Update free list pointers
    //     if let Some(tail_id) = tail_id {
    //         self.get_block_mut(tail_id).next_free_block_id = Some(block_id);
    //     } else {
    //         self.set_free_list_head(Some(block_id)); // List was empty
    //     }
    //     self.set_free_list_tail(Some(block_id));
    // }
}

mod tests {
    use std::vec;

    use super::*;
    use crate::simulator::config::{
        DeviceConfig, ModelConfig, PredictorConfig, ReplicaConfig, SchedulerConfig,
        SimulationConfig,
    };
    use crate::simulator::predictor::TrainedPredictorType;
    use crate::{kvcache, simulator::batch::Request};

    // #[test]
    // fn test_kv_cache_manager() {
    //     let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
    //     let args = SimulationConfig {
    //         replica_config: ReplicaConfig { num_pipeline_stages: 1, tensor_parallel_size: 1 },
    //         scheduler_config: SchedulerConfig {
    //             token_budget: 1024,
    //             block_size: 16,
    //             num_blocks: 35172,
    //         },
    //         model_config: ModelConfig {
    //             num_layers: 32,
    //             num_attention_heads: 32,
    //             post_attn_norm: true,
    //             attention_prefill_batching_overhead_fraction: 0.1,
    //             attention_decode_batching_overhead_fraction: 0.4,
    //         },
    //         predictor_config: PredictorConfig {
    //             model_hash: "d29f0375".to_string(),
    //             predict_cache_path_prefix: "/nvme/zkx/Modified_vidur/cache".to_string(),
    //             trained_type: TrainedPredictorType::LinearRegression,
    //             learning_rate: Some(0.002),
    //             refine_threshold: 50.0,
    //             kv_cache_prediction_granularity: 64,
    //             flops_prediction_granularity: 1024,
    //             nccl_cpu_launch_overhead_ms: None,
    //             nccl_cpu_skew_overhead_per_device_ms: None,
    //             skip_cpu_overhead_modeling: true,
    //         },
    //         device_config: DeviceConfig {},
    //         fake_backend: false,
    //     };

    //     let mut kv_cache_manager = KVCacheManager::new(Arc::new(args));

    //     assert!(kv_cache_manager.blocks.len() == 35172);
    //     assert!(kv_cache_manager.free_list_head == Some(0));
    //     assert!(kv_cache_manager.free_list_tail == Some(35171));
    //     assert!(kv_cache_manager.cached_block.is_empty());
    //     assert!(kv_cache_manager.req_to_block_ids.is_empty());
    //     assert!(kv_cache_manager.req_to_hashes.is_empty());

    //     let hashes = vec![2u64, 3];
    //     let request = Request {
    //         request_id: 1,
    //         prompt_len: 47,
    //         generation_len: Some(16),
    //         processed_tokens: 0,
    //         arrival_time: None,
    //         num_token_per_output: 1,
    //         hashes: Some(hashes),
    //         max_generation_len: 16384,

    //         hit_token_cnt: 0,
    //         ttft: None,
    //     };

    //     kv_cache_manager.add_request(request.request_id, request.hashes.unwrap());

    //     // free_list_head -> 0 -> 1 -> ... -> 35171
    //     // req_to_block_ids: 1 -> []
    //     // req_to_hashes: 1 -> [2, 3]
    //     // cached_block: {}
    //     assert!(kv_cache_manager.req_to_hashes.len() == 1);
    //     assert!(kv_cache_manager.req_to_hashes[&1] == vec![2, 3]);
    //     assert!(kv_cache_manager.req_to_block_ids.len() == 1);
    //     assert!(kv_cache_manager.req_to_block_ids[&1].is_empty());
    //     assert!(kv_cache_manager.cached_block.is_empty());
    //     assert!(kv_cache_manager.free_list_head == Some(0));
    //     assert!(kv_cache_manager.free_list_tail == Some(35171));

    //     // free_list_head -> 2 -> 3 -> ... -> 35171
    //     // req_to_block_ids: 1 -> [0, 1, 2]
    //     // req_to_hashes: 1 -> [2, 3]
    //     // cached_block: 2 -> {0}, 3 -> {1}
    //     kv_cache_manager.set_req_block_ids(request.request_id, &vec![0u64, 1, 2]);
    //     assert!(kv_cache_manager.req_to_block_ids.len() == 1);
    //     assert!(kv_cache_manager.req_to_block_ids[&1] == vec![0, 1, 2]);
    //     assert!(kv_cache_manager.cached_block.len() == 2);
    //     assert!(kv_cache_manager.cached_block[&2].contains(&0));
    //     assert!(kv_cache_manager.cached_block[&3].contains(&1));
    //     assert!(kv_cache_manager.free_list_head == Some(3));

    //     // free_list_head -> 3 -> ... -> 35171
    //     // req_to_block_ids: 1 -> [0, 1, 2]
    //     // req_to_hashes: 1 -> [2, 3, 4]
    //     // cached_block: 2 -> {0}, 3 -> {1}, 4 -> {2}
    //     kv_cache_manager.req_update_hash(request.request_id, 4);
    //     assert!(kv_cache_manager.req_to_hashes[&1] == vec![2, 3, 4]);
    //     assert!(kv_cache_manager.req_to_block_ids[&1] == vec![0, 1, 2]);
    //     assert!(kv_cache_manager.cached_block[&4].contains(&2));
    //     assert!(kv_cache_manager.free_list_head == Some(3));

    //     // free_list_head -> 4 -> ... -> 35171
    //     // req_to_block_ids: 1 -> [0, 1, 2, 3]
    //     // req_to_hashes: 1 -> [2, 3, 4]
    //     // cached_block: 2 -> {0}, 3 -> {1}, 4 -> {2}
    //     kv_cache_manager.set_req_block_ids(request.request_id, &vec![0u64, 1, 2, 3]);
    //     assert!(kv_cache_manager.req_to_block_ids[&1] == vec![0, 1, 2, 3]);
    //     assert!(kv_cache_manager.free_list_head == Some(4));
    //     assert!(kv_cache_manager.cached_block[&4].contains(&2));

    //     // free_list_head -> 4 -> ... -> 35171
    //     // req_to_block_ids: 1 -> [0, 1, 2, 3]
    //     // req_to_hashes: 1 -> [2, 3, 4, 4]
    //     // cached_block: 2 -> {0}, 3 -> {1}, 4 -> {2, 3}
    //     kv_cache_manager.req_update_hash(request.request_id, 4);
    //     assert!(kv_cache_manager.req_to_hashes[&1] == vec![2, 3, 4, 4]);
    //     assert!(kv_cache_manager.req_to_block_ids[&1] == vec![0, 1, 2, 3]);
    //     assert!(kv_cache_manager.cached_block[&4].contains(&2));
    //     assert!(kv_cache_manager.cached_block[&4].contains(&3));
    //     assert!(kv_cache_manager.free_list_head == Some(4));

    //     let new_request = Request {
    //         request_id: 2,
    //         prompt_len: 20,
    //         generation_len: Some(10),
    //         processed_tokens: 0,
    //         arrival_time: None,
    //         num_token_per_output: 1,
    //         hashes: Some(vec![2u64, 6]),
    //         max_generation_len: 16384,

    //         hit_token_cnt: 0,
    //         ttft: None,
    //     };

    //     // free_list_head -> 4 -> ... -> 35171
    //     // req_to_block_ids: 1 -> [0, 1, 2, 3], 2 -> []
    //     // req_to_hashes: 1 -> [2, 3, 4, 4], 2 -> [2, 6]
    //     // cached_block: 2 -> {0}, 3 -> {1}, 4 -> {2, 3}
    //     kv_cache_manager.add_request(new_request.request_id, new_request.hashes.unwrap());

    //     assert!(kv_cache_manager.req_to_block_ids.len() == 2);
    //     assert!(kv_cache_manager.req_to_block_ids[&1] == vec![0, 1, 2, 3]);
    //     assert!(kv_cache_manager.req_to_block_ids[&2].is_empty());
    //     assert!(kv_cache_manager.req_to_hashes[&1] == vec![2, 3, 4, 4]);
    //     assert!(kv_cache_manager.req_to_hashes[&2] == vec![2, 6]);
    //     assert!(kv_cache_manager.free_list_head == Some(4));
    //     assert!(kv_cache_manager.cached_block.len() == 3);
    //     assert!(kv_cache_manager.cached_block[&2].contains(&0));
    //     assert!(kv_cache_manager.cached_block[&3].contains(&1));
    //     assert!(kv_cache_manager.cached_block[&4].contains(&2));
    //     assert!(kv_cache_manager.cached_block[&4].contains(&3));

    //     // free_list_head -> 5 -> ... -> 35171
    //     // req_to_block_ids: 1 -> [0, 1, 2, 3], 2 -> [0, 4]
    //     // req_to_hashes: 1 -> [2, 3, 4, 4], 2 -> [2, 6]
    //     // cached_block: 2 -> {0}, 3 -> {1}, 4 -> {2, 3}, 6 -> {4}
    //     kv_cache_manager.set_req_block_ids(new_request.request_id, &vec![0u64, 4]);
    //     assert!(kv_cache_manager.req_to_block_ids[&2] == vec![0, 4]);
    //     assert!(kv_cache_manager.cached_block[&2].contains(&0));
    //     assert!(kv_cache_manager.cached_block[&6].contains(&4));
    //     assert!(kv_cache_manager.free_list_head == Some(5));
    //     assert!(kv_cache_manager.blocks[0].ref_cnt == 2); // shared block
    //     assert!(kv_cache_manager.blocks[4].ref_cnt == 1); // unique block
    //     assert!(kv_cache_manager.blocks[1].ref_cnt == 1); // unique block
    //     assert!(kv_cache_manager.blocks[2].ref_cnt == 1); // unique block
    //     assert!(kv_cache_manager.blocks[3].ref_cnt == 1); // unique block

    //     // free_list_head -> 5 -> ... -> 35171 -> 3 -> 2 -> 1
    //     // req_to_block_ids: 2 -> [0, 4]
    //     // req_to_hashes: 2 -> [2, 6]
    //     // cached_block: 2 -> {0}, 6 -> {4}
    //     kv_cache_manager.free_request(request.request_id);
    //     assert!(kv_cache_manager.req_to_block_ids.len() == 1);
    //     assert!(kv_cache_manager.req_to_hashes.len() == 1);
    //     assert!(kv_cache_manager.blocks[0].ref_cnt == 1); // shared block
    //     assert!(kv_cache_manager.free_list_head == Some(5));
    //     assert!(kv_cache_manager.free_list_tail == Some(1));
    //     assert!(kv_cache_manager.blocks[1].ref_cnt == 0); // unique block
    //     assert!(kv_cache_manager.blocks[2].ref_cnt == 0); // unique block
    //     assert!(kv_cache_manager.blocks[3].ref_cnt == 0); // unique block
    //     assert!(kv_cache_manager.blocks[4].ref_cnt == 1); // unique block
    //     tracing::info!("cached_block: {:?}", kv_cache_manager.cached_block);
    //     assert!(kv_cache_manager.cached_block.get(&3).is_none());
    //     assert!(kv_cache_manager.cached_block[&2].contains(&0));
    //     assert!(kv_cache_manager.blocks[2].next_free_block_id == Some(1));
    //     assert!(kv_cache_manager.blocks[3].next_free_block_id == Some(2));

    //     // free_list_head 5 -> 35171 -> ... -> 3 -> 2 -> 1 -> 4 -> 0
    //     // req_to_block_ids: {}
    //     // req_to_hashes: {}
    //     // cached_block: {}
    //     kv_cache_manager.free_request(new_request.request_id);
    //     assert!(kv_cache_manager.req_to_block_ids.is_empty());
    //     assert!(kv_cache_manager.req_to_hashes.is_empty());
    //     assert!(kv_cache_manager.free_list_head == Some(5));
    //     assert!(kv_cache_manager.free_list_tail == Some(0));
    //     assert!(kv_cache_manager.blocks[0].ref_cnt == 0);
    //     assert!(kv_cache_manager.blocks[4].ref_cnt == 0);
    //     assert!(kv_cache_manager.cached_block.is_empty());
    //     assert!(kv_cache_manager.free_list_tail == Some(0));
    // }

    // // fn test_kv_cache_manager_allocate_block() {
    //     let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
    //     let args = SimulationConfig {
    //         replica_config: ReplicaConfig { num_pipeline_stages: 1, tensor_parallel_size: 1 },
    //         scheduler_config: SchedulerConfig {
    //             token_budget: 1024,
    //             block_size: 16,
    //             num_blocks: 10,
    //         },
    //         model_config: ModelConfig {
    //             num_layers: 32,
    //             num_attention_heads: 32,
    //             post_attn_norm: true,
    //             attention_prefill_batching_overhead_fraction: 0.1,
    //             attention_decode_batching_overhead_fraction: 0.4,
    //         },
    //         predictor_config: PredictorConfig {
    //             model_hash: "d29f0375".to_string(),
    //             predict_cache_path_prefix: "/nvme/zkx/Modified_vidur/cache".to_string(),
    //             trained_type: TrainedPredictorType::LinearRegression,
    //             learning_rate: Some(0.002),
    //             refine_threshold: 50.0,
    //             kv_cache_prediction_granularity: 64,
    //             flops_prediction_granularity: 1024,
    //             nccl_cpu_launch_overhead_ms: None,
    //             nccl_cpu_skew_overhead_per_device_ms: None,
    //             skip_cpu_overhead_modeling: true,
    //         },
    //         device_config: DeviceConfig {},

    //         fake_backend: false,
    //     };

    //     let mut kv_cache_manager = KVCacheManager::new(Arc::new(args));

    //     let request = Request {
    //         request_id: 1,
    //         prompt_len: 47,
    //         generation_len: Some(16),
    //         processed_tokens: 0,
    //         arrival_time: None,
    //         num_token_per_output: 1,
    //         hashes: Some(vec![]),
    //         max_generation_len: 16384,

    //         hit_token_cnt: 0,
    //         ttft: None,
    //     };
    //     kv_cache_manager.add_request(1, vec![0u64; 1]);

    //     let success = kv_cache_manager.allocate_block(1, 40);
    //     assert!(success);
    //     assert!(kv_cache_manager.req_to_hashes[&1].len() == 3); // 3 new blocks allocated

    //     let success = kv_cache_manager.allocate_block(1, 200);
    //     assert!(!success); // not enough blocks
    // }
}
