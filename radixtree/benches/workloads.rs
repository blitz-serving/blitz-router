// Criterion benchmarks for progressive radix tree optimization levels.
//
// Workloads model the KV cache block hash tracking use case:
// - Key type: u64 (block hash)
// - Key length: 16-64 elements (typical prompt token count / block_size)
// - Alphabet: full u64 range (hash values)

mod lowering_levels;

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use lowering_levels::RadixTree;
use std::hint::black_box as bb;

// ============================================================================
// Workload generators
// ============================================================================

/// Generate N keys of length `key_len` with sequential prefixes.
/// Keys share progressively shorter prefixes, mimicking conversation continuations.
fn gen_sequential_keys(n: usize, key_len: usize) -> Vec<Vec<u64>> {
    let mut keys = Vec::with_capacity(n);
    for i in 0..n {
        let mut key = Vec::with_capacity(key_len);
        // First half is shared prefix based on i / stride
        let stride = (n / 16).max(1);
        let prefix_val = (i / stride) as u64;
        for j in 0..key_len {
            if j < key_len / 2 {
                key.push(prefix_val * 1000 + j as u64);
            } else {
                key.push(i as u64 * 100 + j as u64);
            }
        }
        keys.push(key);
    }
    keys
}

/// Generate N keys with pseudo-random u64 elements (deterministic via simple LCG).
fn gen_random_keys(n: usize, key_len: usize, seed: u64) -> Vec<Vec<u64>> {
    let mut keys = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        let mut key = Vec::with_capacity(key_len);
        for _ in 0..key_len {
            // Simple LCG: state = state * 6364136223846793005 + 1442695040888963407
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            key.push(state);
        }
        keys.push(key);
    }
    keys
}

/// Generate payloads matching keys (same length, sequential block IDs).
fn gen_payloads(keys: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let mut block_id = 0u64;
    keys.iter()
        .map(|k| {
            let p: Vec<u64> = (block_id..block_id + k.len() as u64).collect();
            block_id += k.len() as u64;
            p
        })
        .collect()
}

// ============================================================================
// Benchmark: Sequential Insert
// ============================================================================

fn bench_sequential_insert<T: RadixTree>(keys: &[Vec<u64>], payloads: &[Vec<u64>]) {
    let mut tree = T::new();
    for (k, p) in keys.iter().zip(payloads.iter()) {
        tree.insert(black_box(k), black_box(p));
    }
    black_box(&tree);
}

fn sequential_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_insert");

    for &n in &[1000, 10000] {
        let key_len = 32;
        let keys = gen_sequential_keys(n, key_len);
        let payloads = gen_payloads(&keys);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("L0_safe", n), &n, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l0::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L1_sorted", n), &n, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l1::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L2_rawptr", n), &n, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l2::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L3_split", n), &n, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l3::RadixTreeMap>(&keys, &payloads))
        });
    }
    group.finish();
}

// ============================================================================
// Benchmark: Random Insert
// ============================================================================

fn random_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_insert");

    for &n in &[1000, 10000] {
        let key_len = 32;
        let keys = gen_random_keys(n, key_len, 42);
        let payloads = gen_payloads(&keys);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("L0_safe", n), &n, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l0::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L1_sorted", n), &n, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l1::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L2_rawptr", n), &n, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l2::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L3_split", n), &n, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l3::RadixTreeMap>(&keys, &payloads))
        });
    }
    group.finish();
}

// ============================================================================
// Benchmark: Prefix Query (after bulk insert)
// ============================================================================

fn bench_prefix_query<T: RadixTree>(
    keys: &[Vec<u64>],
    payloads: &[Vec<u64>],
    queries: &[Vec<u64>],
) {
    let mut tree = T::new();
    for (k, p) in keys.iter().zip(payloads.iter()) {
        tree.insert(k, p);
    }
    for q in queries {
        black_box(tree.prefix_len(black_box(q)));
    }
}

fn prefix_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_query");

    let n = 10000;
    let key_len = 32;
    let m = 1000; // number of queries

    let keys = gen_sequential_keys(n, key_len);
    let payloads = gen_payloads(&keys);

    // Queries: mix of existing keys (exact match), prefixes, and random keys
    let mut queries = Vec::with_capacity(m);
    for i in 0..m {
        if i % 3 == 0 {
            // Exact match query — use existing key
            queries.push(keys[i % n].clone());
        } else if i % 3 == 1 {
            // Prefix query — truncate an existing key
            let k = &keys[i % n];
            let prefix_len = (k.len() / 2).max(1);
            queries.push(k[..prefix_len].to_vec());
        } else {
            // Random miss query
            queries.push(gen_random_keys(1, key_len, i as u64 * 997)[0].clone());
        }
    }

    group.throughput(Throughput::Elements(m as u64));

    group.bench_function("L0_safe", |b| {
        b.iter(|| {
            bench_prefix_query::<lowering_levels::l0::RadixTreeMap>(&keys, &payloads, &queries)
        })
    });
    group.bench_function("L1_sorted", |b| {
        b.iter(|| {
            bench_prefix_query::<lowering_levels::l1::RadixTreeMap>(&keys, &payloads, &queries)
        })
    });
    group.bench_function("L2_rawptr", |b| {
        b.iter(|| {
            bench_prefix_query::<lowering_levels::l2::RadixTreeMap>(&keys, &payloads, &queries)
        })
    });
    group.bench_function("L3_split", |b| {
        b.iter(|| {
            bench_prefix_query::<lowering_levels::l3::RadixTreeMap>(&keys, &payloads, &queries)
        })
    });

    group.finish();
}

// ============================================================================
// Benchmark: Mixed insert + query (realistic cache-tracking pattern)
// ============================================================================

fn bench_mixed<T: RadixTree>(keys: &[Vec<u64>], payloads: &[Vec<u64>]) {
    let mut tree = T::new();
    for (i, (k, p)) in keys.iter().zip(payloads.iter()).enumerate() {
        tree.insert(black_box(k), black_box(p));
        // Every 4th insert, do a prefix query on a previous key
        if i >= 4 && i % 4 == 0 {
            let query_idx = i / 2;
            black_box(tree.prefix_len(black_box(&keys[query_idx])));
        }
    }
    black_box(&tree);
}

fn mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed");

    for &n in &[1000, 10000] {
        let key_len = 32;
        let keys = gen_sequential_keys(n, key_len);
        let payloads = gen_payloads(&keys);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("L0_safe", n), &n, |b, _| {
            b.iter(|| bench_mixed::<lowering_levels::l0::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L1_sorted", n), &n, |b, _| {
            b.iter(|| bench_mixed::<lowering_levels::l1::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L2_rawptr", n), &n, |b, _| {
            b.iter(|| bench_mixed::<lowering_levels::l2::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L3_split", n), &n, |b, _| {
            b.iter(|| bench_mixed::<lowering_levels::l3::RadixTreeMap>(&keys, &payloads))
        });
    }
    group.finish();
}

// ============================================================================
// Benchmark: Variable key lengths
// ============================================================================

fn variable_key_len(c: &mut Criterion) {
    let mut group = c.benchmark_group("key_length");

    let n = 5000;

    for &key_len in &[16, 32, 64] {
        let keys = gen_sequential_keys(n, key_len);
        let payloads = gen_payloads(&keys);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("L0_safe", key_len), &key_len, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l0::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L1_sorted", key_len), &key_len, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l1::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L2_rawptr", key_len), &key_len, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l2::RadixTreeMap>(&keys, &payloads))
        });
        group.bench_with_input(BenchmarkId::new("L3_split", key_len), &key_len, |b, _| {
            b.iter(|| bench_sequential_insert::<lowering_levels::l3::RadixTreeMap>(&keys, &payloads))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    sequential_insert,
    random_insert,
    prefix_query,
    mixed_workload,
    variable_key_len,
);
criterion_main!(benches);
