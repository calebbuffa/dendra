use fvdb::{cosine_distance, math, Query, VectorDB, VectorDBConfig};
use rand::{rngs::StdRng, SeedableRng};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let dir = PathBuf::from("my_vector_store");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    let dimension = 128;
    let config = VectorDBConfig::new(32, 4, 128, 42, 100, 2);

    let mut store = VectorDB::new(dir, config);

    let dataset_start = Instant::now();
    let mut checkpoint_start = Instant::now();

    let data_seed = 424242u64;
    let query_source_index = 12_345usize;
    let query_vec = vector_at_index(query_source_index, dimension, data_seed);

    // Use deterministic random unit vectors on-demand.
    // The query is exactly one vector that already exists in the dataset
    // (the vector at query_source_index).
    let num_vectors = 1_000_000;
    for (i, vec) in VectorGenerator::new(num_vectors, dimension, data_seed).enumerate() {
        store.insert(&vec, i as u32).unwrap();
        if (i + 1) % 100_000 == 0 {
            let checkpoint_elapsed = checkpoint_start.elapsed();
            let avg_per_vec_ns = checkpoint_elapsed.as_nanos() / 100_000;
            println!(
                "Inserted {} vectors total | last 100000 took {:?} | avg {} ns/vector",
                i + 1,
                checkpoint_elapsed,
                avg_per_vec_ns
            );
            checkpoint_start = Instant::now();
        }
    }
    store.flush().unwrap();
    println!("Dataset insertion took: {:?}", dataset_start.elapsed());

    let start = Instant::now();
    store.save().unwrap();
    let elapsed = start.elapsed();
    println!("Store save took: {:?}", elapsed);

    let query = Query::new(query_vec, 100, cosine_distance, None);
    let mut results = Vec::new();

    let query_start = Instant::now();
    let _ = store.query(&query, &mut results).unwrap();
    println!("Query took: {:?}", query_start.elapsed());
    println!("Found {} results", results.len());
    for (id, dist) in results {
        println!("ID: {}, Distance: {}", id, dist);
    }
}

/// Lazy iterator that generates vectors on-demand without storing them all in memory.
/// This allows streaming massive datasets without exhausting RAM.
struct VectorGenerator {
    total: usize,
    current: usize,
    dim: usize,
    rng: StdRng,
}

impl VectorGenerator {
    fn new(total: usize, dim: usize, seed: u64) -> Self {
        Self {
            total,
            current: 0,
            dim,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Iterator for VectorGenerator {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total {
            return None;
        }

        let vec = math::random_unit_vector(self.dim, &mut self.rng).unwrap();

        self.current += 1;
        Some(vec)
    }
}

fn vector_at_index(index: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v = Vec::new();
    for _ in 0..=index {
        v = math::random_unit_vector(dim, &mut rng).unwrap();
    }
    v
}
