use dendra::{
    CompactionPolicyType, Query, RoutingPolicyType, VectorDB, VectorDBConfig, cosine_distance, math,
};
use env_logger;
use rand::{SeedableRng, rngs::StdRng};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    env_logger::init();
    let dir = PathBuf::from("my_vector_store");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    let dimension = 128;
    let config = VectorDBConfig::new(64, 2, 128, 42, 100, 2)
        .with_compaction_policy(CompactionPolicyType::SimilarityAware {
            overlap_threshold: 0.1,
        })
        .with_routing_policy(RoutingPolicyType::FlatTopK {
            max_segments: 8,
            min_segments: 2,
        });

    let mut store = VectorDB::new(dir, config);

    let dataset_start = Instant::now();
    let mut checkpoint_start = Instant::now();

    let data_seed = 424242u64;
    let cluster_count = 64usize;
    let query_source_index = 12_345usize;
    let query_vec = vector_at_index(query_source_index, dimension, data_seed, cluster_count);

    // Use deterministic clustered vectors on-demand.
    // The query is exactly one vector that already exists in the dataset
    // (the vector at query_source_index).
    let num_vectors = 10_000_000;
    for (i, vec) in
        VectorGenerator::new(num_vectors, dimension, data_seed, cluster_count).enumerate()
    {
        store.insert(&vec, i as u32).unwrap();
        if (i + 1) % 100_000 == 0 {
            let checkpoint_elapsed = checkpoint_start.elapsed();
            let avg_per_vec_ns = checkpoint_elapsed.as_nanos() / 100_000;
            log::info!(
                "Inserted {} vectors total | last 100000 took {:?} | avg {} ns/vector",
                i + 1,
                checkpoint_elapsed,
                avg_per_vec_ns
            );
            checkpoint_start = Instant::now();
        }
    }
    store.flush().unwrap();
    log::info!("Dataset insertion took: {:?}", dataset_start.elapsed());

    let start = Instant::now();
    store.save().unwrap();
    let elapsed = start.elapsed();
    log::info!("Store save took: {:?}", elapsed);

    let query = Query::new(query_vec, 100, cosine_distance, None);
    let mut results = Vec::new();

    let query_start = Instant::now();
    let _ = store.query(&query, &mut results).unwrap();
    log::info!("Query took: {:?}", query_start.elapsed());
    log::info!("Found {} results", results.len());
    for (id, dist) in results.into_iter().take(10) {
        log::info!("ID: {}, Distance: {}", id, dist);
    }
}

/// Lazy iterator that generates vectors on-demand without storing them all in memory.
/// This allows streaming massive datasets without exhausting RAM.
struct VectorGenerator {
    total: usize,
    current: usize,
    dim: usize,
    cluster_count: usize,
    cluster_block_size: usize,
    cluster_centers: Vec<Vec<f32>>,
    rng: StdRng,
}

impl VectorGenerator {
    fn new(total: usize, dim: usize, seed: u64, cluster_count: usize) -> Self {
        let mut center_rng = StdRng::seed_from_u64(seed ^ 0xC1A0_5EED);
        let mut cluster_centers = Vec::with_capacity(cluster_count);
        for _ in 0..cluster_count {
            cluster_centers.push(math::random_unit_vector(dim, &mut center_rng).unwrap());
        }

        Self {
            total,
            current: 0,
            dim,
            cluster_count,
            cluster_block_size: 50_000,
            cluster_centers,
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

        // Generate locality: vectors arrive in cluster blocks, not round-robin.
        let cluster_idx = (self.current / self.cluster_block_size) % self.cluster_count;
        let center = &self.cluster_centers[cluster_idx];
        let noise = math::random_unit_vector(self.dim, &mut self.rng).unwrap();

        // Blend cluster center + random noise, then renormalize.
        let mut vec = vec![0.0; self.dim];
        for i in 0..self.dim {
            vec[i] = 0.85 * center[i] + 0.15 * noise[i];
        }
        let norm = math::l2_norm(&vec);
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        self.current += 1;
        Some(vec)
    }
}

fn vector_at_index(index: usize, dim: usize, seed: u64, cluster_count: usize) -> Vec<f32> {
    VectorGenerator::new(index + 1, dim, seed, cluster_count)
        .nth(index)
        .unwrap()
}
