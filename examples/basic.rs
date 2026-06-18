use engram::{EngramError, Query, QueryScratch, VectorDB, VectorDBConfig, cosine_distance};
use env_logger;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    env_logger::init();
    let dir = PathBuf::from("my_vector_store");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    let dimension = 128;
    let config = VectorDBConfig::new(dimension, 42, 100, 2)
        .with_lsh_tables(8)
        .with_lsh_bits(16)
        .with_lsh_dims_per_bit(8)
        .with_lsh_probe_hamming_radius(1)
        .with_lsh_bucket_expert_dims(4)
        .with_lsh_min_candidates(384)
        .with_lsh_max_candidates(2048)
        .with_lsh_adaptive_gamma(2.2);

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

    let query = Query::new(query_vec, 100).with_metric(cosine_distance);
    let mut scratch = QueryScratch::default();
    let mut results = Vec::with_capacity(query.k());

    let query_start = Instant::now();
    let _ = store.query(&query, &mut scratch, &mut results).unwrap();
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
            cluster_centers.push(random_unit_vector(dim, &mut center_rng).unwrap());
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
        let noise = random_unit_vector(self.dim, &mut self.rng).unwrap();

        // Blend cluster center + random noise, then renormalize.
        let mut vec = vec![0.0; self.dim];
        for i in 0..self.dim {
            vec[i] = 0.85 * center[i] + 0.15 * noise[i];
        }
        let norm = l2_norm(&vec);
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

fn random_standard_normal_vector(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim)
        .map(|_| {
            let x: f32 = StandardNormal.sample(rng);
            x
        })
        .collect()
}

fn random_unit_vector(dim: usize, rng: &mut impl Rng) -> Result<Vec<f32>, EngramError> {
    let mut v = random_standard_normal_vector(dim, rng);
    normalize(&mut v)?;
    Ok(v)
}

/// Normalize vector in-place.
fn normalize(a: &mut [f32]) -> Result<(), EngramError> {
    let norm = l2_norm(a);
    if norm == 0.0 {
        return Err(EngramError::ZeroNormVector);
    }
    let inv_norm = 1.0 / norm;
    for x in a.iter_mut() {
        *x *= inv_norm;
    }
    Ok(())
}

fn l2_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}
