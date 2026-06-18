use serde::{Deserialize, Serialize};

const DEFAULT_DIMENSION: usize = 128;
const DEFAULT_LSH_NUM_TABLES: usize = 8;
const DEFAULT_LSH_BITS_PER_TABLE: usize = 16;
const DEFAULT_LSH_DIMS_PER_BIT: usize = 8;
const DEFAULT_LSH_PROBE_HAMMING_RADIUS: u8 = 1;
const DEFAULT_LSH_BUCKET_EXPERT_DIMS: usize = 4;
const DEFAULT_LSH_MIN_CANDIDATES: usize = 384;
const DEFAULT_LSH_MAX_CANDIDATES: usize = 2048;
const DEFAULT_LSH_ADAPTIVE_GAMMA: f32 = 2.2;
const DEFAULT_SEGMENT_CAPACITY_MB: usize = 100;
const DEFAULT_DELTA: f32 = 0.05;
const DEFAULT_NUM_WORKERS: usize = 2;
const DEFAULT_SEAL_QUEUE_CAPACITY: usize = 2;
const DEFAULT_COMPACTION_DEPTH_CAP: u8 = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    pub(crate) dimension: usize,
    pub(crate) lsh_num_tables: usize,
    pub(crate) lsh_bits_per_table: usize,
    pub(crate) lsh_dims_per_bit: usize,
    pub(crate) lsh_probe_hamming_radius: u8,
    pub(crate) lsh_bucket_expert_dims: usize,
    pub(crate) lsh_min_candidates: usize,
    pub(crate) lsh_max_candidates: usize,
    pub(crate) lsh_adaptive_gamma: f32,
    pub(crate) segment_capacity_mb: usize,
    pub(crate) delta: f32,
    pub(crate) seed: u64,
    pub(crate) num_workers: usize,
    pub(crate) seal_queue_capacity: usize,
    pub(crate) compaction_depth_cap: u8,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            dimension: DEFAULT_DIMENSION,
            lsh_num_tables: DEFAULT_LSH_NUM_TABLES,
            lsh_bits_per_table: DEFAULT_LSH_BITS_PER_TABLE,
            lsh_dims_per_bit: DEFAULT_LSH_DIMS_PER_BIT,
            lsh_probe_hamming_radius: DEFAULT_LSH_PROBE_HAMMING_RADIUS,
            lsh_bucket_expert_dims: DEFAULT_LSH_BUCKET_EXPERT_DIMS,
            lsh_min_candidates: DEFAULT_LSH_MIN_CANDIDATES,
            lsh_max_candidates: DEFAULT_LSH_MAX_CANDIDATES,
            lsh_adaptive_gamma: DEFAULT_LSH_ADAPTIVE_GAMMA,
            segment_capacity_mb: DEFAULT_SEGMENT_CAPACITY_MB,
            delta: DEFAULT_DELTA,
            seed: 42,
            num_workers: DEFAULT_NUM_WORKERS,
            seal_queue_capacity: DEFAULT_SEAL_QUEUE_CAPACITY,
            compaction_depth_cap: DEFAULT_COMPACTION_DEPTH_CAP,
        }
    }
}

impl EngineConfig {
    pub fn new(
        dimension: usize,
        seed: u64,
        segment_capacity_mb: usize,
        seal_queue_capacity: usize,
    ) -> Self {
        Self {
            dimension,
            lsh_num_tables: DEFAULT_LSH_NUM_TABLES,
            lsh_bits_per_table: DEFAULT_LSH_BITS_PER_TABLE,
            lsh_dims_per_bit: DEFAULT_LSH_DIMS_PER_BIT,
            lsh_probe_hamming_radius: DEFAULT_LSH_PROBE_HAMMING_RADIUS,
            lsh_bucket_expert_dims: DEFAULT_LSH_BUCKET_EXPERT_DIMS,
            lsh_min_candidates: DEFAULT_LSH_MIN_CANDIDATES,
            lsh_max_candidates: DEFAULT_LSH_MAX_CANDIDATES,
            lsh_adaptive_gamma: DEFAULT_LSH_ADAPTIVE_GAMMA,
            seed,
            segment_capacity_mb,
            seal_queue_capacity,
            ..Self::default()
        }
    }

    pub fn with_lsh_tables(mut self, tables: usize) -> Self {
        self.lsh_num_tables = tables.max(1);
        self
    }

    pub fn with_lsh_bits(mut self, bits: usize) -> Self {
        self.lsh_bits_per_table = bits.clamp(1, 64);
        self
    }

    pub fn with_lsh_dims_per_bit(mut self, dims_per_bit: usize) -> Self {
        self.lsh_dims_per_bit = dims_per_bit.max(1);
        self
    }

    pub fn with_lsh_probe_hamming_radius(mut self, radius: u8) -> Self {
        self.lsh_probe_hamming_radius = radius.min(2);
        self
    }

    pub fn with_lsh_bucket_expert_dims(mut self, dims: usize) -> Self {
        self.lsh_bucket_expert_dims = dims.max(1);
        self
    }

    pub fn with_lsh_max_candidates(mut self, max_candidates: usize) -> Self {
        self.lsh_max_candidates = max_candidates.max(64);
        if self.lsh_min_candidates > self.lsh_max_candidates {
            self.lsh_min_candidates = self.lsh_max_candidates;
        }
        self
    }

    pub fn with_lsh_min_candidates(mut self, min_candidates: usize) -> Self {
        self.lsh_min_candidates = min_candidates.max(32).min(self.lsh_max_candidates);
        self
    }

    pub fn with_lsh_adaptive_gamma(mut self, gamma: f32) -> Self {
        self.lsh_adaptive_gamma = gamma.clamp(0.5, 4.0);
        self
    }

    pub fn with_delta(mut self, delta: f32) -> Self {
        self.delta = delta.clamp(0.0, 1.0);
        self
    }

    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers.max(1);
        self
    }

    pub fn with_seal_queue_capacity(mut self, capacity: usize) -> Self {
        self.seal_queue_capacity = capacity.max(1);
        self
    }

    pub fn with_compaction_depth_cap(mut self, cap: u8) -> Self {
        self.compaction_depth_cap = cap;
        self
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn max_active_vectors(&self) -> usize {
        let bytes = self.segment_capacity_mb.saturating_mul(1024 * 1024);
        let bytes_per_vector = self.dimension.saturating_mul(std::mem::size_of::<f32>());
        if bytes_per_vector == 0 {
            return 0;
        }
        bytes / bytes_per_vector
    }
}
