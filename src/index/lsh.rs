//! BayesianLsh — SimHash tables with NIG-tracked statistics.
//!
//! Rather than building routing trees (O(N log N) per tree, tree-by-tree),
//! we maintain a set of SimHash tables:
//!
//!   signature = [sign(dot(r_0, x)), …, sign(dot(r_{B-1}, x))]  -> B-bit int
//!
//! Each table maps a signature to the row-ids of all vectors that hashed
//! there.  Candidate generation is:
//!
//!   query -> signature per table
//!         -> probe exact bucket + Hamming-1 neighbours
//!         -> union of row-ids (already O(1) per table)
//!
//! Build cost: O(N × tables × bits × dim)
//!   e.g. 200K × 16 × 16 × 128 = 6.5 B flops  (~130 ms at 50 GFLOPS)

use crate::index::{NigPrior, NigStats};
use crate::math::sparse_weighted_dot;
use log::debug;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone, Copy, Default)]
struct RowAccum {
    score: f32,
    votes: u16,
}

#[derive(Default)]
pub(crate) struct RouteScratch {
    row_accum: HashMap<u32, RowAccum>,
    sigs: Vec<u64>,
    query_proj: Vec<f32>,
    scored: Vec<(u32, f32, u16)>,
}

/// One SimHash table: `bits_per_table` random hyperplanes -> 64-bit bucket key.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct BitPlane {
    indices: Vec<usize>,
    weights: Vec<f32>,
    bias: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BucketExpert {
    count: u32,
    stats: Vec<NigStats>,
}

/// One sparse-projection LSH table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct LshTable {
    /// Random sparse projection planes.
    planes: Vec<BitPlane>,
    /// Additional sparse projections used only for bucket-level Bayesian scoring.
    expert_planes: Vec<BitPlane>,
    bits_per_table: usize,
    dims_per_bit: usize,
    expert_dims: usize,
    dim: usize,
    /// signature -> sorted list of row_ids that hashed there.
    buckets: HashMap<u64, Vec<u32>>,
    /// signature -> bucket expert sufficient statistics.
    experts: HashMap<u64, BucketExpert>,
}

impl LshTable {
    /// Build an empty table with freshly sampled Gaussian hyperplanes.
    pub(crate) fn new(
        bits_per_table: usize,
        dim: usize,
        dims_per_bit: usize,
        expert_dims: usize,
        rng: &mut impl Rng,
    ) -> Self {
        use rand_distr::{Distribution, StandardNormal};
        // u64 signatures can encode at most 64 bits.
        let bits_per_table = bits_per_table.clamp(1, 64);
        let dims_per_bit = dims_per_bit.clamp(1, dim.max(1));
        let expert_dims = expert_dims.max(1);

        // Sparse planes are much faster than dense dim-wide dots and preserve
        // enough angular locality for candidate generation.
        let mut planes = Vec::with_capacity(bits_per_table);
        let mut dim_pool: Vec<usize> = (0..dim.max(1)).collect();
        for _ in 0..bits_per_table {
            let mut indices = Vec::with_capacity(dims_per_bit);
            let mut weights = Vec::with_capacity(dims_per_bit);

            // Sample dimensions without replacement to avoid duplicate-index
            // collapse in sparse planes (which severely reduces hash entropy).
            for i in 0..dims_per_bit {
                let span = dim_pool.len() - i;
                let j = i + ((rng.next_u32() as usize) % span);
                dim_pool.swap(i, j);
                indices.push(dim_pool[i]);

                let w: f32 = StandardNormal.sample(rng);
                weights.push(w);
            }
            let norm = weights.iter().map(|w| w * w).sum::<f32>().sqrt().max(1e-9);
            for w in &mut weights {
                *w /= norm;
            }

            let z = <StandardNormal as Distribution<f32>>::sample(&StandardNormal, rng);
            let bias: f32 = 0.05f32 * z;
            planes.push(BitPlane {
                indices,
                weights,
                bias,
            });
        }

        let mut expert_planes = Vec::with_capacity(expert_dims);
        for _ in 0..expert_dims {
            let mut indices = Vec::with_capacity(dims_per_bit);
            let mut weights = Vec::with_capacity(dims_per_bit);
            for i in 0..dims_per_bit {
                let span = dim_pool.len() - i;
                let j = i + ((rng.next_u32() as usize) % span);
                dim_pool.swap(i, j);
                indices.push(dim_pool[i]);

                let w: f32 = StandardNormal.sample(rng);
                weights.push(w);
            }
            let norm = weights.iter().map(|w| w * w).sum::<f32>().sqrt().max(1e-9);
            for w in &mut weights {
                *w /= norm;
            }
            let z = <StandardNormal as Distribution<f32>>::sample(&StandardNormal, rng);
            expert_planes.push(BitPlane {
                indices,
                weights,
                bias: 0.02f32 * z,
            });
        }

        Self {
            planes,
            expert_planes,
            bits_per_table,
            dims_per_bit,
            expert_dims,
            dim,
            buckets: HashMap::new(),
            experts: HashMap::new(),
        }
    }

    /// Compute the B-bit SimHash signature of `vector`.
    #[inline]
    pub(crate) fn signature(&self, vector: &[f32]) -> u64 {
        let mut sig: u64 = 0;
        for (b, plane) in self.planes.iter().enumerate() {
            let dot = sparse_weighted_dot(vector, &plane.indices, &plane.weights, plane.bias);
            if dot > 0.0 {
                sig |= 1u64 << b;
            }
        }
        sig
    }

    /// Insert a vector by row_id into its bucket (no NIG stats stored here).
    #[allow(dead_code)]
    pub(crate) fn insert(&mut self, row_id: u32, vector: &[f32]) {
        let sig = self.signature(vector);
        self.buckets.entry(sig).or_default().push(row_id);
    }

    /// Return all row_ids from the exact bucket and (optionally) all
    /// Hamming-distance-1 neighbours.
    #[allow(dead_code)]
    pub(crate) fn probe(&self, query: &[f32], hamming_radius: u8, out: &mut Vec<u32>) {
        let sig = self.signature(query);
        if let Some(ids) = self.buckets.get(&sig) {
            out.extend_from_slice(ids);
        }

        if hamming_radius >= 1 {
            for bit in 0..self.bits_per_table {
                let neighbour = sig ^ (1u64 << bit);
                if let Some(ids) = self.buckets.get(&neighbour) {
                    out.extend_from_slice(ids);
                }
            }
        }

        if hamming_radius >= 2 {
            for i in 0..self.bits_per_table {
                for j in (i + 1)..self.bits_per_table {
                    let neighbour = sig ^ (1u64 << i) ^ (1u64 << j);
                    if let Some(ids) = self.buckets.get(&neighbour) {
                        out.extend_from_slice(ids);
                    }
                }
            }
        }
    }

    fn probe_signatures(&self, query: &[f32], hamming_radius: u8, out: &mut Vec<u64>) {
        let sig = self.signature(query);
        out.push(sig);

        if hamming_radius >= 1 {
            for bit in 0..self.bits_per_table {
                out.push(sig ^ (1u64 << bit));
            }
        }
        if hamming_radius >= 2 {
            for i in 0..self.bits_per_table {
                for j in (i + 1)..self.bits_per_table {
                    out.push(sig ^ (1u64 << i) ^ (1u64 << j));
                }
            }
        }
    }

    fn bucket_ids(&self, sig: u64) -> Option<&[u32]> {
        self.buckets.get(&sig).map(|v| v.as_slice())
    }

    fn bucket_expert(&self, sig: u64) -> Option<&BucketExpert> {
        self.experts.get(&sig)
    }

    fn project_expert(&self, vector: &[f32], out: &mut [f32]) {
        for (i, plane) in self.expert_planes.iter().enumerate() {
            out[i] = sparse_weighted_dot(vector, &plane.indices, &plane.weights, plane.bias);
        }
    }

    pub(crate) fn num_buckets(&self) -> usize {
        self.buckets.len()
    }
}

/// Collection of SimHash tables with bucket-level Bayesian experts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BayesianLsh {
    pub(crate) tables: Vec<LshTable>,
    pub(crate) prior: NigPrior,
    pub(crate) dim: usize,
    pub(crate) bits_per_table: usize,
    pub(crate) num_tables: usize,
    pub(crate) dims_per_bit: usize,
    pub(crate) probe_hamming_radius: u8,
    pub(crate) bucket_expert_dims: usize,
    pub(crate) min_candidates: usize,
    pub(crate) max_candidates: usize,
    pub(crate) adaptive_gamma: f32,
    pub(crate) total_rows: usize,
    /// Global per-dim NIG sufficient statistics across all indexed vectors.
    pub(crate) global_stats: Vec<NigStats>,
}

impl BayesianLsh {
    /// Build a BayesianLsh index from a flat row-major `vectors` slice.
    ///
    /// - `bits_per_table`:  B in [8, 20].  More bits -> finer buckets.
    /// - `num_tables`:      independent hash tables.
    /// - `seed`:            deterministic RNG seed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        vectors: &[f32],
        dim: usize,
        bits_per_table: usize,
        num_tables: usize,
        dims_per_bit: usize,
        probe_hamming_radius: u8,
        bucket_expert_dims: usize,
        min_candidates: usize,
        max_candidates: usize,
        adaptive_gamma: f32,
        seed: u64,
    ) -> Self {
        let t0 = Instant::now();
        let count = vectors.len() / dim.max(1);
        let bits_per_table = bits_per_table.clamp(1, 64);
        let num_tables = num_tables.max(1);
        let dims_per_bit = dims_per_bit.clamp(1, dim.max(1));
        let probe_hamming_radius = probe_hamming_radius.min(2);
        let bucket_expert_dims = bucket_expert_dims.max(1);
        let max_candidates = max_candidates.max(64);
        let min_candidates = min_candidates.max(32).min(max_candidates);
        let adaptive_gamma = adaptive_gamma.clamp(0.5, 4.0);

        // Each table gets an independent seed so tables are diverse.
        let mut tables: Vec<LshTable> = (0..num_tables)
            .map(|i| {
                let ts = seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                LshTable::new(
                    bits_per_table,
                    dim,
                    dims_per_bit,
                    bucket_expert_dims,
                    &mut StdRng::seed_from_u64(ts),
                )
            })
            .collect();

        let mut sum = vec![0.0f64; dim];
        let mut sumsq = vec![0.0f64; dim];

        // Stage signatures per table, then sort/pack once. This avoids millions
        // of random HashMap entry lookups on the hot path.
        let mut pairs_per_table: Vec<Vec<(u64, u32)>> =
            (0..num_tables).map(|_| Vec::with_capacity(count)).collect();

        // Vectors are read sequentially (row-major order), which is the same
        // forward-scan pattern we established for NIG fitting in the tree.
        // Per-table insertion calls `signature()` which reads the hyperplane
        // matrix (small, stays in L2/L3) and the current vector row (just read,
        // likely still warm in L1/L2).
        for (row_id, row) in (0u32..).zip(vectors.chunks_exact(dim)) {
            for (j, &v) in row.iter().enumerate() {
                let x = v as f64;
                sum[j] += x;
                sumsq[j] += x * x;
            }

            for (t_idx, table) in tables.iter().enumerate() {
                let sig = table.signature(row);
                pairs_per_table[t_idx].push((sig, row_id));
            }
        }

        for (table, pairs) in tables.iter_mut().zip(pairs_per_table.iter_mut()) {
            pairs.sort_unstable_by_key(|(sig, _)| *sig);

            let mut buckets: HashMap<u64, Vec<u32>> = HashMap::new();
            let mut i = 0usize;
            while i < pairs.len() {
                let sig = pairs[i].0;
                let start = i;
                i += 1;
                while i < pairs.len() && pairs[i].0 == sig {
                    i += 1;
                }

                let mut ids = Vec::with_capacity(i - start);
                for (_, row_id) in &pairs[start..i] {
                    ids.push(*row_id);
                }
                buckets.insert(sig, ids);
            }
            table.buckets = buckets;

            // Build per-bucket Bayesian experts on table-specific sparse projections.
            let mut experts: HashMap<u64, BucketExpert> = HashMap::new();
            let mut proj = vec![0.0f32; table.expert_dims];
            for (sig, ids) in &table.buckets {
                let mut stats = vec![NigStats::new(); table.expert_dims];
                for row_id in ids {
                    let base = *row_id as usize * dim;
                    let row = &vectors[base..base + dim];
                    table.project_expert(row, &mut proj);
                    for (j, s) in stats.iter_mut().enumerate() {
                        s.update(proj[j]);
                    }
                }
                experts.insert(
                    *sig,
                    BucketExpert {
                        count: ids.len() as u32,
                        stats,
                    },
                );
            }
            table.experts = experts;
        }

        let n = count as f64;
        let global_dim_stats = if count == 0 {
            vec![NigStats::new(); dim]
        } else {
            (0..dim)
                .map(|j| {
                    let mean = sum[j] / n;
                    let m2 = (sumsq[j] - sum[j] * mean).max(0.0);
                    NigStats { n, mean, m2 }
                })
                .collect::<Vec<_>>()
        };

        let total_buckets: usize = tables.iter().map(|t| t.num_buckets()).sum();
        debug!(
            "bayesian lsh build: vectors={} dim={} tables={} bits={} dims_per_bit={} expert_dims={} min_candidates={} max_candidates={} gamma={:.2} buckets={} elapsed_ms={:.3}",
            count,
            dim,
            num_tables,
            bits_per_table,
            dims_per_bit,
            bucket_expert_dims,
            min_candidates,
            max_candidates,
            adaptive_gamma,
            total_buckets,
            t0.elapsed().as_secs_f64() * 1000.0
        );

        Self {
            tables,
            prior: NigPrior::default(),
            dim,
            bits_per_table,
            num_tables,
            dims_per_bit,
            probe_hamming_radius,
            bucket_expert_dims,
            min_candidates,
            max_candidates,
            adaptive_gamma,
            total_rows: count,
            global_stats: global_dim_stats,
        }
    }

    /// Returns candidate row indices for a query vector.
    ///
    /// `delta` controls multi-probe aggressiveness:
    ///   delta < 0.5  -> probe exact bucket + Hamming-1 neighbours (higher recall)
    ///   delta >= 0.5 -> probe exact bucket only (faster, lower recall)
    pub(crate) fn route_candidate_rows_into(
        &self,
        query: &[f32],
        delta: f32,
        out: &mut Vec<u32>,
        scratch: &mut RouteScratch,
    ) {
        let t0 = Instant::now();
        let hamming_radius = if delta < 0.2 {
            self.probe_hamming_radius
        } else if delta < 0.5 {
            self.probe_hamming_radius.min(1)
        } else {
            0
        };

        // Stage 1: gather candidates with table-vote and bucket-Bayesian score.
        out.clear();
        scratch.row_accum.clear();
        scratch.sigs.clear();
        scratch.query_proj.resize(self.bucket_expert_dims, 0.0);
        scratch.scored.clear();

        for table in &self.tables {
            scratch.sigs.clear();
            table.probe_signatures(query, hamming_radius, &mut scratch.sigs);
            table.project_expert(query, &mut scratch.query_proj);

            for sig in scratch.sigs.iter().copied() {
                let Some(ids) = table.bucket_ids(sig) else {
                    continue;
                };
                let Some(expert) = table.bucket_expert(sig) else {
                    continue;
                };

                // Bucket-level Bayesian score in table-local expert space.
                let ll = expert
                    .stats
                    .iter()
                    .zip(scratch.query_proj.iter())
                    .map(|(s, &x)| s.predictive_log_likelihood(x, self.prior))
                    .sum::<f64>();
                let p = (expert.count as f64 / self.total_rows.max(1) as f64).clamp(1e-12, 1.0);
                let entropy_penalty = -p * p.ln();
                let bucket_score =
                    (ll + (expert.count as f64 + 1.0).ln() - 0.25 * entropy_penalty) as f32;

                for row in ids {
                    let entry = scratch.row_accum.entry(*row).or_default();
                    entry.score += bucket_score;
                    entry.votes = entry.votes.saturating_add(1);
                }
            }
        }

        if scratch.row_accum.is_empty() {
            debug!(
                "bayesian lsh route: tables={} radius={} candidates=0 elapsed_ms={:.3}",
                self.num_tables,
                hamming_radius,
                t0.elapsed().as_secs_f64() * 1000.0
            );
            return;
        }

        // Stage 2: smooth Bayesian confidence -> adaptive fanout.
        scratch.scored.extend(
            scratch
                .row_accum
                .drain()
                .map(|(row, acc)| (row, acc.score, acc.votes)),
        );
        scratch
            .scored
            .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
        let (adaptive_budget, confidence, top1_prob, norm_gap, vote_conf) =
            self.adaptive_candidate_budget(&scratch.scored);
        let capped = scratch.scored.len().min(adaptive_budget);
        out.extend(scratch.scored.iter().take(capped).map(|(row, _, _)| *row));

        debug!(
            "bayesian lsh route: tables={} radius={} confidence={:.3} vote_conf={:.3} top1_prob={:.3} norm_gap={:.3} adaptive_budget={} candidates={} elapsed_ms={:.3}",
            self.num_tables,
            hamming_radius,
            confidence,
            vote_conf,
            top1_prob,
            norm_gap,
            adaptive_budget,
            out.len(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }

    fn adaptive_candidate_budget(&self, scored: &[(u32, f32, u16)]) -> (usize, f64, f64, f64, f64) {
        if scored.is_empty() {
            return (self.min_candidates, 0.0, 0.0, 0.0, 0.0);
        }

        // Evidence concentration over the top scores.
        let top_k = scored.len().min(16);
        let max_score = scored[0].1 as f64;
        let temperature = 1.0f64;
        let mut exp_sum = 0.0f64;
        let mut top_exp = 0.0f64;
        for (rank, (_, s, _)) in scored.iter().take(top_k).enumerate() {
            let e = (((*s as f64) - max_score) / temperature).exp();
            if rank == 0 {
                top_exp = e;
            }
            exp_sum += e;
        }
        let top1_prob = if exp_sum > 0.0 {
            top_exp / exp_sum
        } else {
            1.0
        };

        // Margin confidence from top-1 vs top-2 score separation.
        let top1 = scored[0].1 as f64;
        let top2 = scored
            .get(1)
            .map(|(_, s, _)| *s as f64)
            .unwrap_or(top1 - 1.0);
        let raw_gap = (top1 - top2).max(0.0);
        let norm_gap = (raw_gap / (top1.abs() + 1e-6)).clamp(0.0, 1.0);

        // Vote confidence from table support and top-vote separation.
        let top_vote = scored[0].2 as f64;
        let second_vote = scored.get(1).map(|(_, _, v)| *v as f64).unwrap_or(0.0);
        let vote_ratio = (top_vote / self.num_tables.max(1) as f64).clamp(0.0, 1.0);
        let vote_gap = ((top_vote - second_vote) / self.num_tables.max(1) as f64).clamp(0.0, 1.0);
        let vote_conf = (0.7 * vote_ratio + 0.3 * vote_gap).clamp(0.0, 1.0);

        // Bayesian confidence in [0,1]: combine concentration and margin.
        let confidence = (0.55 * vote_conf + 0.30 * top1_prob + 0.15 * norm_gap).clamp(0.0, 1.0);
        let width = (self.max_candidates - self.min_candidates) as f64;
        let scale = (1.0 - confidence).powf(self.adaptive_gamma as f64);
        let budget = (self.min_candidates as f64 + width * scale)
            .round()
            .clamp(self.min_candidates as f64, self.max_candidates as f64)
            as usize;

        (budget, confidence, top1_prob, norm_gap, vote_conf)
    }

    /// Predictive log-likelihood of the query under the global NIG model.
    /// Used by `select_segments_by_routing_mass` to route queries to segments.
    pub(crate) fn root_score(&self, query: &[f32]) -> f64 {
        query
            .iter()
            .zip(self.global_stats.iter())
            .map(|(&x, s)| s.predictive_log_likelihood(x, self.prior))
            .sum()
    }

    pub(crate) fn root_stats(&self) -> &[NigStats] {
        &self.global_stats
    }
}
