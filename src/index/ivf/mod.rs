use std::{
    collections::VecDeque,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use crate::{
    DendraError,
    io::{read_f32_le, read_u32_le, read_u64_le},
    math,
};

const IVF_MAGIC: &[u8; 4] = b"IVFI";
const IVF_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Candidate {
    pub list_index: usize,
    pub start: usize,
    pub end: usize,
}

pub struct IvfBuilder {
    dim: usize,
    seed: u64,
    nlist: Option<usize>,
    nprobe: usize,
    train_sample_size: usize,
    train_iters: usize,
}

impl IvfBuilder {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            dim,
            seed,
            nlist: None,
            nprobe: 4,
            train_sample_size: 8_192,
            train_iters: 4,
        }
    }

    pub fn with_nlist(mut self, nlist: usize) -> Self {
        self.nlist = Some(nlist.max(1));
        self
    }

    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe.max(1);
        self
    }

    pub fn with_train_sample_size(mut self, train_sample_size: usize) -> Self {
        self.train_sample_size = train_sample_size.max(1);
        self
    }

    pub fn with_train_iters(mut self, train_iters: usize) -> Self {
        self.train_iters = train_iters.max(1);
        self
    }

    pub fn build(&self, vectors: &[f32], _ids: &[u32]) -> Result<Ivf, DendraError> {
        let vector_count = if self.dim == 0 {
            0
        } else {
            vectors.len() / self.dim
        };

        let inferred_nlist = (vector_count / 8192).clamp(8, 64).max(1);
        let nlist = self
            .nlist
            .unwrap_or(inferred_nlist)
            .min(vector_count.max(1));

        let mut index = Ivf {
            dim: self.dim,
            nlist,
            nprobe: self.nprobe.min(nlist.max(1)),
            centroids: vec![0.0; nlist * self.dim],
            offsets: vec![0u32; nlist + 1],
            postings: vec![0u32; vector_count],
        };

        if vector_count == 0 || self.dim == 0 {
            return Ok(index);
        }

        index.train_centroids_kmeanspp(
            vectors,
            self.seed,
            self.train_sample_size,
            self.train_iters,
        )?;
        index.assign(vectors)?;
        index.shuffle_postings_in_lists(self.seed);
        Ok(index)
    }
}

pub struct Ivf {
    dim: usize,
    nlist: usize,
    nprobe: usize,
    centroids: Vec<f32>,
    offsets: Vec<u32>,
    postings: Vec<u32>,
}

impl Ivf {
    pub fn builder(dim: usize, seed: u64) -> IvfBuilder {
        IvfBuilder::new(dim, seed)
    }

    pub fn nlist(&self) -> usize {
        self.nlist
    }

    pub fn len(&self) -> usize {
        self.nlist
    }

    pub fn save(&self, path: &Path) -> Result<(), DendraError> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(IVF_MAGIC)?;
        w.write_all(&IVF_VERSION.to_le_bytes())?;
        w.write_all(&(self.dim as u32).to_le_bytes())?;
        w.write_all(&(self.nlist as u32).to_le_bytes())?;
        w.write_all(&(self.nprobe as u32).to_le_bytes())?;
        w.write_all(&(self.postings.len() as u64).to_le_bytes())?;

        for &v in &self.centroids {
            w.write_all(&v.to_le_bytes())?;
        }
        for &off in &self.offsets {
            w.write_all(&off.to_le_bytes())?;
        }
        for &idx in &self.postings {
            w.write_all(&idx.to_le_bytes())?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, DendraError> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != IVF_MAGIC {
            return Err(DendraError::InvalidHeader {
                expected: String::from_utf8_lossy(IVF_MAGIC).to_string(),
                received: String::from_utf8_lossy(&magic).to_string(),
            });
        }

        let version = read_u32_le(&mut r)?;
        if version != IVF_VERSION {
            return Err(DendraError::UnsupportedVersion {
                expected: IVF_VERSION.to_string(),
                received: version.to_string(),
            });
        }

        let dim = read_u32_le(&mut r)? as usize;
        let nlist = read_u32_le(&mut r)? as usize;
        let nprobe = read_u32_le(&mut r)? as usize;
        let posting_len = read_u64_le(&mut r)? as usize;

        let mut centroids = vec![0.0f32; nlist * dim];
        for c in &mut centroids {
            *c = read_f32_le(&mut r)?;
        }

        let mut offsets = vec![0u32; nlist + 1];
        for off in &mut offsets {
            *off = read_u32_le(&mut r)?;
        }

        let mut postings = vec![0u32; posting_len];
        for idx in &mut postings {
            *idx = read_u32_le(&mut r)?;
        }

        Ok(Self {
            dim,
            nlist,
            nprobe: nprobe.max(1).min(nlist.max(1)),
            centroids,
            offsets,
            postings,
        })
    }

    pub fn search(
        &self,
        vector: &[f32],
        max_candidates: usize,
        candidates: &mut Vec<Candidate>,
        _queue: &mut VecDeque<usize>,
    ) -> usize {
        candidates.clear();
        if self.nlist == 0 || max_candidates == 0 {
            return 0;
        }

        let probes = self.nprobe.min(self.nlist).max(1);
        let per_list_budget = max_candidates.div_ceil(probes).max(1);
        let mut ranked_lists: Vec<(usize, f32)> = Vec::with_capacity(self.nlist);
        for list_idx in 0..self.nlist {
            let off = list_idx * self.dim;
            let centroid = &self.centroids[off..off + self.dim];
            ranked_lists.push((list_idx, math::l2_distance_sq(vector, centroid)));
        }
        ranked_lists.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut produced = 0usize;
        for (list_idx, _) in ranked_lists.into_iter().take(probes) {
            let start = self.offsets[list_idx] as usize;
            let end = self.offsets[list_idx + 1] as usize;
            if start >= end {
                continue;
            }

            let remaining = max_candidates.saturating_sub(produced);
            if remaining == 0 {
                break;
            }

            let take = per_list_budget.min(remaining);
            let take_end = (start + take).min(end);
            if take_end > start {
                candidates.push(Candidate {
                    list_index: list_idx,
                    start,
                    end: take_end,
                });
                produced += take_end - start;
            }

            if produced >= max_candidates {
                break;
            }
        }

        produced
    }

    pub fn candidate_lookups<'a>(
        &'a self,
        candidate: &Candidate,
    ) -> Result<&'a [u32], DendraError> {
        if candidate.end > self.postings.len() || candidate.start > candidate.end {
            return Err(DendraError::IndexOutOfBounds {
                index: candidate.end,
                length: self.postings.len(),
            });
        }
        Ok(&self.postings[candidate.start..candidate.end])
    }

    fn train_centroids_kmeanspp(
        &mut self,
        vectors: &[f32],
        seed: u64,
        train_sample_size: usize,
        train_iters: usize,
    ) -> Result<(), DendraError> {
        let count = vectors.len() / self.dim;
        if count == 0 || self.nlist == 0 {
            return Ok(());
        }

        let sample_count = count.min(train_sample_size.max(self.nlist));
        let sample = self.build_training_sample(count, sample_count, seed);
        self.init_kmeanspp(vectors, &sample, seed)?;
        self.lloyd_refine(vectors, &sample, seed, train_iters)?;
        Ok(())
    }

    fn assign(&mut self, vectors: &[f32]) -> Result<(), DendraError> {
        let count = vectors.len() / self.dim;
        let mut assignments = vec![0usize; count];
        let mut counts = vec![0usize; self.nlist];

        for (row, assigned) in assignments.iter_mut().enumerate().take(count) {
            let vec_off = row * self.dim;
            let vector = &vectors[vec_off..vec_off + self.dim];
            let list_idx = self.nearest_list(vector);
            *assigned = list_idx;
            counts[list_idx] += 1;
        }

        self.offsets[0] = 0;
        for (i, cnt) in counts.iter().enumerate().take(self.nlist) {
            self.offsets[i + 1] = self.offsets[i] + *cnt as u32;
        }

        let mut cursor = self.offsets.clone();
        for (row, list_idx) in assignments.into_iter().enumerate() {
            let write_pos = cursor[list_idx] as usize;
            self.postings[write_pos] = row as u32;
            cursor[list_idx] += 1;
        }

        Ok(())
    }

    fn build_training_sample(&self, count: usize, sample_count: usize, seed: u64) -> Vec<usize> {
        let start = (seed as usize) % count;
        let mut sample = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let idx = (start + (i * count) / sample_count) % count;
            sample.push(idx);
        }
        sample
    }

    fn init_kmeanspp(
        &mut self,
        vectors: &[f32],
        sample: &[usize],
        seed: u64,
    ) -> Result<(), DendraError> {
        let mut rng = seed.wrapping_add(0x9E3779B97F4A7C15);
        let first_idx = sample[rand_usize(&mut rng, sample.len())];
        let first_off = first_idx * self.dim;
        self.centroids[0..self.dim].copy_from_slice(&vectors[first_off..first_off + self.dim]);

        let mut nearest_dist2 = vec![f32::INFINITY; sample.len()];
        for c in 1..self.nlist {
            let prev_off = (c - 1) * self.dim;
            let prev = &self.centroids[prev_off..prev_off + self.dim];
            for (i, &row) in sample.iter().enumerate() {
                let off = row * self.dim;
                let v = &vectors[off..off + self.dim];
                let d2 = math::l2_distance_sq(v, prev);
                if d2 < nearest_dist2[i] {
                    nearest_dist2[i] = d2;
                }
            }

            let total: f32 = nearest_dist2.iter().copied().sum();
            let chosen_i = if total.is_finite() && total > 0.0 {
                let target = rand_f32(&mut rng) * total;
                let mut acc = 0.0f32;
                let mut pick = sample.len() - 1;
                for (i, &d2) in nearest_dist2.iter().enumerate() {
                    acc += d2;
                    if acc >= target {
                        pick = i;
                        break;
                    }
                }
                pick
            } else {
                rand_usize(&mut rng, sample.len())
            };

            let chosen_row = sample[chosen_i];
            let src = chosen_row * self.dim;
            let dst = c * self.dim;
            self.centroids[dst..dst + self.dim].copy_from_slice(&vectors[src..src + self.dim]);
        }
        Ok(())
    }

    fn lloyd_refine(
        &mut self,
        vectors: &[f32],
        sample: &[usize],
        seed: u64,
        iters: usize,
    ) -> Result<(), DendraError> {
        let mut rng = seed.wrapping_add(0xD1B54A32D192ED03);

        for _ in 0..iters {
            let mut sums = vec![0.0f32; self.nlist * self.dim];
            let mut counts = vec![0usize; self.nlist];

            for &row in sample.iter() {
                let off = row * self.dim;
                let v = &vectors[off..off + self.dim];
                let k = self.nearest_list(v);
                counts[k] += 1;
                let base = k * self.dim;
                for d in 0..self.dim {
                    sums[base + d] += v[d];
                }
            }

            for k in 0..self.nlist {
                let dst = k * self.dim;
                if counts[k] == 0 {
                    let row = sample[rand_usize(&mut rng, sample.len())];
                    let src = row * self.dim;
                    self.centroids[dst..dst + self.dim]
                        .copy_from_slice(&vectors[src..src + self.dim]);
                    continue;
                }
                let inv = 1.0f32 / counts[k] as f32;
                for d in 0..self.dim {
                    self.centroids[dst + d] = sums[dst + d] * inv;
                }
            }
        }
        Ok(())
    }

    fn shuffle_postings_in_lists(&mut self, seed: u64) {
        for list_idx in 0..self.nlist {
            let start = self.offsets[list_idx] as usize;
            let end = self.offsets[list_idx + 1] as usize;
            if end <= start + 1 {
                continue;
            }
            let mut state =
                seed ^ ((list_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)) ^ 0xA24BAED4963EE407;
            for i in (start + 1..end).rev() {
                let j = start + rand_usize(&mut state, i - start + 1);
                self.postings.swap(i, j);
            }
        }
    }

    fn nearest_list(&self, vector: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_dist = f32::INFINITY;
        for list_idx in 0..self.nlist {
            let off = list_idx * self.dim;
            let centroid = &self.centroids[off..off + self.dim];
            let d = math::l2_distance_sq(vector, centroid);
            if d < best_dist {
                best_dist = d;
                best = list_idx;
            }
        }
        best
    }
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn rand_usize(state: &mut u64, upper: usize) -> usize {
    if upper <= 1 {
        return 0;
    }
    (splitmix64_next(state) as usize) % upper
}

fn rand_f32(state: &mut u64) -> f32 {
    let x = splitmix64_next(state) >> 40;
    (x as f32) / ((1u32 << 24) as f32)
}
