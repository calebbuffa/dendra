use crate::{
    core::memory::size_of_vec,
    distance::MetricFn,
    err::DendraError,
    index::{IndexCandidate, RpfIndex, SegmentIndex},
    math,
    query::Query,
};
use log::debug;
use memmap2::Mmap;
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SEGMENT_MAGIC: [u8; 4] = *b"SEGM";
pub const SEGMENT_FORMAT_VERSION: u8 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegmentHeader {
    pub magic: [u8; 4],
    pub format_version: u8,
    pub flags: u8,
    pub reserved0: u16,
    pub dim: u32,
    pub count: u64,
    pub vectors_bytes: u64,
    pub ids_bytes: u64,
    pub index_bytes: u64,
}

impl SegmentHeader {
    pub fn new(
        dim: usize,
        count: usize,
        vectors_bytes: u64,
        ids_bytes: u64,
        index_bytes: u64,
    ) -> Self {
        Self {
            magic: SEGMENT_MAGIC,
            format_version: SEGMENT_FORMAT_VERSION,
            flags: 0,
            reserved0: 0,
            dim: dim as u32,
            count: count as u64,
            vectors_bytes,
            ids_bytes,
            index_bytes,
        }
    }

    fn write_to<W: Write>(&self, w: &mut W) -> Result<(), DendraError> {
        w.write_all(&self.magic)?;
        w.write_all(&[self.format_version])?;
        w.write_all(&[self.flags])?;
        w.write_all(&self.reserved0.to_le_bytes())?;
        w.write_all(&self.dim.to_le_bytes())?;
        w.write_all(&self.count.to_le_bytes())?;
        w.write_all(&self.vectors_bytes.to_le_bytes())?;
        w.write_all(&self.ids_bytes.to_le_bytes())?;
        w.write_all(&self.index_bytes.to_le_bytes())?;
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self, DendraError> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;

        let mut b1 = [0u8; 1];
        r.read_exact(&mut b1)?;
        let format_version = b1[0];

        r.read_exact(&mut b1)?;
        let flags = b1[0];

        let mut b2 = [0u8; 2];
        r.read_exact(&mut b2)?;
        let reserved0 = u16::from_le_bytes(b2);

        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)?;
        let dim = u32::from_le_bytes(b4);

        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8)?;
        let count = u64::from_le_bytes(b8);

        r.read_exact(&mut b8)?;
        let vectors_bytes = u64::from_le_bytes(b8);

        r.read_exact(&mut b8)?;
        let ids_bytes = u64::from_le_bytes(b8);

        r.read_exact(&mut b8)?;
        let index_bytes = u64::from_le_bytes(b8);

        Ok(Self {
            magic,
            format_version,
            flags,
            reserved0,
            dim,
            count,
            vectors_bytes,
            ids_bytes,
            index_bytes,
        })
    }
}

/// Per-segment statistics for compaction decisions.
#[derive(Clone)]
pub(crate) struct SegmentSummary {
    pub segment_id: u64,
    pub vector_count: usize,
    pub dimension: usize,
    pub created_at: u64, // unix timestamp in seconds
    pub query_visit_count: Arc<AtomicU64>,
    pub centroid: Vec<f32>, // mean of all vectors for overlap detection
    pub r2: f32,            // radius squared of the segment (max distance from centroid)
    pub entropy: f32,       // heterogeneity score in [0, 1]
}

impl SegmentSummary {
    pub fn new(
        segment_id: u64,
        vector_count: usize,
        dimension: usize,
        centroid: Vec<f32>,
        r2: f32,
        entropy: f32,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            segment_id,
            vector_count,
            dimension,
            created_at: now,
            query_visit_count: Arc::new(AtomicU64::new(0)),
            centroid,
            r2,
            entropy,
        }
    }

    pub fn age_secs(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now.saturating_sub(self.created_at)
    }

    pub fn size_bytes(&self) -> usize {
        self.vector_count * self.dimension * std::mem::size_of::<f32>()
    }

    pub fn size_mb(&self) -> f32 {
        self.size_bytes() as f32 / (1024.0 * 1024.0)
    }

    pub fn visit_count(&self) -> u64 {
        self.query_visit_count.load(Ordering::Relaxed)
    }

    pub fn record_visit(&self) {
        self.query_visit_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// A `SegmentBitSet` is a compact `u64`-based bitset for tracking seen positions.
/// Avoids `HashSet<usize>` overhead for deduplication in the query hot path.
pub(crate) struct SegmentBitSet {
    inline: u64,
    vec: Vec<u64>,
}

impl SegmentBitSet {
    #[inline]
    fn with_capacity(max_id: usize) -> Self {
        let vec = if max_id > 64 {
            vec![0u64; (max_id + 63) / 64]
        } else {
            Vec::new()
        };
        Self { inline: 0, vec }
    }

    #[inline]
    fn insert(&mut self, id: usize) -> bool {
        if id < 64 {
            let mask = 1u64 << (id as u64);
            if self.inline & mask != 0 {
                return false;
            }
            self.inline |= mask;
            true
        } else {
            let idx = id / 64;
            let bit = id % 64;
            if let Some(word) = self.vec.get_mut(idx) {
                let mask = 1u64 << (bit as u64);
                if *word & mask != 0 {
                    return false;
                }
                *word |= mask;
                true
            } else {
                false
            }
        }
    }

    pub fn clear(&mut self) {
        self.inline = 0;
        self.vec.clear();
    }
}

enum SegmentStorage {
    Owned {
        vectors: Vec<f32>,
        external_ids: Vec<u32>,
    },
    Mapped {
        _vectors_file: File,
        _ids_file: File,
        vectors_mmap: Mmap,
        ids_mmap: Mmap,
    },
}

pub struct SegmentBuilder {
    vectors: Vec<f32>,
    external_ids: Vec<u32>,
    pub dim: usize,
    pub max_capacity: usize,
    pub current_capacity: usize,
}

impl SegmentBuilder {
    pub fn new(dim: usize, max_capacity: usize) -> Self {
        Self {
            vectors: Vec::new(),
            external_ids: Vec::new(),
            dim,
            max_capacity: max_capacity.saturating_mul(1024 * 1024),
            current_capacity: 0,
        }
    }

    pub fn try_insert(&mut self, vector: &[f32], external_id: u32) -> Result<bool, DendraError> {
        if vector.len() != self.dim {
            return Err(DendraError::InvalidVectorDimension {
                expected: self.dim,
                received: vector.len(),
            });
        }
        let size = size_of_vec(vector);
        if self.will_exceed_capacity(size.bytes) {
            return Ok(false);
        }
        self.vectors.extend_from_slice(vector);
        self.external_ids.push(external_id);
        self.current_capacity += size.bytes;
        Ok(true)
    }

    pub fn is_full(&self) -> bool {
        self.current_capacity >= self.max_capacity
    }

    pub fn will_exceed_capacity(&self, additional: usize) -> bool {
        self.current_capacity + additional > self.max_capacity
    }

    pub fn clear(&mut self) {
        self.vectors.clear();
        self.external_ids.clear();
        self.current_capacity = 0;
    }

    pub fn build(
        &mut self,
        leaf_size: usize,
        num_trees: usize,
        seed: u64,
    ) -> Result<Segment, DendraError> {
        let vectors = std::mem::take(&mut self.vectors);
        let external_ids = std::mem::take(&mut self.external_ids);
        self.current_capacity = 0;

        let index = RpfIndex::builder(self.dim, leaf_size, num_trees, seed)
            .build(&vectors, &external_ids)?;
        Ok(Segment::from_owned(
            self.dim,
            vectors,
            external_ids,
            Box::new(index),
        ))
    }
}

pub(crate) struct SegmentQueryContext {
    pub(crate) candidates: Vec<IndexCandidate>,
    pub(crate) queue: VecDeque<usize>,
    pub(crate) best_map: HashMap<u32, f32>,
    pub(crate) seen: SegmentBitSet,
}

impl SegmentQueryContext {
    pub fn new(candidate_capacity: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(candidate_capacity),
            queue: VecDeque::new(),
            best_map: HashMap::new(),
            seen: SegmentBitSet::with_capacity(0),
        }
    }
}

pub struct Segment {
    pub dim: usize,
    pub count: usize,
    pub index: Box<dyn SegmentIndex>,
    vector_bytes_per_row: usize,
    storage: SegmentStorage,
}

impl Segment {
    pub fn from_owned(
        dim: usize,
        vectors: Vec<f32>,
        external_ids: Vec<u32>,
        index: Box<dyn SegmentIndex>,
    ) -> Self {
        let count = external_ids.len();
        let vector_bytes_per_row = dim * std::mem::size_of::<f32>();

        Self {
            dim,
            count,
            index,
            vector_bytes_per_row,
            storage: SegmentStorage::Owned {
                vectors,
                external_ids,
            },
        }
    }

    pub fn estimate_entropy(&self) -> Result<f32, DendraError> {
        if self.count < 2 {
            return Ok(0.0);
        }

        // Fast heterogeneity proxy for embedding segments:
        // sample vectors uniformly and measure average pairwise cosine distance.
        // 0 => very coherent cluster, 1 => highly mixed content.
        let sample_cap = 64usize;
        let sample_size = self.count.min(sample_cap);
        if sample_size < 2 {
            return Ok(0.0);
        }

        let mut samples: Vec<Vec<f32>> = Vec::with_capacity(sample_size);
        for k in 0..sample_size {
            let i = (k * self.count) / sample_size;
            let v = self.vector_at(i)?;
            samples.push(v.to_vec());
        }

        let norms: Vec<f32> = samples.iter().map(|v| math::l2_norm(v)).collect();

        let mut sum = 0.0f32;
        let mut n = 0usize;
        for i in 0..samples.len() {
            if norms[i] <= 0.0 {
                continue;
            }
            for j in (i + 1)..samples.len() {
                if norms[j] <= 0.0 {
                    continue;
                }
                let cosine =
                    (math::dot(&samples[i], &samples[j]) / (norms[i] * norms[j])).clamp(-1.0, 1.0);
                let dist01 = 0.5 * (1.0 - cosine);
                sum += dist01;
                n += 1;
            }
        }

        if n == 0 {
            return Ok(0.5);
        }

        Ok((sum / n as f32).clamp(0.0, 1.0))
    }

    /// r2 is the largest squared distance from the centroid to any vector in the segment.
    pub fn describe(&self, centroid: &mut [f32], r2: &mut f32) -> Result<(), DendraError> {
        if centroid.len() != self.dim {
            return Err(DendraError::InvalidVectorDimension {
                expected: self.dim,
                received: centroid.len(),
            });
        }

        for c in centroid.iter_mut() {
            *c = 0.0;
        }
        *r2 = 0.0;

        if self.count == 0 {
            return Ok(());
        }

        // Pass 1: centroid accumulation.
        for i in 0..self.count {
            let vec = self.vector_at(i)?;
            for (j, &val) in vec.iter().enumerate() {
                centroid[j] += val;
            }
        }

        // Normalize centroid.
        let count = self.count as f32;
        for val in centroid.iter_mut() {
            *val /= count;
        }

        // Pass 2: compute max squared distance to centroid.
        for i in 0..self.count {
            let vec = self.vector_at(i)?;
            let dist2 = math::l2_distance_sq(vec, centroid);
            *r2 = (*r2).max(dist2);
        }

        Ok(())
    }

    #[inline]
    pub fn vector_at(&self, row: usize) -> Result<&[f32], DendraError> {
        let raw_row_bytes = self.dim * std::mem::size_of::<f32>();
        if self.vector_bytes_per_row != raw_row_bytes {
            return Err(DendraError::UnsupportedOperation(
                "vector_at is unavailable for quantized segment storage".to_string(),
            ));
        }

        if row >= self.count {
            return Err(DendraError::IndexOutOfBounds {
                index: row,
                length: self.count,
            });
        }

        let start = row * self.dim;
        let end = start + self.dim;

        match &self.storage {
            SegmentStorage::Owned { vectors, .. } => Ok(&vectors[start..end]),
            SegmentStorage::Mapped { vectors_mmap, .. } => {
                let start_bytes = start * std::mem::size_of::<f32>();
                let end_bytes = end * std::mem::size_of::<f32>();
                let bytes = &vectors_mmap[start_bytes..end_bytes];

                if (bytes.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
                    return Err(DendraError::MmapFailed("unaligned f32 payload".to_string()));
                }

                let ptr = bytes.as_ptr() as *const f32;
                Ok(unsafe { std::slice::from_raw_parts(ptr, self.dim) })
            }
        }
    }

    #[inline]
    pub fn id_at(&self, row: usize) -> Result<u32, DendraError> {
        if row >= self.count {
            return Err(DendraError::IndexOutOfBounds {
                index: row,
                length: self.count,
            });
        }

        match &self.storage {
            SegmentStorage::Owned { external_ids, .. } => Ok(external_ids[row]),
            SegmentStorage::Mapped { ids_mmap, .. } => {
                let start = row * std::mem::size_of::<u32>();
                let end = start + std::mem::size_of::<u32>();
                let mut b = [0u8; 4];
                b.copy_from_slice(&ids_mmap[start..end]);
                Ok(u32::from_le_bytes(b))
            }
        }
    }

    pub(crate) fn query(
        &self,
        query: &Query,
        max_candidates: usize,
        metric: MetricFn,
        context: &mut SegmentQueryContext,
    ) -> Result<(), DendraError> {
        self.index.search(
            &query.vector,
            max_candidates,
            &mut context.candidates,
            &mut context.queue,
        );

        context.seen.clear();
        let best_map = &mut context.best_map;
        let query_vec = &query.vector;
        let threshold = query.threshold;

        for candidate in context.candidates.iter() {
            let lookups = self.index.candidate_lookups(candidate)?;

            for &lookup_idx in lookups {
                let id_idx = lookup_idx as usize;

                if !context.seen.insert(id_idx) {
                    continue;
                }

                let candidate_id = self.id_at(id_idx)?;
                let vector = self.vector_at(id_idx)?;
                let distance = metric(query_vec, vector)?;

                if threshold.is_some_and(|t| distance > t) {
                    continue;
                }

                match best_map.get_mut(&candidate_id) {
                    Some(prev) => {
                        if distance < *prev {
                            *prev = distance;
                        }
                    }
                    None => {
                        best_map.insert(candidate_id, distance);
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn flush(self, path: &Path) -> Result<Segment, DendraError> {
        let start = std::time::Instant::now();
        debug!(
            "[Segment::flush] Starting, {} vectors, dim={}, path={:?}",
            self.count, self.dim, path
        );
        fs::create_dir_all(path)?;

        let meta_path = path.join("metadata.bin");
        let vectors_path = path.join("vectors.bin");
        let ids_path = path.join("ids.bin");
        let index_path = path.join("index.bin");
        let (vectors, external_ids, index) = match self.storage {
            SegmentStorage::Owned {
                vectors,
                external_ids,
            } => (vectors, external_ids, self.index),
            SegmentStorage::Mapped { .. } => {
                return Err(DendraError::UnsupportedOperation(
                    "cannot flush a mapped segment".to_string(),
                ));
            }
        };

        // For f32 vectors, bytes_per_row is dim * 4
        let bytes_per_row = self.dim * std::mem::size_of::<f32>();
        let total_encoded_bytes = external_ids.len() * bytes_per_row;

        let meta_start = std::time::Instant::now();
        {
            let mut w = BufWriter::new(File::create(&meta_path)?);
            let header = SegmentHeader::new(
                self.dim,
                external_ids.len(),
                total_encoded_bytes as u64,
                (external_ids.len() * std::mem::size_of::<u32>()) as u64,
                0,
            );
            header.write_to(&mut w)?;
            w.flush()?
        }
        debug!(
            "  [meta] {:.3}ms",
            meta_start.elapsed().as_secs_f64() * 1000.0
        );

        // Stream vectors in chunks to avoid allocating all at once
        let chunk_size = 16_384; // ~16k vectors per chunk
        let vec_start = std::time::Instant::now();
        let mut total_chunks = 0;
        {
            let mut w = BufWriter::new(File::create(&vectors_path)?);
            let num_vectors = external_ids.len();

            for chunk_start in (0..num_vectors).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(num_vectors);
                total_chunks += 1;
                let vec_chunk = &vectors[chunk_start * self.dim..chunk_end * self.dim];
                let bytes: &[u8] = bytemuck::cast_slice(vec_chunk);
                w.write_all(bytes)?;
            }
            w.flush()?;
        }
        let t2 = start.elapsed();
        debug!(
            "  [vectors] {:.3}s ({} chunks)",
            vec_start.elapsed().as_secs_f64(),
            total_chunks
        );

        let ids_start = std::time::Instant::now();
        {
            let mut w = BufWriter::new(File::create(&ids_path)?);
            let bytes: &[u8] = bytemuck::cast_slice(&external_ids);
            w.write_all(bytes)?;
            w.flush()?;
        }
        let _t3 = start.elapsed();
        debug!(
            "  [ids] {:.3}ms",
            ids_start.elapsed().as_secs_f64() * 1000.0
        );

        let index_start = std::time::Instant::now();
        index.save(&index_path)?;
        debug!("  [index] {:.3}s", index_start.elapsed().as_secs_f64());

        debug!(
            "[Segment::flush] TOTAL: {:.3}s | meta={:.1}ms, vec={:.1}ms, ids={:.1}ms, idx={:.1}ms",
            start.elapsed().as_secs_f64(),
            t2.as_secs_f64() * 1000.0,
            (vec_start.elapsed().as_secs_f64()
                - (t2.as_secs_f64() - vec_start.elapsed().as_secs_f64()))
                * 1000.0,
            ids_start.elapsed().as_secs_f64() * 1000.0,
            index_start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(Segment::open(path)?)
    }

    pub fn open(path: &Path) -> Result<Self, DendraError> {
        let meta_path = path.join("metadata.bin");
        let vectors_path = path.join("vectors.bin");
        let ids_path = path.join("ids.bin");
        let index_path = path.join("index.bin");

        let header = {
            let mut r = std::io::BufReader::new(File::open(&meta_path)?);
            SegmentHeader::read_from(&mut r)?
        };

        if header.magic != SEGMENT_MAGIC {
            return Err(DendraError::InvalidHeader {
                expected: String::from_utf8_lossy(&SEGMENT_MAGIC).to_string(),
                received: String::from_utf8_lossy(&header.magic).to_string(),
            });
        }

        if header.format_version != SEGMENT_FORMAT_VERSION {
            return Err(DendraError::UnsupportedVersion {
                expected: SEGMENT_FORMAT_VERSION.to_string(),
                received: header.format_version.to_string(),
            });
        }

        let vectors_file = File::open(&vectors_path)?;
        let ids_file = File::open(&ids_path)?;

        let vectors_len = vectors_file.metadata()?.len() as usize;
        let ids_len = ids_file.metadata()?.len() as usize;

        if vectors_len != header.vectors_bytes as usize {
            return Err(DendraError::MmapSizeMismatch {
                expected: header.vectors_bytes as usize,
                received: vectors_len,
            });
        }

        let count = header.count as usize;
        let vector_bytes_per_row = if count == 0 {
            header.dim as usize * std::mem::size_of::<f32>()
        } else {
            if vectors_len % count != 0 {
                return Err(DendraError::MmapSizeMismatch {
                    expected: count,
                    received: vectors_len,
                });
            }
            vectors_len / count
        };

        if ids_len != header.ids_bytes as usize {
            return Err(DendraError::MmapSizeMismatch {
                expected: header.ids_bytes as usize,
                received: ids_len,
            });
        }

        let vectors_mmap = unsafe { Mmap::map(&vectors_file) }
            .map_err(|e| DendraError::MmapFailed(e.to_string()))?;

        let ids_mmap =
            unsafe { Mmap::map(&ids_file) }.map_err(|e| DendraError::MmapFailed(e.to_string()))?;

        let index = RpfIndex::load(&index_path)?;

        Ok(Self {
            dim: header.dim as usize,
            count: header.count as usize,
            index: Box::new(index),
            vector_bytes_per_row,
            storage: SegmentStorage::Mapped {
                _vectors_file: vectors_file,
                _ids_file: ids_file,
                vectors_mmap,
                ids_mmap,
            },
        })
    }
}
