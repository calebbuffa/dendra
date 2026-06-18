use crate::err::EngramError;
use crate::index::{BayesianLsh, NigStats};
use crate::storage::Sq8Store;
use log::debug;
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const SEGMENT_FILE: &str = "segment.bin";
const SEGMENT_MAGIC: &[u8; 4] = b"SGM2";
const SEGMENT_VERSION: u8 = 5;

pub(crate) struct ActiveState;
pub(crate) struct SealedState;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentTelemetry {
    pub vector_count: usize,
    pub created_at_unix_s: u64,
    pub query_visits: u64,
    pub centroid: Vec<f32>,
    pub radius_sq: f32,
    pub mean_norm: f32,
}

impl SegmentTelemetry {
    fn new(vectors: &[f32], dim: usize) -> Self {
        let created_at_unix_s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let count = vectors.len().checked_div(dim).unwrap_or(0);
        let mut centroid = vec![0.0f32; dim];

        if count == 0 || dim == 0 {
            return Self {
                vector_count: count,
                created_at_unix_s,
                query_visits: 0,
                centroid,
                radius_sq: 0.0,
                mean_norm: 0.0,
            };
        }

        let mut norm_sum = 0.0f32;
        for row in vectors.chunks_exact(dim) {
            let mut norm_sq = 0.0f32;
            for (j, &value) in row.iter().enumerate() {
                centroid[j] += value;
                norm_sq = value.mul_add(value, norm_sq);
            }
            norm_sum += norm_sq.sqrt();
        }

        let inv_count = 1.0f32 / count as f32;
        for value in &mut centroid {
            *value *= inv_count;
        }

        let mut radius_sq = 0.0f32;
        for row in vectors.chunks_exact(dim) {
            let mut dist_sq = 0.0f32;
            for j in 0..dim {
                let diff = row[j] - centroid[j];
                dist_sq = diff.mul_add(diff, dist_sq);
            }
            radius_sq = radius_sq.max(dist_sq);
        }

        Self {
            vector_count: count,
            created_at_unix_s,
            query_visits: 0,
            centroid,
            radius_sq,
            mean_norm: norm_sum * inv_count,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct SegmentCore {
    id: u64,
    dim: usize,
    depth: u8,
    vectors: Vec<f32>,
    ids: Vec<u32>,
    store: Option<Sq8Store>,
    lsh: Option<BayesianLsh>,
    telemetry: Option<SegmentTelemetry>,
}

pub(crate) struct Segment<S> {
    core: SegmentCore,
    _state: PhantomData<S>,
}

impl<S> Clone for Segment<S> {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            _state: PhantomData,
        }
    }
}

impl Segment<ActiveState> {
    pub(crate) fn new_active(id: u64, dim: usize) -> Self {
        Self {
            core: SegmentCore {
                id,
                dim,
                depth: 0,
                vectors: Vec::new(),
                ids: Vec::new(),
                store: None,
                lsh: None,
                telemetry: None,
            },
            _state: PhantomData,
        }
    }

    pub(crate) fn insert(&mut self, vector: &[f32], id: u32) -> Result<(), EngramError> {
        if vector.len() != self.core.dim {
            return Err(EngramError::InvalidVectorDimension {
                expected: self.core.dim,
                received: vector.len(),
            });
        }
        self.core.vectors.extend_from_slice(vector);
        self.core.ids.push(id);
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.core.ids.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.core.ids.is_empty()
    }

    #[allow(dead_code)]
    pub(crate) fn clear(&mut self) {
        self.core.vectors.clear();
        self.core.ids.clear();
    }

    pub(crate) fn id(&self) -> u64 {
        self.core.id
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn seal(
        self,
        bits_per_table: usize,
        num_tables: usize,
        dims_per_bit: usize,
        probe_hamming_radius: u8,
        bucket_expert_dims: usize,
        min_candidates: usize,
        max_candidates: usize,
        adaptive_gamma: f32,
        seed: u64,
    ) -> Result<Segment<SealedState>, EngramError> {
        let store = Sq8Store::from_vectors(&self.core.vectors, &self.core.ids, self.core.dim)?;
        let lsh = BayesianLsh::build(
            &self.core.vectors,
            self.core.dim,
            bits_per_table,
            num_tables,
            dims_per_bit,
            probe_hamming_radius,
            bucket_expert_dims,
            min_candidates,
            max_candidates,
            adaptive_gamma,
            seed,
        );
        let telemetry = SegmentTelemetry::new(&self.core.vectors, self.core.dim);

        Ok(Segment {
            core: SegmentCore {
                id: self.core.id,
                dim: self.core.dim,
                depth: self.core.depth,
                vectors: Vec::new(),
                ids: Vec::new(),
                store: Some(store),
                lsh: Some(lsh),
                telemetry: Some(telemetry),
            },
            _state: PhantomData,
        })
    }
}

impl Segment<SealedState> {
    pub(crate) fn id(&self) -> u64 {
        self.core.id
    }

    pub(crate) fn depth(&self) -> u8 {
        self.core.depth
    }

    pub(crate) fn dim(&self) -> usize {
        self.core.dim
    }

    pub(crate) fn len(&self) -> usize {
        self.core.store.as_ref().map_or(0, |s| s.len())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn store(&self) -> Result<&Sq8Store, EngramError> {
        self.core.store.as_ref().ok_or_else(|| {
            EngramError::InvariantViolation("sealed segment missing SQ8 store".to_string())
        })
    }

    pub(crate) fn lsh(&self) -> Result<&BayesianLsh, EngramError> {
        self.core.lsh.as_ref().ok_or_else(|| {
            EngramError::InvariantViolation("sealed segment missing Bayesian LSH index".to_string())
        })
    }

    pub(crate) fn root_stats(&self) -> Result<&[NigStats], EngramError> {
        Ok(self.lsh()?.root_stats())
    }

    pub(crate) fn telemetry(&self) -> Result<&SegmentTelemetry, EngramError> {
        self.core.telemetry.as_ref().ok_or_else(|| {
            EngramError::InvariantViolation("sealed segment missing telemetry".to_string())
        })
    }

    fn telemetry_mut(&mut self) -> Result<&mut SegmentTelemetry, EngramError> {
        self.core.telemetry.as_mut().ok_or_else(|| {
            EngramError::InvariantViolation("sealed segment missing telemetry".to_string())
        })
    }

    pub(crate) fn telemetry_snapshot(&self) -> Result<SegmentTelemetry, EngramError> {
        Ok(self.telemetry()?.clone())
    }

    pub(crate) fn set_query_visits(&mut self, query_visits: u64) -> Result<(), EngramError> {
        self.telemetry_mut()?.query_visits = query_visits;
        Ok(())
    }

    pub(crate) fn external_id(&self, row: usize) -> Result<Option<u32>, EngramError> {
        Ok(self.store()?.id_at(row))
    }

    pub(crate) fn decode_row(&self, row: usize, out: &mut [f32]) -> Result<(), EngramError> {
        self.store()?.decode_row(row, out);
        Ok(())
    }

    pub(crate) fn with_depth(mut self, depth: u8) -> Self {
        self.core.depth = depth;
        self
    }

    pub(crate) fn from_parts(core: SegmentCore) -> Self {
        Self {
            core,
            _state: PhantomData,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn into_core(self) -> SegmentCore {
        self.core
    }

    pub(crate) fn persist(&self, dir: &Path) -> Result<(), EngramError> {
        std::fs::create_dir_all(dir)?;
        if let Some(store) = &self.core.store {
            store.write_sidecars(dir)?;
        }
        let out_path = dir.join(SEGMENT_FILE);
        let file = std::fs::File::create(&out_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(SEGMENT_MAGIC)?;
        writer.write_all(&[SEGMENT_VERSION])?;
        bincode::serde::encode_into_std_write(&self.core, &mut writer, bincode::config::standard())
            .map_err(|e| EngramError::Codec(e.to_string()))?;
        writer.flush()?;

        let file_size = std::fs::metadata(&out_path)?.len() as usize;
        let payload_len = file_size.saturating_sub(SEGMENT_MAGIC.len() + 1);
        let ids_len = std::fs::metadata(dir.join("ids.bin"))?.len() as usize;
        let codes_len = std::fs::metadata(dir.join("codes.bin"))?.len() as usize;
        debug!(
            "segment persisted: id={} path={} payload_bytes={} ids_bytes={} codes_bytes={} total_bytes={}",
            self.id(),
            out_path.display(),
            payload_len,
            ids_len,
            codes_len,
            payload_len + ids_len + codes_len
        );
        Ok(())
    }

    pub(crate) fn load(dir: &Path) -> Result<Self, EngramError> {
        let path = dir.join(SEGMENT_FILE);
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(EngramError::Io)?;

        if mmap.len() < SEGMENT_MAGIC.len() + 1 {
            return Err(EngramError::InvalidHeader {
                expected: "segment header".to_string(),
                received: "too short".to_string(),
            });
        }

        let magic = &mmap[..SEGMENT_MAGIC.len()];
        if magic != SEGMENT_MAGIC {
            return Err(EngramError::InvalidHeader {
                expected: String::from_utf8_lossy(SEGMENT_MAGIC).to_string(),
                received: String::from_utf8_lossy(magic).to_string(),
            });
        }

        let version = mmap[SEGMENT_MAGIC.len()];
        if version != SEGMENT_VERSION {
            return Err(EngramError::UnsupportedVersion {
                expected: SEGMENT_VERSION.to_string(),
                received: version.to_string(),
            });
        }

        let payload = &mmap[SEGMENT_MAGIC.len() + 1..];
        let mut core: SegmentCore =
            bincode::serde::decode_from_slice(payload, bincode::config::standard())
                .map(|(core, _)| core)
                .map_err(|e| EngramError::Codec(e.to_string()))?;
        if let Some(store) = core.store.as_mut() {
            store.try_enable_mmap(dir)?;
        }
        Ok(Self::from_parts(core))
    }
}
