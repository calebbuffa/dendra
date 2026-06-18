use crate::err::EngramError;
use crate::math::adc_l2_sq as adc_l2_sq_math;
use memmap2::MmapOptions;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fs;
use std::path::Path;

const IDS_FILE: &str = "ids.bin";
const CODES_FILE: &str = "codes.bin";

#[derive(Debug)]
pub(crate) struct Sq8Store {
    dim: usize,
    count: usize,
    min_vals: Vec<f32>,
    max_vals: Vec<f32>,
    ids: Vec<u32>,
    codes: Vec<u8>,
    mapped_ids: Option<memmap2::Mmap>,
    mapped_codes: Option<memmap2::Mmap>,
}

#[derive(Serialize, Deserialize)]
struct Sq8StoreMeta {
    dim: usize,
    count: usize,
    min_vals: Vec<f32>,
    max_vals: Vec<f32>,
}

impl Serialize for Sq8Store {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Sq8StoreMeta {
            dim: self.dim,
            count: self.count,
            min_vals: self.min_vals.clone(),
            max_vals: self.max_vals.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Sq8Store {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let meta = Sq8StoreMeta::deserialize(deserializer)?;
        Ok(Self {
            dim: meta.dim,
            count: meta.count,
            min_vals: meta.min_vals,
            max_vals: meta.max_vals,
            ids: Vec::new(),
            codes: Vec::new(),
            mapped_ids: None,
            mapped_codes: None,
        })
    }
}

impl Clone for Sq8Store {
    fn clone(&self) -> Self {
        Self {
            dim: self.dim,
            count: self.count,
            min_vals: self.min_vals.clone(),
            max_vals: self.max_vals.clone(),
            ids: self.ids_vec(),
            codes: self.codes_slice().to_vec(),
            mapped_ids: None,
            mapped_codes: None,
        }
    }
}

impl Sq8Store {
    pub(crate) fn from_vectors(
        vectors: &[f32],
        ids: &[u32],
        dim: usize,
    ) -> Result<Self, EngramError> {
        if !vectors.len().is_multiple_of(dim) {
            return Err(EngramError::InvalidVectorDimension {
                expected: dim,
                received: vectors.len(),
            });
        }
        let count = vectors.len() / dim;
        if count != ids.len() {
            return Err(EngramError::InvalidVectorDimension {
                expected: count,
                received: ids.len(),
            });
        }

        let mut min_vals = vec![f32::INFINITY; dim];
        let mut max_vals = vec![f32::NEG_INFINITY; dim];
        for row in vectors.chunks_exact(dim) {
            for (j, value) in row.iter().enumerate() {
                if *value < min_vals[j] {
                    min_vals[j] = *value;
                }
                if *value > max_vals[j] {
                    max_vals[j] = *value;
                }
            }
        }

        let mut codes = vec![0u8; vectors.len()];
        for (i, row) in vectors.chunks_exact(dim).enumerate() {
            let out = &mut codes[i * dim..(i + 1) * dim];
            for j in 0..dim {
                let min_v = min_vals[j];
                let max_v = max_vals[j];
                let range = (max_v - min_v).max(1e-12);
                let scaled = ((row[j] - min_v) / range) * 255.0;
                let clamped = scaled.round().clamp(0.0, 255.0);
                out[j] = clamped as u8;
            }
        }

        Ok(Self {
            dim,
            count,
            ids: ids.to_vec(),
            min_vals,
            max_vals,
            codes,
            mapped_ids: None,
            mapped_codes: None,
        })
    }

    pub(crate) fn write_sidecars(&self, dir: &Path) -> Result<(), EngramError> {
        let ids_bytes = bytemuck::cast_slice(self.ids_slice());
        fs::write(dir.join(IDS_FILE), ids_bytes)?;
        fs::write(dir.join(CODES_FILE), self.codes_slice())?;
        Ok(())
    }

    pub(crate) fn try_enable_mmap(&mut self, dir: &Path) -> Result<(), EngramError> {
        let ids_path = dir.join(IDS_FILE);
        let codes_path = dir.join(CODES_FILE);
        if !ids_path.exists() || !codes_path.exists() {
            return Err(EngramError::InvalidHeader {
                expected: format!("{} and {} sidecar files", IDS_FILE, CODES_FILE),
                received: "missing sidecar file(s)".to_string(),
            });
        }

        let ids_file = fs::File::open(&ids_path)?;
        let codes_file = fs::File::open(&codes_path)?;
        let ids_mmap = unsafe { MmapOptions::new().map(&ids_file) }.map_err(EngramError::Io)?;
        let codes_mmap = unsafe { MmapOptions::new().map(&codes_file) }.map_err(EngramError::Io)?;

        let expected_ids_bytes = self.count * std::mem::size_of::<u32>();
        let expected_codes_bytes = self.count * self.dim;
        if ids_mmap.len() != expected_ids_bytes || codes_mmap.len() != expected_codes_bytes {
            return Err(EngramError::Codec("sq8 sidecar size mismatch".to_string()));
        }

        // Ensure mmap'd ids are byte-aligned for u32 reads.
        if bytemuck::try_cast_slice::<u8, u32>(&ids_mmap).is_err() {
            return Err(EngramError::Codec(
                "sq8 ids sidecar alignment mismatch".to_string(),
            ));
        }

        self.mapped_ids = Some(ids_mmap);
        self.mapped_codes = Some(codes_mmap);
        self.ids.clear();
        self.codes.clear();
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub(crate) fn decode_row(&self, row: usize, out: &mut [f32]) {
        let dim = self.dim;
        let start = row * dim;
        let end = start + dim;
        let code = &self.codes_slice()[start..end];

        for j in 0..dim {
            let min_v = self.min_vals[j];
            let max_v = self.max_vals[j];
            let alpha = (max_v - min_v) / 255.0;
            out[j] = min_v + alpha * code[j] as f32;
        }
    }

    #[allow(dead_code)]
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn id_at(&self, row: usize) -> Option<u32> {
        self.ids_slice().get(row).copied()
    }

    pub(crate) fn ids_vec(&self) -> Vec<u32> {
        self.ids_slice().to_vec()
    }

    pub(crate) fn decode_all(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.len() * self.dim];
        let mut tmp = vec![0.0f32; self.dim];
        for row in 0..self.len() {
            self.decode_row(row, &mut tmp);
            out[row * self.dim..(row + 1) * self.dim].copy_from_slice(&tmp);
        }
        out
    }

    pub(crate) fn adc_l2_sq(&self, query: &[f32], row: usize) -> f32 {
        let dim = self.dim;
        let start = row * dim;
        let end = start + dim;
        let code = &self.codes_slice()[start..end];
        adc_l2_sq_math(query, code, &self.min_vals, &self.max_vals)
    }

    fn ids_slice(&self) -> &[u32] {
        if let Some(mmap) = &self.mapped_ids {
            let len = mmap.len() / std::mem::size_of::<u32>();
            let ptr = mmap.as_ptr() as *const u32;
            // Alignment and size are validated in try_enable_mmap before mmap is stored.
            unsafe { std::slice::from_raw_parts(ptr, len) }
        } else {
            &self.ids
        }
    }

    fn codes_slice(&self) -> &[u8] {
        if let Some(mmap) = &self.mapped_codes {
            mmap
        } else {
            &self.codes
        }
    }
}
