///! Quantized Johnson-Lindenstrauss (QJL) algorthm
use crate::math::{dot_signed, l2_norm, mat_vec_mul, random_projection_matrix};
use crate::quantization::QuantizeError;
use faer::Mat;

#[derive(Clone, Debug)]
pub(crate) struct QjlConfig {
    pub dim: usize,
    pub projection_dim: Option<usize>,
    pub seed: Option<u64>,
}

impl QjlConfig {
    fn resolved_projection_dim(&self) -> usize {
        self.projection_dim.unwrap_or(self.dim)
    }

    fn validate(&self) -> Result<(), QuantizeError> {
        if self.dim == 0 {
            return Err(QuantizeError::DimensionMismatch {
                expected: 1,
                received: 0,
            });
        }
        let proj = self.resolved_projection_dim();
        if proj == 0 || proj > self.dim {
            return Err(QuantizeError::UnsupportedMethod(format!(
                "projection_dim {} must be in 1..={} for dim {}",
                proj, self.dim, self.dim
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct QjlCompressedVectors {
    pub signs: Vec<i8>,
    pub norms: Vec<f32>,
    pub n_vectors: usize,
    pub dim: usize,
    pub projection_dim: usize,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct Qjl {
    dim: usize,
    projection_dim: usize,
    seed: Option<u64>,
    projection: Mat<f32>,
}

impl Qjl {
    pub(crate) fn new(cfg: QjlConfig) -> Result<Self, QuantizeError> {
        cfg.validate()?;
        let projection_dim = cfg.resolved_projection_dim();
        let projection = random_projection_matrix(projection_dim, cfg.dim, cfg.seed);
        Ok(Self {
            dim: cfg.dim,
            projection_dim,
            seed: cfg.seed,
            projection,
        })
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn projection_dim(&self) -> usize {
        self.projection_dim
    }

    pub(crate) fn quantize_vector(
        &self,
        vector: &[f32],
    ) -> Result<QjlCompressedVectors, QuantizeError> {
        if vector.len() != self.dim {
            return Err(QuantizeError::DimensionMismatch {
                expected: self.dim,
                received: vector.len(),
            });
        }

        let norm = l2_norm(vector);

        let projected = mat_vec_mul(&self.projection, vector);
        let mut signs = vec![0i8; self.projection_dim];
        for j in 0..self.projection_dim {
            signs[j] = if projected[j] >= 0.0 { 1 } else { -1 };
        }

        Ok(QjlCompressedVectors {
            signs,
            norms: vec![norm],
            n_vectors: 1,
            dim: self.dim,
            projection_dim: self.projection_dim,
            seed: self.seed,
        })
    }

    pub(crate) fn quantize_vectors(
        &self,
        vectors: &[Vec<f32>],
    ) -> Result<QjlCompressedVectors, QuantizeError> {
        let n = vectors.len();
        let mut norms = vec![0.0f32; n];
        let mut signs = vec![0i8; n * self.projection_dim];

        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dim {
                return Err(QuantizeError::DimensionMismatch {
                    expected: self.dim,
                    received: v.len(),
                });
            }

            norms[i] = l2_norm(v);

            let projected = mat_vec_mul(&self.projection, v);
            for j in 0..self.projection_dim {
                signs[i * self.projection_dim + j] = if projected[j] >= 0.0 { 1 } else { -1 };
            }
        }

        Ok(QjlCompressedVectors {
            signs,
            norms,
            n_vectors: n,
            dim: self.dim,
            projection_dim: self.projection_dim,
            seed: self.seed,
        })
    }

    pub(crate) fn inner_product(
        &self,
        query: &[f32],
        compressed: &QjlCompressedVectors,
    ) -> Result<Vec<f32>, QuantizeError> {
        if query.len() != self.dim {
            return Err(QuantizeError::DimensionMismatch {
                expected: self.dim,
                received: query.len(),
            });
        }
        if compressed.dim != self.dim || compressed.projection_dim != self.projection_dim {
            return Err(QuantizeError::InvalidEncoding);
        }
        if compressed.signs.len() != compressed.n_vectors * self.projection_dim
            || compressed.norms.len() != compressed.n_vectors
        {
            return Err(QuantizeError::InvalidEncoding);
        }

        let projected_query = mat_vec_mul(&self.projection, query);

        let mut out = vec![0.0f32; compressed.n_vectors];
        let scale = (std::f32::consts::PI * 0.5).sqrt() / self.projection_dim as f32;

        for (i, slot) in out.iter_mut().enumerate() {
            let row = &compressed.signs[i * self.projection_dim..(i + 1) * self.projection_dim];
            *slot = scale * compressed.norms[i] * dot_signed(&projected_query, row);
        }

        Ok(out)
    }
}
