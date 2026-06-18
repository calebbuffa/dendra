//! TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
//!
//! Paper: Zandieh et al., arXiv:2504.19874v1 [cs.LG] 28 Apr 2025
//!
//! This module implements TurboQuantmse (Theorem 1) and TurboQuantprod (Theorem 2),
//! achieving near-optimal distortion bounds for MSE and inner product quantization.
//! Both algorithms are data-oblivious (online) and suitable for applications like
//! KV cache compression and nearest neighbor search (Section 4.4).

use crate::quantization::{
    Dequantizer, QuantizeError, Quantizer,
    bitpack::{pack_indices, unpack_indices},
};
use crate::quantization::math::{
    batch_mat_mul, batch_mat_t_mul, beta_pdf, dot_packed_signs, dot_scaled, l2_norm,
    mat_t_vec_mul, mat_vec_mul, random_orthogonal_matrix, random_projection_matrix, trapezoid,
};
use faer::Mat;
use log::debug;

const NORM_BYTES: usize = 4;
const LLOYD_MAX_GRID_POINTS: usize = 10_000;
const LLOYD_MAX_MAX_ITER: usize = 200;
const LLOYD_MAX_TOL: f64 = 1e-10;
const QJL_SEED_OFFSET: u64 = 1_000_000;

/// Quantization optimization objective.
///
/// - **Mse**: Minimize MSE distortion (Theorem 1, Section 3.1)
/// - **InnerProduct**: Minimize inner product error with unbiased estimation (Theorem 2, Section 3.2)
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TurboQuantMode {
    Mse,
    InnerProduct,
}

/// Configuration for TurboQuant instantiation.
///
/// **Paper References**:
/// - Theorem 1: MSE quantizer requires bit_width $\geq$ 1
/// - Theorem 2: Inner product quantizer requires bit_width $\geq$ 2 (one bit reserved for QJL residual)
/// - Core algorithms: Sections 3.1-3.2 (no outlier extension in paper-parity path)
#[derive(Clone, Debug)]
pub struct TurboQuantConfig {
    /// Dimension of input vectors (d in the paper)
    pub dim: usize,
    /// Bits per coordinate. For InnerProduct mode, one bit is used for residual QJL.
    pub bit_width: u8,
    /// Optimization objective: Mse (Theorem 1) or InnerProduct (Theorem 2)
    pub mode: TurboQuantMode,
    /// Seed for random rotation matrix (determinism)
    pub seed: Option<u64>,
}

impl TurboQuantConfig {
    pub fn validate(&self) -> Result<(), QuantizeError> {
        if self.dim == 0 {
            return Err(QuantizeError::DimensionMismatch {
                expected: 1,
                received: 0,
            });
        }
        if !(1..=4).contains(&self.bit_width) {
            return Err(QuantizeError::UnsupportedMethod(format!(
                "bit_width {} is not supported (expected 1..=4)",
                self.bit_width
            )));
        }
        if self.mode == TurboQuantMode::InnerProduct && self.bit_width < 2 {
            return Err(QuantizeError::UnsupportedMethod(
                "inner_product mode requires bit_width $\\geq$ 2".to_string(),
            ));
        }
        Ok(())
    }

    /// Effective bit-width for MSE quantizer.
    /// For InnerProduct mode, one bit is reserved for residual QJL (Algorithm 2, Line 2).
    #[inline]
    fn bit_width(&self) -> u8 {
        if self.mode == TurboQuantMode::InnerProduct {
            self.bit_width - 1
        } else {
            self.bit_width
        }
    }
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            bit_width: 2,
            mode: TurboQuantMode::InnerProduct,
            seed: Some(42),
        }
    }
}

/// Lloyd-Max codebook for scalar quantization.
///
/// **Paper Reference**: Equation (4), Section 3.1
/// Centroids and boundaries are precomputed by solving the k-means optimization:
/// $\min \Sigma_j \int (x - c_j)^2 f_X(x) dx$
/// where $f_X$ is the Beta distribution induced by random rotation (Lemma 1).
#[derive(Clone, Debug)]
struct ScalarCodebook {
    /// Quantization centroids (codewords)
    centroids: Vec<f32>,
    /// Voronoi partition boundaries (midpoints between consecutive centroids)
    boundaries: Vec<f32>,
}

/// TurboQuant quantizer instance.
///
/// **Paper References**:
/// - Algorithm 1: TurboQuantmse (MSE optimization, Theorem 1)
/// - Algorithm 2: TurboQuantprod (Inner product optimization, Theorem 2)
#[derive(Debug)]
pub struct TurboQuant {
    /// Public config for access by adapters
    pub cfg: TurboQuantConfig,
    /// Codebook for MSE quantizer (Algorithm 1, Line 3)
    inlier_codebook: ScalarCodebook,
    /// Random rotation matrix $\Pi \in \mathbb{R}^{d \times d}$ (Algorithm 1, Line 2)
    /// See Lemma 1: each coordinate of $\Pi \cdot x$ follows Beta distribution
    rotation: Mat<f32>,
    /// Gaussian projection matrix $S \in \mathbb{R}^{d \times d}$ for QJL on residual (Algorithm 2, Line 3)
    /// Used only in InnerProduct mode. See Definition 1: $S_{ij} \sim N(0,1)$
    projection: Option<Mat<f32>>,
    /// Byte count for packed quantization indices
    packed_index_bytes: usize,
    /// Byte count for QJL residual sign bits
    qjl_sign_bytes: usize,
}

impl TurboQuant {
    /// **Algorithm**: Combines Algorithm 1 (MSE) and Algorithm 2 (InnerProduct)
    /// - Precomputes Lloyd-Max codebook based on Beta distribution (Lemma 1)
    /// - Generates random rotation $\Pi$ for data-oblivious property
    /// - For InnerProduct mode, generates Gaussian projection $S$ for QJL
    pub fn new(cfg: TurboQuantConfig) -> Result<Self, QuantizeError> {
        cfg.validate()?;

        let inlier_bw = cfg.bit_width();
        let inlier_codebook = compute_codebook(cfg.dim, inlier_bw)?;

        // Algorithm 1, Line 2: Generate random rotation matrix $\Pi$ via QR on Gaussian
        // This induces Beta distribution on rotated coordinates (Lemma 1)
        let rotation = random_orthogonal_matrix(cfg.dim, cfg.seed);

        // Algorithm 2, Line 3: For InnerProduct mode, generate Gaussian projection S
        // QJL will be applied to residual vector (Algorithm 2, Line 7)
        // See Definition 1: $S_{ij} \sim N(0,1)$ (i.i.d. standard normal)
        let projection = if cfg.mode == TurboQuantMode::InnerProduct {
            let pseed = cfg.seed.map(|s| s + QJL_SEED_OFFSET);
            Some(random_projection_matrix(cfg.dim, cfg.dim, pseed))
        } else {
            None
        };

        let packed_index_bytes = (cfg.dim * inlier_bw as usize).div_ceil(8);
        let qjl_sign_bytes = cfg.dim.div_ceil(8);
        debug!(
            "Initialized TurboQuant: dim={}, bit_width={}, bit_width={}, mode={:?}",
            cfg.dim, cfg.bit_width, inlier_bw, cfg.mode
        );

        Ok(Self {
            cfg,
            inlier_codebook,
            rotation,
            projection,
            packed_index_bytes,
            qjl_sign_bytes,
        })
    }

    #[inline]
    fn indices_offset(&self) -> usize {
        self.packed_index_bytes
    }

    #[inline]
    fn residual_signs_offset(&self) -> usize {
        self.indices_offset()
    }

    #[inline]
    fn residual_norm_offset(&self) -> usize {
        self.residual_signs_offset() + self.qjl_sign_bytes
    }

    #[inline]
    fn norm_offset(&self) -> usize {
        if self.cfg.mode == TurboQuantMode::InnerProduct {
            self.residual_norm_offset() + NORM_BYTES
        } else {
            self.indices_offset()
        }
    }

    #[inline]
    fn expected_quantized_len(&self) -> usize {
        self.norm_offset() + NORM_BYTES
    }

    /// Unpack codebook indices from bit-packed representation.
    ///
    /// **Paper Reference**: Algorithm 1, Line 9 (dequantization)
    /// Reverses the packing applied in quantize() to recover centroid indices.
    fn decode_indices(&self, encoded: &[u8]) -> Result<Vec<u8>, QuantizeError> {
        unpack_indices(
            &encoded[..self.indices_offset()],
            self.cfg.bit_width(),
            self.cfg.dim,
        )
    }

    /// Estimate inner product $\langle \text{query}, x \rangle$ using quantized representation of x.
    ///
    /// **For Mse mode**: Returns MSE-based reconstruction dot product
    /// **For InnerProduct mode**: Returns unbiased inner product estimator
    ///
    /// **Paper Reference**: Theorem 2, Algorithm 2
    /// The estimator combines:
    /// 1. MSE quantization score (main term)
    /// 2. QJL residual correction (unbiasedness via Lemma 4)
    ///
    /// **Nearest Neighbor Application**: This enables efficient inner product-based
    /// NN search by avoiding full dequantization. Used in Section 4.4 (NN experiments).
    pub fn inner_product_query(&self, query: &[f32], encoded: &[u8]) -> Result<f32, QuantizeError> {
        if query.len() != self.cfg.dim {
            return Err(QuantizeError::DimensionMismatch {
                expected: self.cfg.dim,
                received: query.len(),
            });
        }
        if encoded.len() < self.expected_quantized_len() {
            return Err(QuantizeError::BufferTooSmall {
                required: self.expected_quantized_len(),
                provided: encoded.len(),
            });
        }

        let norm = Self::read_f32_at(encoded, self.norm_offset())?;
        let recon_unit = self.reconstruct_unit_from_encoded(encoded)?;
        let mse_score = dot_scaled(query, &recon_unit, norm);

        if self.cfg.mode != TurboQuantMode::InnerProduct {
            return Ok(mse_score);
        }

        // Algorithm 2, Lines 10-11: QJL-based residual correction for unbiased inner product
        let residual_norm = Self::read_f32_at(encoded, self.residual_norm_offset())?;
        let projection = self
            .projection
            .as_ref()
            .ok_or_else(|| QuantizeError::UnsupportedMethod("missing projection".to_string()))?;

        // Project query onto residual subspace (Algorithm 2, Line 10 operation)
        let projected_query = mat_vec_mul(projection, query);

        // Retrieve QJL signs from encoded buffer
        let signs = &encoded
            [self.residual_signs_offset()..self.residual_signs_offset() + self.qjl_sign_bytes];
        let dot = dot_packed_signs(&projected_query, signs, self.cfg.dim);

        // Scaling factor from Lemma 4 (QJL performance guarantee): $\sqrt{\pi/2} / d$
        let scale = (std::f32::consts::PI * 0.5).sqrt() / self.cfg.dim as f32;
        // Unbiased inner product: MSE term + residual correction (Theorem 2)
        Ok(mse_score + norm * residual_norm * scale * dot)
    }

    fn read_f32_at(bytes: &[u8], offset: usize) -> Result<f32, QuantizeError> {
        if bytes.len() < offset + NORM_BYTES {
            return Err(QuantizeError::InvalidEncoding);
        }
        let v = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        if !v.is_finite() {
            return Err(QuantizeError::InvalidEncoding);
        }
        Ok(v)
    }

    #[inline]
    fn quantize_scalar_with_book(book: &ScalarCodebook, value: f32) -> u8 {
        let v = value.clamp(-1.0, 1.0);
        let boundaries = &book.boundaries;
        let mut lo = 0usize;
        let mut hi = boundaries.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if boundaries[mid] <= v {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.saturating_sub(1).clamp(0, book.centroids.len() - 1) as u8
    }

    #[inline]
    fn dequantize_scalar_with_book(book: &ScalarCodebook, idx: u8) -> f32 {
        book.centroids.get(idx as usize).copied().unwrap_or(0.0)
    }

    /// Reconstruct unit-norm vector from quantization indices.
    ///
    /// **Paper Reference**: Algorithm 1, Lines 8-10 (dequantization)
    /// 1. Look up centroid for each coordinate: $\tilde{y}_j = c_{\text{idx}_j}$
    /// 2. Apply inverse rotation: $\tilde{x} = \Pi^T \cdot \tilde{y}$
    fn reconstruct_unit_from_indices(&self, indices: &[u8]) -> Vec<f32> {
        let mut rotated = vec![0.0f32; self.cfg.dim];
        for (i, slot) in rotated.iter_mut().enumerate() {
            *slot = Self::dequantize_scalar_with_book(&self.inlier_codebook, indices[i]);
        }
        mat_vec_mul(&self.rotation, &rotated)
    }

    fn reconstruct_unit_from_encoded(&self, encoded: &[u8]) -> Result<Vec<f32>, QuantizeError> {
        let indices = self.decode_indices(encoded)?;
        Ok(self.reconstruct_unit_from_indices(&indices))
    }

    fn dequantized_full_from_encoded(&self, encoded: &[u8]) -> Result<Vec<f32>, QuantizeError> {
        let norm = Self::read_f32_at(encoded, self.norm_offset())?;
        let mut unit = self.reconstruct_unit_from_encoded(encoded)?;
        for v in &mut unit {
            *v *= norm;
        }
        Ok(unit)
    }
}

impl Quantizer for TurboQuant {
    /// Batch quantize: process all vectors using batched matrix multiplies.
    /// This mirrors the Python reference which does all matrix ops in batch via numpy.
    fn batch_quantize(&self, vectors: &[f32], output: &mut [u8]) -> Result<(), QuantizeError> {
        let batch_start = std::time::Instant::now();
        let dim = self.cfg.dim;
        let bytes_per = self.expected_quantized_len();
        let n = vectors.len() / dim;

        debug!(
            "[batch_quantize] Starting: {} vectors, {} dim, {} bytes_per_vec",
            n, dim, bytes_per
        );

        if output.len() != n * bytes_per {
            return Err(QuantizeError::BufferTooSmall {
                required: n * bytes_per,
                provided: output.len(),
            });
        }

        // Phase 1: compute all norms and normalize (batch)
        let p1_start = std::time::Instant::now();
        let mut norms = vec![0.0f32; n];
        let mut unit_vectors = vec![0.0f32; vectors.len()];
        for i in 0..n {
            let base = i * dim;
            let slice = &vectors[base..base + dim];
            let norm = l2_norm(slice);
            if norm == 0.0 {
                return Err(QuantizeError::ZeroNormVector);
            }
            norms[i] = norm;
            let inv = 1.0 / norm;
            for j in 0..dim {
                unit_vectors[base + j] = slice[j] * inv;
            }
        }
        debug!(
            "  Phase 1 (norms/normalize): {:.3}ms",
            p1_start.elapsed().as_secs_f64() * 1000.0
        );

        // Phase 2: rotate all vectors at once: rotated = Pi^T * unit_vectors
        let p2_start = std::time::Instant::now();
        let rotated = batch_mat_t_mul(&self.rotation, &unit_vectors, dim);
        debug!(
            "  Phase 2 (batch_mat_t_mul rotation): {:.3}ms",
            p2_start.elapsed().as_secs_f64() * 1000.0
        );

        // Phase 3: quantize each coordinate of each vector (scalar quant)
        // Store indices temporarily per vector
        let p3_start = std::time::Instant::now();
        let inlier_bw = self.cfg.bit_width();
        let mut all_indices = vec![0u8; n * dim];
        let mut recon_rotated = vec![0.0f32; n * dim];

        for vec_idx in 0..n {
            let base = vec_idx * dim;
            let rbase = vec_idx * dim;
            for coord in 0..dim {
                let idx =
                    Self::quantize_scalar_with_book(&self.inlier_codebook, rotated[base + coord]);
                all_indices[rbase + coord] = idx;
                recon_rotated[rbase + coord] =
                    Self::dequantize_scalar_with_book(&self.inlier_codebook, idx);
            }
        }
        debug!(
            "  Phase 3 (scalar quantization): {:.3}ms",
            p3_start.elapsed().as_secs_f64() * 1000.0
        );

        // Phase 4: pack indices for all vectors
        let p4_start = std::time::Instant::now();
        let packed_bytes = self.packed_index_bytes;
        for i in 0..n {
            let base = i * dim;
            let indices_slice = &all_indices[base..base + dim];
            let packed = crate::quantization::bitpack::pack_indices(indices_slice, inlier_bw)?;
            let dst = &mut output[i * bytes_per..i * bytes_per + packed_bytes];
            dst.copy_from_slice(&packed);
        }
        debug!(
            "  Phase 4 (pack indices): {:.3}ms",
            p4_start.elapsed().as_secs_f64() * 1000.0
        );

        // Phase 5 (InnerProduct mode): QJL on residuals
        // OPTIMIZED: Batch all residual projections into ONE matrix multiply instead of n individual calls
        if self.cfg.mode == TurboQuantMode::InnerProduct {
            let p5_start = std::time::Instant::now();
            debug!("  Phase 5 (InnerProduct QJL) starting...");

            // Compute MSE reconstructions in one batch: recon_unit = Pi * recon_rotated
            let p5a_start = std::time::Instant::now();
            let recon_unit = batch_mat_mul(&self.rotation, &recon_rotated, dim);
            debug!(
                "    5a (batch_mat_mul recon): {:.3}ms",
                p5a_start.elapsed().as_secs_f64() * 1000.0
            );

            // Get projection matrix upfront (needed for batch operation)
            let projection = self.projection.as_ref().ok_or_else(|| {
                QuantizeError::UnsupportedMethod("missing projection".to_string())
            })?;

            // Compute all residuals and their norms in ONE pass (no per-vector allocation)
            // This eliminates the vec![0.0f32; dim] allocation that happened n times
            let p5b_start = std::time::Instant::now();
            let mut all_residuals = vec![0.0f32; n * dim];
            let mut residual_norms = vec![0.0f32; n];

            for i in 0..n {
                let base = i * dim;
                let mut res_norm_sq = 0.0f32;
                for j in 0..dim {
                    let res = unit_vectors[base + j] - recon_unit[base + j];
                    all_residuals[base + j] = res;
                    res_norm_sq = res.mul_add(res, res_norm_sq);
                }
                residual_norms[i] = res_norm_sq.sqrt();
            }
            debug!(
                "    5b (compute residuals): {:.3}ms",
                p5b_start.elapsed().as_secs_f64() * 1000.0
            );

            // PROJECT ALL RESIDUALS AT ONCE: projection * all_residuals (d x n)
            // This single batch_mat_mul replaces the loop of n individual mat_vec_mul calls!
            // For n=100k, this is 100k->1 reduction in matrix multiply operations
            let p5c_start = std::time::Instant::now();
            let all_projected = batch_mat_mul(projection, &all_residuals, dim);
            debug!(
                "    5c (batch_mat_mul projection): {:.3}ms",
                p5c_start.elapsed().as_secs_f64() * 1000.0
            );

            // Encode signs and store norms per-vector
            let p5d_start = std::time::Instant::now();
            let sign_start = self.residual_signs_offset();
            let sign_len = self.qjl_sign_bytes;
            let norm_off = self.residual_norm_offset();

            for i in 0..n {
                let base = i * dim;

                // Store QJL signs from batch-computed projections
                let sign_end = sign_start + sign_len;
                let signs_slice = &mut output[i * bytes_per + sign_start..i * bytes_per + sign_end];
                signs_slice.fill(0);
                for j in 0..dim {
                    if all_projected[base + j] >= 0.0 {
                        set_bit(signs_slice, j, true);
                    }
                }

                // Store residual norm
                let dst = &mut output[i * bytes_per + norm_off..i * bytes_per + norm_off + 4];
                dst.copy_from_slice(&residual_norms[i].to_le_bytes());
            }
            debug!(
                "    5d (encode signs/norms): {:.3}ms",
                p5d_start.elapsed().as_secs_f64() * 1000.0
            );
            debug!(
                "  Phase 5 (total): {:.3}ms",
                p5_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Phase 6: store original norms for all vectors
        let p6_start = std::time::Instant::now();
        let norm_off = self.norm_offset();
        for i in 0..n {
            let dst = &mut output[i * bytes_per + norm_off..i * bytes_per + norm_off + 4];
            dst.copy_from_slice(&norms[i].to_le_bytes());
        }
        debug!(
            "  Phase 6 (store norms): {:.3}ms",
            p6_start.elapsed().as_secs_f64() * 1000.0
        );
        debug!(
            "[batch_quantize] Complete in {:.3}s",
            batch_start.elapsed().as_secs_f64()
        );

        Ok(())
    }

    fn quantize(&self, vector: &[f32], out: &mut [u8]) -> Result<(), QuantizeError> {
        if vector.len() != self.cfg.dim {
            return Err(QuantizeError::DimensionMismatch {
                expected: self.cfg.dim,
                received: vector.len(),
            });
        }
        let expected = self.expected_quantized_len();
        if out.len() < expected {
            return Err(QuantizeError::BufferTooSmall {
                required: expected,
                provided: out.len(),
            });
        }

        let norm = l2_norm(vector);
        if norm == 0.0 {
            return Err(QuantizeError::ZeroNormVector);
        }

        let mut unit = vec![0.0f32; self.cfg.dim];
        for (i, slot) in unit.iter_mut().enumerate() {
            *slot = vector[i] / norm;
        }

        // Algorithm 1, Line 5: Apply inverse rotation to normalize coordinates
        // rotated = $\Pi^T \cdot \text{unit}$ (each coordinate becomes Beta-distributed)
        let rotated = mat_t_vec_mul(&self.rotation, &unit);

        out[..expected].fill(0);
        let mut indices = vec![0u8; self.cfg.dim];
        let mut recon_rotated = vec![0.0f32; self.cfg.dim];

        // Algorithm 1, Line 6: Quantize each coordinate independently
        // $\text{idx}_j = \arg\min_k |y_j - c_k|$ (nearest centroid)
        for i in 0..self.cfg.dim {
            let idx = Self::quantize_scalar_with_book(&self.inlier_codebook, rotated[i]);
            indices[i] = idx;
            recon_rotated[i] = Self::dequantize_scalar_with_book(&self.inlier_codebook, idx);
        }

        // Pack indices into bit-efficient format (true b bits per coordinate)
        // Achieves optimal bit budget: $b \cdot d$ bits total
        let packed = pack_indices(&indices, self.cfg.bit_width())?;
        if packed.len() != self.packed_index_bytes {
            return Err(QuantizeError::InvalidEncoding);
        }
        out[..self.packed_index_bytes].copy_from_slice(&packed);

        if self.cfg.mode == TurboQuantMode::InnerProduct {
            // Algorithm 2, Lines 5-7: QJL on residual for unbiased inner product
            let recon_unit = mat_vec_mul(&self.rotation, &recon_rotated);
            let mut residual = vec![0.0f32; self.cfg.dim];
            // Algorithm 2, Line 6: r = x - Q^{-1}(Q(x))
            for i in 0..self.cfg.dim {
                residual[i] = unit[i] - recon_unit[i];
            }
            let residual_norm = l2_norm(&residual);

            let projection = self.projection.as_ref().ok_or_else(|| {
                QuantizeError::UnsupportedMethod("missing projection".to_string())
            })?;
            // Algorithm 2, Line 7: $q_{jl} = \text{sign}(S \cdot r)$ where $S \sim N(0,1)^{d \times d}$
            // See Definition 1 (QJL) and Lemma 4 (performance guarantee)
            let projected = mat_vec_mul(projection, &residual);
            let signs_slice = &mut out
                [self.residual_signs_offset()..self.residual_signs_offset() + self.qjl_sign_bytes];
            for (i, &v) in projected.iter().enumerate() {
                if v >= 0.0 {
                    set_bit(signs_slice, i, true);
                }
            }

            // Store residual norm for reconstruction scaling (Algorithm 2, Line 11: $\gamma = \|r\|_2$)
            let roff = self.residual_norm_offset();
            out[roff..roff + NORM_BYTES].copy_from_slice(&residual_norm.to_le_bytes());
        }

        let noff = self.norm_offset();
        out[noff..noff + NORM_BYTES].copy_from_slice(&norm.to_le_bytes());
        Ok(())
    }

    /// Score inner product between two quantized vectors in quantized space.
    ///
    /// **For Mse mode**: Computes $\langle a_{\text{recon}}, b_{\text{recon}} \rangle$ using rotated coordinate dot product.
    /// Avoids full dequantization overhead.
    ///
    /// **For InnerProduct mode**: Uses symmetric estimator (average of both directions)
    /// for unbiased estimation (Theorem 2).
    ///
    /// **Nearest Neighbor Improvement** (Paper Section 4.4):
    /// This quantized-space scoring is crucial for efficient NN search:
    /// - No memory overhead from reconstructing full float vectors
    /// - $O(d)$ instead of $O(d)$ with dequantization (same asymptotic but better constants)
    /// - Direct computation on packed binary representation
    fn score(&self, a: &[u8], b: &[u8]) -> Result<f32, QuantizeError> {
        let expected = self.expected_quantized_len();
        if a.len() < expected || b.len() < expected {
            return Err(QuantizeError::BufferTooSmall {
                required: expected,
                provided: a.len().min(b.len()),
            });
        }

        let norm_a = Self::read_f32_at(a, self.norm_offset())?;
        let norm_b = Self::read_f32_at(b, self.norm_offset())?;
        let ia = self.decode_indices(a)?;
        let ib = self.decode_indices(b)?;

        if self.cfg.mode == TurboQuantMode::Mse {
            // Compute $\langle a, b \rangle$ in rotated codebook space, scaled by original norms
            // Avoids full reconstruction to original space
            let mut dot_rot = 0.0f32;
            for i in 0..self.cfg.dim {
                let ca = Self::dequantize_scalar_with_book(&self.inlier_codebook, ia[i]);
                let cb = Self::dequantize_scalar_with_book(&self.inlier_codebook, ib[i]);
                dot_rot = ca.mul_add(cb, dot_rot);
            }
            return Ok(norm_a * norm_b * dot_rot);
        }

        // Inner-product mode: symmetric estimator for unbiasedness (Theorem 2)
        let mut qa = self.reconstruct_unit_from_indices(&ia);
        let mut qb = self.reconstruct_unit_from_indices(&ib);
        for v in &mut qa {
            *v *= norm_a;
        }
        for v in &mut qb {
            *v *= norm_b;
        }

        let ab = self.inner_product_query(&qb, a)?;
        let ba = self.inner_product_query(&qa, b)?;
        Ok((ab + ba) * 0.5)
    }

    fn bytes_per_vector(&self) -> usize {
        self.expected_quantized_len()
    }
}

impl Dequantizer for TurboQuant {
    fn dequantize(&self, quantized: &[u8], out: &mut [f32]) -> Result<(), QuantizeError> {
        let expected = self.expected_quantized_len();
        if quantized.len() < expected {
            return Err(QuantizeError::BufferTooSmall {
                required: expected,
                provided: quantized.len(),
            });
        }
        if out.len() < self.cfg.dim {
            return Err(QuantizeError::BufferTooSmall {
                required: self.cfg.dim,
                provided: out.len(),
            });
        }

        let full = self.dequantized_full_from_encoded(quantized)?;
        out[..self.cfg.dim].copy_from_slice(&full);
        Ok(())
    }
}

fn set_bit(bytes: &mut [u8], bit_index: usize, value: bool) {
    let byte_index = bit_index / 8;
    let bit = bit_index % 8;
    if value {
        bytes[byte_index] |= 1 << bit;
    } else {
        bytes[byte_index] &= !(1 << bit);
    }
}

/// Precompute Lloyd-Max codebook for scalar quantization.
///
/// **Paper Reference**: Equation (4), Section 3.1
/// Solves the continuous k-means problem:
/// $$\min \sum_j \int_{\text{boundary}_{j}}^{\text{boundary}_{j+1}} (x - c_j)^2 \cdot f_X(x) \, dx$$
/// where $f_X(x) = \text{Beta}$ distribution (from random rotation of unit hypersphere)
///
/// **Algorithm**:
/// 1. Initialize centroids uniformly on $[-1, 1]$
/// 2. For each iteration:
///    a. Compute Voronoi partition boundaries (midpoints between centroids)
///    b. Update each centroid to weighted mean within its partition
///    c. Check convergence (max centroid shift < tolerance)
/// 3. Return centroids and boundaries
///
/// **Numerical Method**: Trapezoidal integration on sampled Beta distribution points
fn compute_codebook(dim: usize, bit_width: u8) -> Result<ScalarCodebook, QuantizeError> {
    let n_centroids = 1usize << bit_width;
    if n_centroids == 0 {
        return Err(QuantizeError::InvalidEncoding);
    }

    let mut centroids = (0..n_centroids)
        .map(|i| {
            let frac = (i as f64 + 0.5) / n_centroids as f64;
            -1.0 + 2.0 * frac
        })
        .collect::<Vec<_>>();

    let n_points = LLOYD_MAX_GRID_POINTS;
    let mut x_grid = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let t = i as f64 / (n_points as f64 - 1.0);
        x_grid.push(-0.9999 + 1.9998 * t);
    }
    let pdf_vals = x_grid.iter().map(|&x| beta_pdf(x, dim)).collect::<Vec<_>>();

    for _ in 0..LLOYD_MAX_MAX_ITER {
        let mut boundaries = vec![0.0f64; n_centroids + 1];
        boundaries[0] = -1.0;
        boundaries[n_centroids] = 1.0;
        for i in 0..(n_centroids - 1) {
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) * 0.5;
        }

        let mut new_centroids = vec![0.0f64; n_centroids];
        for i in 0..n_centroids {
            let lo = boundaries[i];
            let hi = boundaries[i + 1];
            let mut xs = Vec::new();
            let mut ws = Vec::new();
            for (j, &x) in x_grid.iter().enumerate() {
                let inside = if i + 1 == n_centroids {
                    x >= lo && x <= hi
                } else {
                    x >= lo && x < hi
                };
                if inside {
                    xs.push(x);
                    ws.push(pdf_vals[j]);
                }
            }
            if xs.len() < 2 {
                new_centroids[i] = centroids[i];
                continue;
            }

            let total_weight = trapezoid(&xs, &ws);
            if total_weight <= 0.0 {
                new_centroids[i] = centroids[i];
                continue;
            }
            let xw = xs
                .iter()
                .zip(ws.iter())
                .map(|(&x, &w)| x * w)
                .collect::<Vec<_>>();
            let numerator = trapezoid(&xs, &xw);
            new_centroids[i] = numerator / total_weight;
        }

        let mut max_shift = 0.0f64;
        for i in 0..n_centroids {
            max_shift = max_shift.max((new_centroids[i] - centroids[i]).abs());
            centroids[i] = new_centroids[i];
        }
        if max_shift < LLOYD_MAX_TOL {
            break;
        }
    }

    let mut boundaries = vec![0.0f32; n_centroids + 1];
    boundaries[0] = -1.0;
    boundaries[n_centroids] = 1.0;
    for i in 0..(n_centroids - 1) {
        boundaries[i + 1] = ((centroids[i] + centroids[i + 1]) * 0.5) as f32;
    }

    Ok(ScalarCodebook {
        centroids: centroids.iter().map(|&x| x as f32).collect(),
        boundaries,
    })
}
