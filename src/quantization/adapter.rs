//! Adapters for integrating quantizers with VectorDB.
//!
//! This module provides implementations of Quantizer for:
//! - TurboQuantAdapter: Wraps TurboQuant (MSE and InnerProduct modes)
//! - NoOpQuantizer: Pass-through for disabled quantization

use super::{QuantizeError, Quantizer, TurboQuant, TurboQuantMode};
use crate::math;
use serde::{Deserialize, Serialize};
use std::io::Write;

/// Wraps TurboQuant with serialization support for VectorDB persistence.
#[derive(Serialize, Deserialize)]
pub struct TurboQuantAdapter {
    /// Serialized TurboQuant state (stored as binary blob)
    #[serde(with = "serde_bytes")]
    turbo_bytes: Vec<u8>,

    /// Dimension for reconstruction
    dimension: usize,

    /// Bit-width for reconstruction
    bit_width: u8,

    /// Mode for reconstruction
    mode: TurboQuantMode,

    /// Seed for determinism
    seed: u64,

    /// Bytes per quantized vector
    bytes_per_vec: usize,

    /// Cached TurboQuant instance (reconstructed on deserialization)
    #[serde(skip)]
    turbo: Option<Box<TurboQuant>>,
}

impl TurboQuantAdapter {
    /// Create from a TurboQuant instance.
    pub fn new(turbo: TurboQuant) -> Result<Self, QuantizeError> {
        let dimension = turbo.cfg.dim;
        let bit_width = turbo.cfg.bit_width;
        let mode = turbo.cfg.mode;
        let seed = turbo.cfg.seed.unwrap_or(42);
        let bytes_per_vec = turbo.bytes_per_vector();

        Ok(Self {
            turbo_bytes: Vec::new(),
            dimension,
            bit_width,
            mode,
            seed,
            bytes_per_vec,
            turbo: Some(Box::new(turbo)),
        })
    }
}

impl Quantizer for TurboQuantAdapter {
    fn quantize(&self, vector: &[f32], out: &mut [u8]) -> Result<(), QuantizeError> {
        if vector.len() != self.dimension {
            return Err(QuantizeError::DimensionMismatch {
                expected: self.dimension,
                received: vector.len(),
            });
        }

        if let Some(turbo) = self.turbo.as_ref() {
            turbo.quantize(vector, out)?;
            Ok(())
        } else {
            Err(QuantizeError::SerializationError(
                "TurboQuant not available".to_string(),
            ))
        }
    }

    /// Delegate batch quantize to the inner TurboQuant's batch path.
    fn batch_quantize(&self, vectors: &[f32], output: &mut [u8]) -> Result<(), QuantizeError> {
        if let Some(turbo) = self.turbo.as_ref() {
            turbo.batch_quantize(vectors, output)
        } else {
            Err(QuantizeError::SerializationError(
                "TurboQuant not available".to_string(),
            ))
        }
    }

    fn quantize_to_vec(&self, vector: &[f32]) -> Result<Vec<u8>, QuantizeError> {
        let mut out = vec![0u8; self.bytes_per_vec];
        self.quantize(vector, &mut out)?;
        Ok(out)
    }

    fn inner_product_query(&self, query: &[f32], encoded: &[u8]) -> Result<f32, QuantizeError> {
        if query.len() != self.dimension {
            return Err(QuantizeError::DimensionMismatch {
                expected: self.dimension,
                received: query.len(),
            });
        }

        if let Some(turbo) = self.turbo.as_ref() {
            turbo.inner_product_query(query, encoded)
        } else {
            Err(QuantizeError::SerializationError(
                "TurboQuant not available".to_string(),
            ))
        }
    }

    fn query_distance(
        &self,
        query: &[f32],
        encoded: &[u8],
        _metric: crate::distance::MetricFn,
    ) -> Result<f32, QuantizeError> {
        let score = self.inner_product_query(query, encoded)?;
        Ok(1.0 - score)
    }

    fn score(&self, encoded_a: &[u8], encoded_b: &[u8]) -> Result<f32, QuantizeError> {
        if let Some(turbo) = self.turbo.as_ref() {
            turbo.score(encoded_a, encoded_b)
        } else {
            Err(QuantizeError::SerializationError(
                "TurboQuant not available".to_string(),
            ))
        }
    }

    fn bytes_per_vector(&self) -> usize {
        self.bytes_per_vec
    }

    fn serialize(&self, writer: &mut dyn Write) -> Result<(), QuantizeError> {
        let encoded = serde_json::to_vec(self)
            .map_err(|e| QuantizeError::SerializationError(e.to_string()))?;
        writer.write_all(&encoded)?;
        Ok(())
    }
}

/// No-op quantizer for disabled quantization (full precision f32).
/// Stores vectors as raw f32 bytes, and uses SIMD-accelerated
/// dot products for the inner product and score computations.
#[derive(Serialize, Deserialize, Clone)]
pub struct NoOpQuantizer;

#[allow(dead_code)]
impl NoOpQuantizer {
    pub fn new() -> Self {
        Self
    }
}

impl Quantizer for NoOpQuantizer {
    fn quantize(&self, vector: &[f32], out: &mut [u8]) -> Result<(), QuantizeError> {
        let required = vector.len() * 4;
        if out.len() < required {
            return Err(QuantizeError::BufferTooSmall {
                required,
                provided: out.len(),
            });
        }
        // Bulk copy f32 bytes using bytemuck (safe, checked)
        let dst = bytemuck::cast_slice_mut(&mut out[..required]);
        dst.copy_from_slice(vector);
        Ok(())
    }

    fn quantize_to_vec(&self, vector: &[f32]) -> Result<Vec<u8>, QuantizeError> {
        Ok(bytemuck::cast_slice(vector).to_vec())
    }

    /// SIMD-accelerated inner product between f32 query and f32 encoded bytes.
    /// Uses `math::dot` (which uses `wide::f32x8` SIMD).
    fn inner_product_query(&self, query: &[f32], encoded: &[u8]) -> Result<f32, QuantizeError> {
        let dim = query.len();
        if encoded.len() != dim * 4 {
            return Err(QuantizeError::BufferTooSmall {
                required: dim * 4,
                provided: encoded.len(),
            });
        }

        // Cast encoded bytes to f32 slice via bytemuck — safe because:
        // 1. We verified len is dim * 4
        // 2. bytemuck checks alignment requirements
        let encoded_float: &[f32] = bytemuck::cast_slice(encoded);

        // Use the SIMD dot product from math.rs
        Ok(math::dot(query, encoded_float))
    }

    /// Compute distance by decoding and calling the metric function.
    fn query_distance(
        &self,
        query: &[f32],
        encoded: &[u8],
        metric: crate::distance::MetricFn,
    ) -> Result<f32, QuantizeError> {
        let dim = query.len();
        if encoded.len() != dim * 4 {
            return Err(QuantizeError::BufferTooSmall {
                required: dim * 4,
                provided: encoded.len(),
            });
        }

        // Decode f32 from bytes using bytemuck (safe, checks alignment)
        let decoded: &[f32] = bytemuck::cast_slice(encoded);
        metric(query, decoded).map_err(|e| QuantizeError::UnsupportedMethod(e.to_string()))
    }

    /// SIMD-accelerated dot product between two f32 vectors stored as bytes.
    fn score(&self, encoded_a: &[u8], encoded_b: &[u8]) -> Result<f32, QuantizeError> {
        if encoded_a.len() != encoded_b.len() || encoded_a.len() % 4 != 0 {
            return Err(QuantizeError::BufferTooSmall {
                required: encoded_a.len().max(encoded_b.len()),
                provided: encoded_a.len().min(encoded_b.len()),
            });
        }

        // Cast both to f32 slices and compute SIMD dot product
        let a: &[f32] = bytemuck::cast_slice(encoded_a);
        let b: &[f32] = bytemuck::cast_slice(encoded_b);
        Ok(math::dot(a, b))
    }

    fn bytes_per_vector(&self) -> usize {
        0 // Variable per vector (depends on dimension)
    }

    fn serialize(&self, _writer: &mut dyn Write) -> Result<(), QuantizeError> {
        // NoOpQuantizer is stateless; nothing to serialize
        Ok(())
    }
}
