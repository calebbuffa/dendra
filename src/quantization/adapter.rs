//! Adapters for integrating quantizers with VectorDB.
//!
//! This module provides implementations of Quantizer for:
//! - TurboQuantAdapter: Wraps TurboQuant (MSE and InnerProduct modes)
//! - NoOpQuantizer: Pass-through for disabled quantization

use super::{Dequantizer, QuantizeError, Quantizer, TurboQuant, TurboQuantMode};
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
            Err(QuantizeError::Serialization(
                "TurboQuant not available".to_string(),
            ))
        }
    }

    /// Delegate batch quantize to the inner TurboQuant's batch path.
    fn batch_quantize(&self, vectors: &[f32], output: &mut [u8]) -> Result<(), QuantizeError> {
        if let Some(turbo) = self.turbo.as_ref() {
            turbo.batch_quantize(vectors, output)
        } else {
            Err(QuantizeError::Serialization(
                "TurboQuant not available".to_string(),
            ))
        }
    }

    fn quantize_to_vec(&self, vector: &[f32]) -> Result<Vec<u8>, QuantizeError> {
        let mut out = vec![0u8; self.bytes_per_vec];
        self.quantize(vector, &mut out)?;
        Ok(out)
    }

    fn bytes_per_vector(&self) -> usize {
        self.bytes_per_vec
    }

    fn serialize(&self, writer: &mut dyn Write) -> Result<(), QuantizeError> {
        let encoded =
            serde_json::to_vec(self).map_err(|e| QuantizeError::Serialization(e.to_string()))?;
        writer.write_all(&encoded)?;
        Ok(())
    }
}

impl Dequantizer for TurboQuantAdapter {
    fn dequantize(&self, quantized: &[u8], out: &mut [f32]) -> Result<(), QuantizeError> {
        if let Some(turbo) = self.turbo.as_ref() {
            turbo.dequantize(quantized, out)
        } else {
            Err(QuantizeError::Serialization(
                "TurboQuant not available".to_string(),
            ))
        }
    }
}

/// No-op quantizer for disabled quantization (full precision f32).
/// Stores and retrieves vectors as raw f32 bytes.
#[derive(Serialize, Deserialize, Clone)]
pub struct NoOpQuantizer;

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
        let dst = bytemuck::cast_slice_mut(&mut out[..required]);
        dst.copy_from_slice(vector);
        Ok(())
    }

    fn quantize_to_vec(&self, vector: &[f32]) -> Result<Vec<u8>, QuantizeError> {
        Ok(bytemuck::cast_slice(vector).to_vec())
    }

    fn bytes_per_vector(&self) -> usize {
        0 // Variable per vector (depends on dimension)
    }

    fn serialize(&self, _writer: &mut dyn Write) -> Result<(), QuantizeError> {
        Ok(())
    }
}

impl Dequantizer for NoOpQuantizer {
    fn dequantize(&self, quantized: &[u8], out: &mut [f32]) -> Result<(), QuantizeError> {
        let required_bytes = out.len() * 4;
        if quantized.len() != required_bytes {
            return Err(QuantizeError::BufferTooSmall {
                required: required_bytes,
                provided: quantized.len(),
            });
        }
        let src: &[f32] = bytemuck::cast_slice(quantized);
        out.copy_from_slice(src);
        Ok(())
    }
}
