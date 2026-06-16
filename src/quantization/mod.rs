mod adapter;
mod bitpack;
mod qjl;
mod turbo_quant;

use crate::distance::MetricFn;
use serde::{Deserialize, Serialize};
use std::io::Write;
use thiserror::Error;

pub use adapter::{NoOpQuantizer, TurboQuantAdapter};
pub use turbo_quant::{TurboQuant, TurboQuantConfig, TurboQuantMode};

#[derive(Error, Debug)]
pub enum QuantizeError {
    #[error("Dimension mismatch: expected {expected}, received {received}")]
    DimensionMismatch { expected: usize, received: usize },
    #[error("Buffer too small: required {required} bytes, provided {provided} bytes")]
    BufferTooSmall { required: usize, provided: usize },
    #[error("Invalid quantized encoding")]
    InvalidEncoding,
    #[error("Zero-norm vector")]
    ZeroNormVector,
    #[error("Unsupported quantization method: {0}")]
    UnsupportedMethod(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl From<std::io::Error> for QuantizeError {
    fn from(err: std::io::Error) -> Self {
        QuantizeError::IoError(err.to_string())
    }
}

impl From<bincode::error::EncodeError> for QuantizeError {
    fn from(err: bincode::error::EncodeError) -> Self {
        QuantizeError::SerializationError(err.to_string())
    }
}

impl From<bincode::error::DecodeError> for QuantizeError {
    fn from(err: bincode::error::DecodeError) -> Self {
        QuantizeError::SerializationError(err.to_string())
    }
}

/// Quantization mode for VectorDB storage.
///
/// Controls how vectors are stored and searched:
/// - **Disabled**: Store full f32 precision (no compression)
/// - **Mse**: TurboQuantmse with reduced distortion (Section 3.1, Theorem 1)
/// - **InnerProduct**: TurboQuantprod with unbiased inner product estimates (Section 3.2, Theorem 2)
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, bincode::Encode, bincode::Decode,
)]
pub enum QuantizationMode {
    Disabled,
    Mse,
    InnerProduct,
}

/// Configuration for VectorDB quantization.
///
/// When enabled, vectors are stored as quantized u8 arrays instead of f32,
/// while maintaining near-optimal distortion bounds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct QuantizationConfig {
    pub enabled: bool,
    /// Bits per coordinate (1-4 recommended)
    pub bit_width: u8,
    /// Optimization objective
    pub mode: QuantizationMode,
}

impl QuantizationConfig {
    pub const NONE: QuantizationConfig = QuantizationConfig {
        enabled: false,
        bit_width: 0,
        mode: QuantizationMode::Disabled,
    };
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bit_width: 4,
            mode: QuantizationMode::InnerProduct,
        }
    }
}

pub trait Quantizer: Send + Sync {
    fn quantize(&self, vector: &[f32], out: &mut [u8]) -> Result<(), QuantizeError>;

    /// Quantize many vectors at once (batch mode).
    /// `vectors` is flat [f32] of shape (n * dim).
    /// `output` is flat [u8] of shape (n * bytes_per_vector).
    /// Default implementation calls quantize() per vector.
    fn batch_quantize(&self, vectors: &[f32], output: &mut [u8]) -> Result<(), QuantizeError> {
        let bytes_per = self.bytes_per_vector();
        if bytes_per == 0 {
            return Err(QuantizeError::UnsupportedMethod(
                "batch_quantize requires fixed bytes_per_vector".to_string(),
            ));
        }
        // Compute n = number of vectors = output.len() / bytes_per
        let n = output.len() / bytes_per;
        if n == 0 {
            return Ok(());
        }
        let dim = vectors.len() / n;
        for i in 0..n {
            let src = &vectors[i * dim..(i + 1) * dim];
            let dst = &mut output[i * bytes_per..(i + 1) * bytes_per];
            self.quantize(src, dst)?;
        }
        Ok(())
    }

    fn quantize_to_vec(&self, vector: &[f32]) -> Result<Vec<u8>, QuantizeError> {
        let bytes = self.bytes_per_vector();
        if bytes == 0 {
            return Err(QuantizeError::UnsupportedMethod(
                "quantize_to_vec requires fixed bytes_per_vector".to_string(),
            ));
        }
        let mut out = vec![0u8; bytes];
        self.quantize(vector, &mut out)?;
        Ok(out)
    }

    fn inner_product_query(&self, query: &[f32], encoded: &[u8]) -> Result<f32, QuantizeError> {
        let _ = (query, encoded);
        Err(QuantizeError::UnsupportedMethod(
            "inner_product_query not implemented".to_string(),
        ))
    }

    fn query_distance(
        &self,
        query: &[f32],
        encoded: &[u8],
        _metric: MetricFn,
    ) -> Result<f32, QuantizeError> {
        let score = self.inner_product_query(query, encoded)?;
        Ok(score)
    }

    fn score(&self, a: &[u8], b: &[u8]) -> Result<f32, QuantizeError>;

    fn bytes_per_vector(&self) -> usize;

    fn serialize(&self, writer: &mut dyn Write) -> Result<(), QuantizeError> {
        let _ = writer;
        Ok(())
    }
}

pub trait Dequantizer: Send + Sync {
    fn dequantize(&self, quantized: &[u8], out: &mut [f32]) -> Result<(), QuantizeError>;
}
