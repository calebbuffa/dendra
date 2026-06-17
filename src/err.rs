use crate::core::TaskSystemError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DendraError {
    #[error("Zero-norm vector")]
    ZeroNormVector,
    #[error("Degenerate split for a Random Projection Tree Node")]
    DegenerateSplit,
    #[error("Empty Random Projection Tree Node")]
    EmptyNode,
    #[error("Index {index} is out of bounds for length {length}")]
    IndexOutOfBounds { index: usize, length: usize },
    #[error("Invalid vector dimension, expected {expected}, received {received}")]
    InvalidVectorDimension { expected: usize, received: usize },
    #[error("Invalid tree index {index}, max is {max}")]
    InvalidTreeIndex { index: usize, max: usize },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid header: expected {expected}, received {received}")]
    InvalidHeader { expected: String, received: String },
    #[error("Unsupported version: expected {expected}, received {received}")]
    UnsupportedVersion { expected: String, received: String },
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    #[error("mmap failed: {0}")]
    MmapFailed(String),
    #[error("mmap size mismatch: expected {expected}, received {received}")]
    MmapSizeMismatch { expected: usize, received: usize },
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Task system error: {0}")]
    TaskSystem(#[from] TaskSystemError),
}
