use crate::core::TaskSystemError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngramError {
    #[error("Zero-norm vector")]
    ZeroNormVector,
    #[error("Index {index} is out of bounds for length {length}")]
    IndexOutOfBounds { index: usize, length: usize },
    #[error("Invalid vector dimension, expected {expected}, received {received}")]
    InvalidVectorDimension { expected: usize, received: usize },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid header: expected {expected}, received {received}")]
    InvalidHeader { expected: String, received: String },
    #[error("Unsupported version: expected {expected}, received {received}")]
    UnsupportedVersion { expected: String, received: String },
    #[error("Invariant violation: {0}")]
    InvariantViolation(String),
    #[error("Codec error: {0}")]
    Codec(String),
    #[error("Task system error: {0}")]
    TaskSystem(#[from] TaskSystemError),
}
