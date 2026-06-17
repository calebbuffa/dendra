use crate::{err::DendraError, math::cosine_similarity};

/// A distance function that takes two vectors and returns a similarity score.
/// "lower is better"
pub type MetricFn = fn(&[f32], &[f32]) -> Result<f32, DendraError>;

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    a[..n]
        .iter()
        .zip(&b[..n])
        .fold(0.0f32, |acc, (&x, &y)| {
            let d = x - y;
            d.mul_add(d, acc)
        })
        .sqrt()
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32, DendraError> {
    let sim = cosine_similarity(a, b)?;
    Ok(1.0 - sim)
}
