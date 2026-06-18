use crate::err::EngramError;

/// A distance function that takes two vectors and returns a distance score.
/// Lower is better.
pub type MetricFn = fn(&[f32], &[f32]) -> Result<f32, EngramError>;

pub fn l2_distance(a: &[f32], b: &[f32]) -> Result<f32, EngramError> {
    Ok(l2_distance_sq(a, b).sqrt())
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32, EngramError> {
    let sim = cosine_similarity(a, b)?;
    Ok(1.0 - sim)
}

/// Dot product for f32 slices.
/// Kept intentionally simple to let LLVM auto-vectorize for each target.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut total = 0.0f32;
    for i in 0..n {
        total = a[i].mul_add(b[i], total);
    }
    total
}

/// Sparse weighted dot product with an additive bias term.
#[inline(always)]
pub fn sparse_weighted_dot(vector: &[f32], indices: &[usize], weights: &[f32], bias: f32) -> f32 {
    let n = indices.len().min(weights.len());
    let mut total = bias;
    for i in 0..n {
        total = weights[i].mul_add(vector[indices[i]], total);
    }
    total
}

#[inline]
fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        let diff = a[i] - b[i];
        sum = diff.mul_add(diff, sum);
    }
    sum
}

/// L2 distance over only the first `prefix_len` entries.
#[inline]
pub fn l2_distance_sq_prefix(a: &[f32], b: &[f32], prefix_len: usize) -> f32 {
    let n = a.len().min(b.len()).min(prefix_len);
    let mut sum = 0.0f32;
    for i in 0..n {
        let diff = a[i] - b[i];
        sum = diff.mul_add(diff, sum);
    }
    sum
}

/// Asymmetric-distance-computation (ADC) squared L2 distance for SQ8 rows.
#[inline]
pub fn adc_l2_sq(query: &[f32], code: &[u8], min_vals: &[f32], max_vals: &[f32]) -> f32 {
    let n = query
        .len()
        .min(code.len())
        .min(min_vals.len())
        .min(max_vals.len());
    let mut total = 0.0f32;
    for j in 0..n {
        let min_v = min_vals[j];
        let max_v = max_vals[j];
        let alpha = (max_v - min_v) / 255.0;
        let q_adapt = (query[j] - min_v) / alpha.max(1e-12);
        let diff = q_adapt - code[j] as f32;
        let scaled = alpha * diff;
        total = scaled.mul_add(scaled, total);
    }
    total
}

#[inline]
fn l2_norm_sq(a: &[f32]) -> f32 {
    dot(a, a)
}

#[inline]
fn l2_norm(a: &[f32]) -> f32 {
    l2_norm_sq(a).sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, EngramError> {
    let dp = dot(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(EngramError::ZeroNormVector);
    }
    Ok(dp / (norm_a * norm_b))
}
