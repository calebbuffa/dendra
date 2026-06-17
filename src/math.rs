use crate::err::DendraError;
use faer::{Mat, Stride};
use log::debug;
use rand::Rng;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use wide::f32x8;

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

/// SIMD dot product between f32 query and u8 encoded data.
#[inline(always)]
pub fn dot_u8_f32(query: &[f32], encoded: &[u8]) -> f32 {
    let mut sum = f32x8::ZERO;
    let q_chunks = query.chunks_exact(8);

    for (i, q_chunk) in q_chunks.clone().enumerate() {
        let base = i * 8;
        let e_chunk = &encoded[base..base + 8];
        let vq = f32x8::from(q_chunk);
        let ve = f32x8::from([
            e_chunk[0] as f32,
            e_chunk[1] as f32,
            e_chunk[2] as f32,
            e_chunk[3] as f32,
            e_chunk[4] as f32,
            e_chunk[5] as f32,
            e_chunk[6] as f32,
            e_chunk[7] as f32,
        ]);
        sum = vq.mul_add(ve, sum);
    }

    let rem = q_chunks.remainder();
    let base = (query.len() / 8) * 8;
    let mut total: f32 = sum.reduce_add();
    for (i, &q) in rem.iter().enumerate() {
        total += q * encoded[base + i] as f32;
    }
    total
}

/// SIMD dot product between f32 vector and i8 vector.
#[inline(always)]
pub fn dot_signed(a: &[f32], b: &[i8]) -> f32 {
    let n = a.len().min(b.len());
    let a = &a[..n];
    let b = &b[..n];

    let mut sum = f32x8::ZERO;
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    for (ca, cb) in a_chunks.clone().zip(b_chunks.clone()) {
        let va = f32x8::from(ca);
        let vb = f32x8::from([
            cb[0] as f32,
            cb[1] as f32,
            cb[2] as f32,
            cb[3] as f32,
            cb[4] as f32,
            cb[5] as f32,
            cb[6] as f32,
            cb[7] as f32,
        ]);
        sum = va.mul_add(vb, sum);
    }

    let rem_a = a_chunks.remainder();
    let rem_b = b_chunks.remainder();
    let mut total: f32 = sum.reduce_add();
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        total = x.mul_add(y as f32, total);
    }
    total
}

/// SIMD dot product with per-element scaling.
#[inline]
pub fn dot_scaled(a: &[f32], b: &[f32], scale: f32) -> f32 {
    scale * dot(a, b)
}

/// Dot product between a float vector and packed sign bits (u8).
#[inline]
pub fn dot_packed_signs(projected: &[f32], signs: &[u8], len: usize) -> f32 {
    let n = len.min(projected.len());

    let mut total = 0.0f32;
    let chunks = n / 8;
    let remainder = n % 8;

    for chunk in 0..chunks {
        let base = chunk * 8;
        let sign_byte = signs[chunk];
        let p0 = projected[base];
        let p1 = projected[base + 1];
        let p2 = projected[base + 2];
        let p3 = projected[base + 3];
        let p4 = projected[base + 4];
        let p5 = projected[base + 5];
        let p6 = projected[base + 6];
        let p7 = projected[base + 7];

        let s0 = 1.0f32 - ((sign_byte >> 0) & 1) as f32 * 2.0;
        let s1 = 1.0f32 - ((sign_byte >> 1) & 1) as f32 * 2.0;
        let s2 = 1.0f32 - ((sign_byte >> 2) & 1) as f32 * 2.0;
        let s3 = 1.0f32 - ((sign_byte >> 3) & 1) as f32 * 2.0;
        let s4 = 1.0f32 - ((sign_byte >> 4) & 1) as f32 * 2.0;
        let s5 = 1.0f32 - ((sign_byte >> 5) & 1) as f32 * 2.0;
        let s6 = 1.0f32 - ((sign_byte >> 6) & 1) as f32 * 2.0;
        let s7 = 1.0f32 - ((sign_byte >> 7) & 1) as f32 * 2.0;

        total = p0.mul_add(s0, total);
        total = p1.mul_add(s1, total);
        total = p2.mul_add(s2, total);
        total = p3.mul_add(s3, total);
        total = p4.mul_add(s4, total);
        total = p5.mul_add(s5, total);
        total = p6.mul_add(s6, total);
        total = p7.mul_add(s7, total);
    }

    if remainder > 0 {
        let base = chunks * 8;
        for i in 0..remainder {
            let idx = base + i;
            let byte_index = idx / 8;
            let bit_index = idx % 8;
            let is_pos = ((signs[byte_index] >> bit_index) & 1) == 1;
            let s = if is_pos { 1.0 } else { -1.0 };
            total = projected[idx].mul_add(s, total);
        }
    }
    total
}

#[inline]
pub fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Compute L2 norm squared.
#[inline]
pub fn l2_norm_sq(a: &[f32]) -> f32 {
    dot(a, a)
}

/// Compute L2 norm.
#[inline]
pub fn l2_norm(a: &[f32]) -> f32 {
    l2_norm_sq(a).sqrt()
}

/// Batch compute L2 norms for a matrix of vectors stored as flat [f32].
/// `vectors` has shape (n_rows * dim).
pub fn batch_l2_norms(vectors: &[f32], dim: usize) -> Vec<f32> {
    let n = vectors.len() / dim;
    let mut norms = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * dim;
        norms.push(l2_norm(&vectors[start..start + dim]));
    }
    norms
}

/// Multiply `rotation^T * vectors` where vectors is flat [f32] of shape (n * dim).
/// Returns flat result of same layout.
/// Uses faer's native batched matrix multiplication for optimal performance.
pub fn batch_mat_t_mul(rotation: &Mat<f32>, vectors: &[f32], dim: usize) -> Vec<f32> {
    let start = std::time::Instant::now();
    let n = vectors.len() / dim;
    // Zero-copy input: vectors is row-major (n x dim).
    // That layout is byte-for-byte identical to faer column-major (dim x n):
    // column c = vector c = vectors[c*dim .. (c+1)*dim], all contiguous.
    let mat_create = std::time::Instant::now();
    let vec_mat = unsafe {
        faer::MatRef::<f32>::from_raw_parts(
            vectors.as_ptr(),
            dim,
            n,
            1isize,       // row_stride: rows contiguous within each column
            dim as isize, // col_stride: each vector starts dim elements apart
        )
    };
    let mat_time = mat_create.elapsed().as_secs_f64() * 1000.0;

    // rotation^T * vec_mat gives (dim x n) result
    let mul_start = std::time::Instant::now();
    let result = rotation.transpose() * vec_mat;
    let mul_time = mul_start.elapsed().as_secs_f64() * 1000.0;

    // faer stores (dim x n) column-major: col c = all dim coords of vector c.
    // If col_stride == dim, the whole matrix is one contiguous flat block already
    // in our desired layout — just memcpy it.
    let flat_start = std::time::Instant::now();
    let as_ref = result.as_ref();
    let col_stride = as_ref.col_stride().element_stride();
    let flat = if col_stride == dim as isize {
        let ptr = as_ref.ptr_at(0, 0);
        unsafe { std::slice::from_raw_parts(ptr, dim * n).to_vec() }
    } else {
        // Fallback: column-by-column copy (col_stride had alignment padding)
        let mut flat = vec![0.0f32; dim * n];
        for c in 0..n {
            let col = result.col(c);
            let col_start = c * dim;
            for r in 0..dim {
                flat[col_start + r] = col[r];
            }
        }
        flat
    };
    let flat_time = flat_start.elapsed().as_secs_f64() * 1000.0;

    if n > 10000 {
        debug!(
            "      batch_mat_t_mul({} x {}): mat_create={:.1}ms, mul={:.1}ms, flatten={:.1}ms (stride={}), TOTAL={:.1}ms",
            dim,
            n,
            mat_time,
            mul_time,
            flat_time,
            col_stride,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    flat
}

/// Multiply `rotation * vectors` where vectors is flat [f32] of shape (n * dim).
/// Returns flat result of same layout.
/// Uses faer's native batched matrix multiplication.
pub fn batch_mat_mul(rotation: &Mat<f32>, vectors: &[f32], dim: usize) -> Vec<f32> {
    let start = std::time::Instant::now();
    let n = vectors.len() / dim;
    // Zero-copy input: same layout as column-major (dim x n)
    let mat_create = std::time::Instant::now();
    let vec_mat = unsafe {
        faer::MatRef::<f32>::from_raw_parts(vectors.as_ptr(), dim, n, 1isize, dim as isize)
    };
    let mat_time = mat_create.elapsed().as_secs_f64() * 1000.0;

    // rotation * vec_mat gives (dim x n) result
    let mul_start = std::time::Instant::now();
    let result = rotation * vec_mat;
    let mul_time = mul_start.elapsed().as_secs_f64() * 1000.0;

    // Zero-copy flatten: faer (dim x n) column-major = our row-major output
    let flat_start = std::time::Instant::now();
    let as_ref = result.as_ref();
    let col_stride = as_ref.col_stride().element_stride();
    let flat = if col_stride == dim as isize {
        let ptr = as_ref.ptr_at(0, 0);
        unsafe { std::slice::from_raw_parts(ptr, dim * n).to_vec() }
    } else {
        let mut flat = vec![0.0f32; dim * n];
        for c in 0..n {
            let col = result.col(c);
            let col_start = c * dim;
            for r in 0..dim {
                flat[col_start + r] = col[r];
            }
        }
        flat
    };
    let flat_time = flat_start.elapsed().as_secs_f64() * 1000.0;

    if n > 10000 {
        debug!(
            "      batch_mat_mul({} x {}): mat_create={:.1}ms, mul={:.1}ms, flatten={:.1}ms (stride={}), TOTAL={:.1}ms",
            dim,
            n,
            mat_time,
            mul_time,
            flat_time,
            col_stride,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    flat
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, DendraError> {
    let dp = dot(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(DendraError::ZeroNormVector);
    }
    Ok(dp / (norm_a * norm_b))
}

/// Normalize vector in-place.
pub fn normalize(a: &mut [f32]) -> Result<(), DendraError> {
    let norm = l2_norm(a);
    if norm == 0.0 {
        return Err(DendraError::ZeroNormVector);
    }
    let inv_norm = 1.0 / norm;
    for x in a.iter_mut() {
        *x *= inv_norm;
    }
    Ok(())
}

pub fn mat_vec_mul(matrix: &Mat<f32>, vector: &[f32]) -> Vec<f32> {
    let x = faer::Col::from_fn(vector.len(), |i| vector[i]);
    let y = matrix * &x;
    (0..y.nrows()).map(|i| y[i]).collect()
}

pub fn mat_t_vec_mul(matrix: &Mat<f32>, vector: &[f32]) -> Vec<f32> {
    let x = faer::Col::from_fn(vector.len(), |i| vector[i]);
    let y = matrix.transpose() * &x;
    (0..y.nrows()).map(|i| y[i]).collect()
}

pub fn random_orthogonal_matrix(dim: usize, seed: Option<u64>) -> Mat<f32> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(0x5EED_BAAD_F00D_u64),
    };
    let m = Mat::from_fn(dim, dim, |_, _| {
        let v: f32 = StandardNormal.sample(&mut rng);
        v
    });
    m.qr().compute_Q()
}

pub fn random_projection_matrix(rows: usize, cols: usize, seed: Option<u64>) -> Mat<f32> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(0xA11C_E5EED_u64),
    };
    Mat::from_fn(rows, cols, |_, _| {
        let x: f32 = StandardNormal.sample(&mut rng);
        x
    })
}

pub fn random_standard_normal_vector(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim)
        .map(|_| {
            let x: f32 = StandardNormal.sample(rng);
            x
        })
        .collect()
}

pub fn random_unit_vector(dim: usize, rng: &mut impl Rng) -> Result<Vec<f32>, DendraError> {
    let mut v = random_standard_normal_vector(dim, rng);
    normalize(&mut v)?;
    Ok(v)
}

/// Integrate using the trapezoidal rule.
pub fn trapezoid(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 || y.len() < 2 || x.len() != y.len() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for i in 0..(x.len() - 1) {
        let dx = x[i + 1] - x[i];
        acc += 0.5 * (y[i] + y[i + 1]) * dx;
    }
    acc
}

pub fn beta_pdf(x: f64, dim: usize) -> f64 {
    if x.abs() >= 1.0 {
        return 0.0;
    }
    let dim_f = dim as f64;
    let log_coeff =
        ln_gamma(dim_f / 2.0) - 0.5 * std::f64::consts::PI.ln() - ln_gamma((dim_f - 1.0) / 2.0);
    let exponent = (dim_f - 3.0) / 2.0;
    (log_coeff + exponent * (1.0 - x * x).ln()).exp()
}

const LN_GAMMA_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    0.000_009_984_369_578_019_572,
    0.000_000_150_563_273_514_931_16,
];

pub fn ln_gamma(z: f64) -> f64 {
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = LN_GAMMA_COEFFS[0];
    for (i, coeff) in LN_GAMMA_COEFFS.iter().enumerate().skip(1) {
        x += coeff / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

pub fn softmax_probs(values: &[f32], temperature: f32, out: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    let inv_t = 1.0f32 / temperature.max(1e-4);
    let max_v = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut exps = Vec::with_capacity(values.len());
    let mut sum = 0.0f32;
    for &v in values {
        let e = ((v - max_v) * inv_t).exp();
        exps.push(e);
        sum += e;
    }

    if sum <= 0.0 || !sum.is_finite() {
        let uniform = 1.0f32 / values.len() as f32;
        for o in out.iter_mut() {
            *o = uniform;
        }
        return;
    }

    for (i, &e) in exps.iter().enumerate() {
        out[i] = e / sum;
    }
}

pub fn normalized_entropy(probs: &[f32]) -> f32 {
    if probs.len() <= 1 {
        return 0.0;
    }

    let mut h = 0.0f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    let h_max = (probs.len() as f32).ln().max(1e-8);
    (h / h_max).clamp(0.0, 1.0)
}

pub fn cosine_similarity01(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.5;
    }
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na <= 0.0 || nb <= 0.0 {
        return 0.5;
    }
    let cosine = (dot(a, b) / (na * nb)).clamp(-1.0, 1.0);
    0.5 * (cosine + 1.0)
}
