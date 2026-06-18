use crate::quantization::QuantizeError;
use faer::{Mat, Stride};
use log::debug;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use wide::f32x8;

#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut total = 0.0f32;
    for i in 0..n {
        total = a[i].mul_add(b[i], total);
    }
    total
}

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

#[inline]
pub fn dot_scaled(a: &[f32], b: &[f32], scale: f32) -> f32 {
    scale * dot(a, b)
}

#[inline]
pub fn dot_packed_signs(projected: &[f32], signs: &[u8], len: usize) -> f32 {
    let n = len.min(projected.len());
    let mut total = 0.0f32;
    let chunks = n / 8;
    let remainder = n % 8;

    for (chunk, sign_byte) in signs.iter().enumerate().take(chunks) {
        let base = chunk * 8;
        let p0 = projected[base];
        let p1 = projected[base + 1];
        let p2 = projected[base + 2];
        let p3 = projected[base + 3];
        let p4 = projected[base + 4];
        let p5 = projected[base + 5];
        let p6 = projected[base + 6];
        let p7 = projected[base + 7];

        let s0 = 1.0f32 - ((*sign_byte) & 1) as f32 * 2.0;
        let s1 = 1.0f32 - (((*sign_byte) >> 1) & 1) as f32 * 2.0;
        let s2 = 1.0f32 - (((*sign_byte) >> 2) & 1) as f32 * 2.0;
        let s3 = 1.0f32 - (((*sign_byte) >> 3) & 1) as f32 * 2.0;
        let s4 = 1.0f32 - (((*sign_byte) >> 4) & 1) as f32 * 2.0;
        let s5 = 1.0f32 - (((*sign_byte) >> 5) & 1) as f32 * 2.0;
        let s6 = 1.0f32 - (((*sign_byte) >> 6) & 1) as f32 * 2.0;
        let s7 = 1.0f32 - (((*sign_byte) >> 7) & 1) as f32 * 2.0;

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
pub fn l2_norm_sq(a: &[f32]) -> f32 {
    dot(a, a)
}

#[inline]
pub fn l2_norm(a: &[f32]) -> f32 {
    l2_norm_sq(a).sqrt()
}

pub fn batch_l2_norms(vectors: &[f32], dim: usize) -> Vec<f32> {
    let n = vectors.len() / dim;
    let mut norms = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * dim;
        norms.push(l2_norm(&vectors[start..start + dim]));
    }
    norms
}

pub fn batch_mat_t_mul(rotation: &Mat<f32>, vectors: &[f32], dim: usize) -> Vec<f32> {
    let start = std::time::Instant::now();
    let n = vectors.len() / dim;
    let mat_create = std::time::Instant::now();
    let vec_mat = unsafe {
        faer::MatRef::<f32>::from_raw_parts(vectors.as_ptr(), dim, n, 1isize, dim as isize)
    };
    let mat_time = mat_create.elapsed().as_secs_f64() * 1000.0;

    let mul_start = std::time::Instant::now();
    let result = rotation.transpose() * vec_mat;
    let mul_time = mul_start.elapsed().as_secs_f64() * 1000.0;

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

pub fn batch_mat_mul(rotation: &Mat<f32>, vectors: &[f32], dim: usize) -> Vec<f32> {
    let start = std::time::Instant::now();
    let n = vectors.len() / dim;
    let mat_create = std::time::Instant::now();
    let vec_mat = unsafe {
        faer::MatRef::<f32>::from_raw_parts(vectors.as_ptr(), dim, n, 1isize, dim as isize)
    };
    let mat_time = mat_create.elapsed().as_secs_f64() * 1000.0;

    let mul_start = std::time::Instant::now();
    let result = rotation * vec_mat;
    let mul_time = mul_start.elapsed().as_secs_f64() * 1000.0;

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

pub fn normalize(a: &mut [f32]) -> Result<(), QuantizeError> {
    let norm = l2_norm(a);
    if norm == 0.0 {
        return Err(QuantizeError::ZeroNormVector);
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
