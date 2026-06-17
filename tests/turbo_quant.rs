use dendra::{Dequantizer, QuantizeError, Quantizer, TurboQuant, TurboQuantConfig, TurboQuantMode};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |acc, &x| x.mul_add(x, acc)).sqrt()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n = l2_norm(v);
    if n == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / n).collect()
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).fold(0.0f32, |acc, (&x, &y)| {
        let d = x - y;
        d.mul_add(d, acc)
    }) / a.len() as f32
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |acc, (&x, &y)| x.mul_add(y, acc))
}

#[test]
fn construction_and_validation() {
    let ok = TurboQuantConfig {
        dim: 64,
        bit_width: 3,
        mode: TurboQuantMode::Mse,
        seed: Some(42),
    };
    assert!(TurboQuant::new(ok).is_ok());

    let bad_bw = TurboQuantConfig {
        dim: 64,
        bit_width: 0,
        mode: TurboQuantMode::Mse,
        seed: None,
    };
    assert!(TurboQuant::new(bad_bw).is_err());

    let bad_ip = TurboQuantConfig {
        dim: 64,
        bit_width: 1,
        mode: TurboQuantMode::InnerProduct,
        seed: None,
    };
    assert!(TurboQuant::new(bad_ip).is_err());
}

#[test]
fn deterministic_with_seed() {
    let cfg = TurboQuantConfig {
        dim: 32,
        bit_width: 3,
        mode: TurboQuantMode::Mse,
        seed: Some(99),
    };

    let q1 = TurboQuant::new(cfg.clone()).unwrap();
    let q2 = TurboQuant::new(cfg).unwrap();

    let v = random_vectors(1, 32, 7).pop().unwrap();
    let mut b1 = vec![0u8; Quantizer::bytes_per_vector(&q1)];
    let mut b2 = vec![0u8; Quantizer::bytes_per_vector(&q2)];

    q1.quantize(&v, &mut b1).unwrap();
    q2.quantize(&v, &mut b2).unwrap();

    assert_eq!(b1, b2);
}

#[test]
fn mse_decreases_with_bit_width() {
    let dim = 96;
    let vectors = random_vectors(128, dim, 123)
        .into_iter()
        .map(|v| normalize(&v))
        .collect::<Vec<_>>();

    let mut mses = Vec::new();
    for bw in [1u8, 2, 3, 4] {
        let q = TurboQuant::new(TurboQuantConfig {
            dim,
            bit_width: bw,
            mode: TurboQuantMode::Mse,
            seed: Some(42),
        })
        .unwrap();

        let mut total = 0.0f32;
        for v in &vectors {
            let mut enc = vec![0u8; Quantizer::bytes_per_vector(&q)];
            let mut rec = vec![0.0f32; dim];
            q.quantize(v, &mut enc).unwrap();
            q.dequantize(&enc, &mut rec).unwrap();
            total += mse(v, &rec);
        }
        mses.push(total / vectors.len() as f32);
    }

    for i in 0..(mses.len() - 1) {
        assert!(
            mses[i + 1] <= mses[i] * 1.03,
            "mse should generally drop with more bits: {:?}",
            mses
        );
    }
}

#[test]
fn inner_product_mode_scores_and_query_path() {
    let dim = 64;
    let q = TurboQuant::new(TurboQuantConfig {
        dim,
        bit_width: 3,
        mode: TurboQuantMode::InnerProduct,
        seed: Some(123),
    })
    .unwrap();

    let vectors = random_vectors(32, dim, 999);
    let query = random_vectors(1, dim, 1000).pop().unwrap();

    let mut scores = Vec::new();
    for v in &vectors {
        let mut enc = vec![0u8; Quantizer::bytes_per_vector(&q)];
        q.quantize(v, &mut enc).unwrap();
        scores.push(q.inner_product_query(&query, &enc).unwrap());
    }

    assert_eq!(scores.len(), vectors.len());
    assert!(scores.iter().all(|s| s.is_finite()));

    let mut enc = vec![0u8; Quantizer::bytes_per_vector(&q)];
    q.quantize(&vectors[0], &mut enc).unwrap();
    let bad = q.inner_product_query(&query[..dim - 1], &enc);
    assert!(matches!(bad, Err(QuantizeError::DimensionMismatch { .. })));
}

#[test]
fn zero_norm_rejected() {
    let q = TurboQuant::new(TurboQuantConfig {
        dim: 16,
        bit_width: 2,
        mode: TurboQuantMode::Mse,
        seed: Some(1),
    })
    .unwrap();

    let v = vec![0.0f32; 16];
    let mut out = vec![0u8; Quantizer::bytes_per_vector(&q)];
    let err = q.quantize(&v, &mut out);
    assert!(matches!(err, Err(QuantizeError::ZeroNormVector)));
}

#[test]
fn query_estimator_reasonable_on_average() {
    let dim = 96;
    let query = random_vectors(1, dim, 14).pop().unwrap();

    let mut true_sum = 0.0f32;
    let mut est_sum = 0.0f32;
    let n = 64usize;

    for seed in 0..n {
        let tq = TurboQuant::new(TurboQuantConfig {
            dim,
            bit_width: 3,
            mode: TurboQuantMode::InnerProduct,
            seed: Some(seed as u64),
        })
        .unwrap();

        let x = normalize(&random_vectors(1, dim, 1000 + seed as u64).pop().unwrap());
        let mut enc = vec![0u8; Quantizer::bytes_per_vector(&tq)];
        tq.quantize(&x, &mut enc).unwrap();

        true_sum += dot(&query, &x);
        est_sum += tq.inner_product_query(&query, &enc).unwrap();
    }

    let true_avg = true_sum / n as f32;
    let est_avg = est_sum / n as f32;
    assert!((true_avg - est_avg).abs() < 0.25);
}

#[test]
fn quantize_output_size_matches_bytes_per_vector() {
    let dim = 64;
    for bw in [1u8, 2, 3, 4] {
        let q = TurboQuant::new(TurboQuantConfig {
            dim,
            bit_width: bw,
            mode: TurboQuantMode::Mse,
            seed: Some(7),
        })
        .unwrap();

        let v = normalize(&random_vectors(1, dim, 99 + bw as u64).pop().unwrap());
        let mut enc = vec![0u8; Quantizer::bytes_per_vector(&q)];
        q.quantize(&v, &mut enc).unwrap();
        assert_eq!(enc.len(), Quantizer::bytes_per_vector(&q));
    }
}

#[test]
fn quantize_rejects_dimension_mismatch() {
    let q = TurboQuant::new(TurboQuantConfig {
        dim: 32,
        bit_width: 2,
        mode: TurboQuantMode::Mse,
        seed: Some(1),
    })
    .unwrap();

    let bad = vec![0.0f32; 31];
    let mut out = vec![0u8; Quantizer::bytes_per_vector(&q)];
    let err = q.quantize(&bad, &mut out);
    assert!(matches!(err, Err(QuantizeError::DimensionMismatch { .. })));
}

#[test]
fn dequantize_requires_full_encoded_buffer() {
    let q = TurboQuant::new(TurboQuantConfig {
        dim: 32,
        bit_width: 3,
        mode: TurboQuantMode::InnerProduct,
        seed: Some(5),
    })
    .unwrap();

    let good = normalize(&random_vectors(1, 32, 10).pop().unwrap());
    let mut enc = vec![0u8; Quantizer::bytes_per_vector(&q)];
    q.quantize(&good, &mut enc).unwrap();

    let mut out = vec![0.0f32; 32];
    let err = q.dequantize(&enc[..enc.len() - 1], &mut out);
    assert!(matches!(err, Err(QuantizeError::BufferTooSmall { .. })));
}

#[test]
fn score_is_finite_and_symmetric() {
    let q = TurboQuant::new(TurboQuantConfig {
        dim: 96,
        bit_width: 3,
        mode: TurboQuantMode::InnerProduct,
        seed: Some(42),
    })
    .unwrap();

    let a = normalize(&random_vectors(1, 96, 100).pop().unwrap());
    let b = normalize(&random_vectors(1, 96, 200).pop().unwrap());
    let mut ea = vec![0u8; Quantizer::bytes_per_vector(&q)];
    let mut eb = vec![0u8; Quantizer::bytes_per_vector(&q)];
    q.quantize(&a, &mut ea).unwrap();
    q.quantize(&b, &mut eb).unwrap();

    let ab = q.score(&ea, &eb).unwrap();
    let ba = q.score(&eb, &ea).unwrap();
    assert!(ab.is_finite() && ba.is_finite());
    assert!((ab - ba).abs() < 1e-5);
}

#[test]
fn score_rejects_short_buffers() {
    let q = TurboQuant::new(TurboQuantConfig {
        dim: 48,
        bit_width: 2,
        mode: TurboQuantMode::Mse,
        seed: Some(3),
    })
    .unwrap();

    let v = normalize(&random_vectors(1, 48, 1234).pop().unwrap());
    let mut enc = vec![0u8; Quantizer::bytes_per_vector(&q)];
    q.quantize(&v, &mut enc).unwrap();

    let err = q.score(&enc[..enc.len() - 1], &enc);
    assert!(matches!(err, Err(QuantizeError::BufferTooSmall { .. })));
}
