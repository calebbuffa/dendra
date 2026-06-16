use crate::quantization::QuantizeError;

pub fn pack_indices(indices: &[u8], bit_width: u8) -> Result<Vec<u8>, QuantizeError> {
    if !(1..=8).contains(&bit_width) {
        return Err(QuantizeError::UnsupportedMethod(format!(
            "bit_width {} is not supported (expected 1..=8)",
            bit_width
        )));
    }

    if bit_width == 8 {
        return Ok(indices.to_vec());
    }

    let bits_per_value = bit_width as usize;
    let total_bits = indices.len() * bits_per_value;
    let out_len = total_bits.div_ceil(8);
    let mut out = vec![0u8; out_len];

    let mask = (1u16 << bit_width) - 1;
    for (i, &v) in indices.iter().enumerate() {
        if (v as u16) > mask {
            return Err(QuantizeError::InvalidEncoding);
        }

        let bit_offset = i * bits_per_value;
        for b in 0..bits_per_value {
            if ((v >> b) & 1) == 1 {
                let idx = bit_offset + b;
                out[idx / 8] |= 1u8 << (idx % 8);
            }
        }
    }

    Ok(out)
}

pub fn unpack_indices(
    packed: &[u8],
    bit_width: u8,
    n_values: usize,
) -> Result<Vec<u8>, QuantizeError> {
    if !(1..=8).contains(&bit_width) {
        return Err(QuantizeError::UnsupportedMethod(format!(
            "bit_width {} is not supported (expected 1..=8)",
            bit_width
        )));
    }

    if bit_width == 8 {
        if packed.len() < n_values {
            return Err(QuantizeError::BufferTooSmall {
                required: n_values,
                provided: packed.len(),
            });
        }
        return Ok(packed[..n_values].to_vec());
    }

    let bits_per_value = bit_width as usize;
    let total_bits = n_values * bits_per_value;
    let required_bytes = total_bits.div_ceil(8);
    if packed.len() < required_bytes {
        return Err(QuantizeError::BufferTooSmall {
            required: required_bytes,
            provided: packed.len(),
        });
    }

    let mut out = vec![0u8; n_values];
    for (i, slot) in out.iter_mut().enumerate() {
        let bit_offset = i * bits_per_value;
        let mut v = 0u8;
        for b in 0..bits_per_value {
            let idx = bit_offset + b;
            let bit = (packed[idx / 8] >> (idx % 8)) & 1;
            v |= bit << b;
        }
        *slot = v;
    }

    Ok(out)
}
