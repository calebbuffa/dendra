use crate::DendraError;
use std::io::Read;

pub(crate) fn read_u8_le<R: Read>(r: &mut R) -> Result<u8, DendraError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub(crate) fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, DendraError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub(crate) fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, DendraError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub(crate) fn read_f32_le<R: Read>(r: &mut R) -> Result<f32, DendraError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
