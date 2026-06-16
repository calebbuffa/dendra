use crate::FvdbError;
use std::io::{Read, Write};

pub const SEGMENT_MAGIC: [u8; 4] = *b"SEGM";
pub const SEGMENT_FORMAT_VERSION: u8 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SegmentHeader {
    pub magic: [u8; 4],
    pub format_version: u8,
    pub flags: u8,
    pub reserved0: u16,
    pub dim: u32,
    pub count: u64,
    pub vectors_bytes: u64,
    pub ids_bytes: u64,
    pub index_bytes: u64,
}

impl SegmentHeader {
    pub fn new(
        dim: usize,
        count: usize,
        vectors_bytes: u64,
        ids_bytes: u64,
        index_bytes: u64,
    ) -> Self {
        Self {
            magic: SEGMENT_MAGIC,
            format_version: SEGMENT_FORMAT_VERSION,
            flags: 0,
            reserved0: 0,
            dim: dim as u32,
            count: count as u64,
            vectors_bytes,
            ids_bytes,
            index_bytes,
        }
    }
}

pub fn write_segment_header<W: Write>(w: &mut W, h: &SegmentHeader) -> Result<(), FvdbError> {
    w.write_all(&h.magic)?;
    w.write_all(&[h.format_version])?;
    w.write_all(&[h.flags])?;
    w.write_all(&h.reserved0.to_le_bytes())?;
    w.write_all(&h.dim.to_le_bytes())?;
    w.write_all(&h.count.to_le_bytes())?;
    w.write_all(&h.vectors_bytes.to_le_bytes())?;
    w.write_all(&h.ids_bytes.to_le_bytes())?;
    w.write_all(&h.index_bytes.to_le_bytes())?;
    Ok(())
}

pub fn read_segment_header<R: Read>(r: &mut R) -> Result<SegmentHeader, FvdbError> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;

    let mut b1 = [0u8; 1];
    r.read_exact(&mut b1)?;
    let format_version = b1[0];

    r.read_exact(&mut b1)?;
    let flags = b1[0];

    let mut b2 = [0u8; 2];
    r.read_exact(&mut b2)?;
    let reserved0 = u16::from_le_bytes(b2);

    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4)?;
    let dim = u32::from_le_bytes(b4);

    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8)?;
    let count = u64::from_le_bytes(b8);

    r.read_exact(&mut b8)?;
    let vectors_bytes = u64::from_le_bytes(b8);

    r.read_exact(&mut b8)?;
    let ids_bytes = u64::from_le_bytes(b8);

    r.read_exact(&mut b8)?;
    let index_bytes = u64::from_le_bytes(b8);

    Ok(SegmentHeader {
        magic,
        format_version,
        flags,
        reserved0,
        dim,
        count,
        vectors_bytes,
        ids_bytes,
        index_bytes,
    })
}

pub(crate) fn read_u8_le<R: Read>(r: &mut R) -> Result<u8, FvdbError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub(crate) fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, FvdbError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub(crate) fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, FvdbError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub(crate) fn read_f32_le<R: Read>(r: &mut R) -> Result<f32, FvdbError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
