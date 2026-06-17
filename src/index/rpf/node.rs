use log::debug;
use std::io::{Read, Write};

use crate::{
    DendraError,
    io::{read_f32_le, read_u8_le, read_u32_le},
};

pub struct Node {
    pub left: u32,
    pub right: u32,

    /// Projection vector for the hyperplane
    pub projection: Vec<f32>,
    /// Threshold for the hyperplane (dot product < threshold goes left, >= goes right)
    pub threshold: f32,

    pub is_leaf: bool,

    /// Inclusive offset into the contiguous ids buffer
    pub start: usize,
    /// Exclusive offset into the contiguous ids buffer
    pub end: usize,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            left: 0,
            right: 0,
            projection: Vec::new(),
            threshold: 0.0,
            is_leaf: false,
            start: 0,
            end: 0,
        }
    }
}

impl Node {
    pub fn leaf(start: usize, end: usize) -> Node {
        let mut node = Node::default();
        node.is_leaf = true;
        node.start = start;
        node.end = end;
        node
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<(), DendraError> {
        let _start = std::time::Instant::now();
        w.write_all(&self.left.to_le_bytes())?;
        w.write_all(&self.right.to_le_bytes())?;
        w.write_all(&(self.is_leaf as u8).to_le_bytes())?;
        w.write_all(&(self.start as u32).to_le_bytes())?;
        w.write_all(&(self.end as u32).to_le_bytes())?;
        w.write_all(&self.threshold.to_le_bytes())?;
        w.write_all(&(self.projection.len() as u32).to_le_bytes())?;
        let proj_start = std::time::Instant::now();
        for &p in self.projection.iter() {
            w.write_all(&p.to_le_bytes())?;
        }
        if self.projection.len() > 128 {
            debug!(
                "      node proj write: {} bytes in {:.2}µs",
                self.projection.len() * 4,
                proj_start.elapsed().as_secs_f64() * 1_000_000.0
            );
        }
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> Result<Self, DendraError> {
        let left = read_u32_le(r)?;
        let right = read_u32_le(r)?;
        let is_leaf = read_u8_le(r)? != 0;
        let start = read_u32_le(r)? as usize;
        let end = read_u32_le(r)? as usize;
        let threshold = read_f32_le(r)?;
        let proj_len = read_u32_le(r)? as usize;
        let mut projection = Vec::with_capacity(proj_len);
        for _ in 0..proj_len {
            let p = read_f32_le(r)?;
            projection.push(p);
        }
        Ok(Node {
            left,
            right,
            is_leaf,
            start,
            end,
            threshold,
            projection,
        })
    }
}
