use crate::core::{Segment, SegmentQueryContext};
use crate::err::DendraError;
use crate::query::Query;
use std::collections::BinaryHeap;

#[derive(Debug)]
struct Scored {
    id: u32,
    distance: f32,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.distance.to_bits() == other.distance.to_bits()
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.distance.total_cmp(&other.distance) {
            std::cmp::Ordering::Equal => self.id.cmp(&other.id),
            ord => ord,
        }
    }
}

pub struct VectorIndex;

impl VectorIndex {
    pub fn new() -> Self {
        Self
    }

    pub fn query(
        &self,
        segments: &[Segment],
        query: &Query,
        results: &mut Vec<(u32, f32)>,
    ) -> Result<bool, DendraError> {
        let candidates_per_segment = std::cmp::max(10, query.k * 2);
        let mut context = SegmentQueryContext::new(candidates_per_segment);

        for segment in segments {
            context.candidates.clear();
            context.queue.clear();
            segment.query(query, candidates_per_segment, query.metric, &mut context)?;
        }

        let k = query.k;
        if context.best_map.is_empty() {
            return Ok(false);
        }

        let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(k + 1);
        for (id, distance) in context.best_map.into_iter() {
            if heap.len() < k {
                heap.push(Scored { id, distance });
            } else if let Some(top) = heap.peek() {
                if distance < top.distance {
                    heap.pop();
                    heap.push(Scored { id, distance });
                }
            }
        }

        results.clear();
        while let Some(s) = heap.pop() {
            results.push((s.id, s.distance));
        }
        results.reverse();
        Ok(true)
    }
}
