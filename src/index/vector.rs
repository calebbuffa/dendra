use crate::core::{RoutingPolicyType, Segment, SegmentQueryContext, SegmentSummary};
use crate::err::DendraError;
use crate::math;
use crate::query::Query;
use std::collections::BinaryHeap;

const ROUTER_SOFTMAX_TEMPERATURE: f32 = 0.12;
const ROUTER_SEGMENT_ENTROPY_WEIGHT: f32 = 0.15;

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

impl Default for VectorIndex {
    fn default() -> Self {
        Self
    }
}

impl VectorIndex {
    pub fn new() -> Self {
        Self
    }

    pub(crate) fn query(
        &self,
        segments: &[Segment],
        summaries: &[SegmentSummary],
        routing_policy: &RoutingPolicyType,
        query: &Query,
        results: &mut Vec<(u32, f32)>,
    ) -> Result<bool, DendraError> {
        let candidates_per_segment = std::cmp::max(10, query.k * 2);
        let mut context = SegmentQueryContext::new(candidates_per_segment);

        let selected_indices =
            select_segment_indices(summaries, query, routing_policy, segments.len());

        for idx in selected_indices {
            let segment = &segments[idx];
            context.candidates.clear();
            context.queue.clear();
            segment.query(query, candidates_per_segment, query.metric, &mut context)?;
            summaries[idx].record_visit();
        }

        let k = query.k;
        if context.best_map.is_empty() {
            return Ok(false);
        }

        let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(k + 1);
        for (id, distance) in context.best_map.into_iter() {
            if heap.len() < k {
                heap.push(Scored { id, distance });
            } else if let Some(top) = heap.peek()
                && distance < top.distance
            {
                heap.pop();
                heap.push(Scored { id, distance });
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

fn select_segment_indices(
    summaries: &[SegmentSummary],
    query: &Query,
    routing_policy: &RoutingPolicyType,
    segment_count: usize,
) -> Vec<usize> {
    if segment_count == 0 {
        return Vec::new();
    }

    if summaries.len() != segment_count {
        return (0..segment_count).collect();
    }

    match routing_policy {
        RoutingPolicyType::Disabled => (0..segment_count).collect(),
        RoutingPolicyType::FlatTopK {
            max_segments,
            min_segments,
        } => {
            let max_keep = (*max_segments).max(*min_segments).min(segment_count).max(1);
            let min_keep = (*min_segments).min(max_keep);

            let mut sims = Vec::with_capacity(segment_count);
            let mut proximity = Vec::with_capacity(segment_count);
            let mut segment_entropy = Vec::with_capacity(segment_count);

            for summary in summaries.iter() {
                sims.push(math::cosine_similarity01(&query.vector, &summary.centroid));
                let center_dist2 = math::l2_distance_sq(&query.vector, &summary.centroid);
                proximity.push((center_dist2 - summary.r2).max(0.0));
                segment_entropy.push(summary.entropy.clamp(0.0, 1.0));
            }

            let mut probs = vec![0.0f32; sims.len()];
            math::softmax_probs(&sims, ROUTER_SOFTMAX_TEMPERATURE, &mut probs);
            let query_entropy = math::normalized_entropy(&probs);

            let span = max_keep.saturating_sub(min_keep);
            let dynamic_keep = min_keep + ((query_entropy * span as f32).round() as usize);
            let keep = dynamic_keep.clamp(min_keep, max_keep);

            let mut ranked: Vec<(usize, f32)> = (0..segment_count)
                .map(|idx| {
                    // Lower score is better: confidence + geometry + segment heterogeneity prior.
                    let confidence_penalty = -(probs[idx].max(1e-8)).ln();
                    let score = confidence_penalty
                        + proximity[idx]
                        + ROUTER_SEGMENT_ENTROPY_WEIGHT * segment_entropy[idx];
                    (idx, score)
                })
                .collect();

            ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
            ranked.into_iter().take(keep).map(|(idx, _)| idx).collect()
        }
    }
}
