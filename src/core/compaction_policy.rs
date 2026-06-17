use crate::core::SegmentSummary;
use crate::math;

/// Compaction cost estimate.
#[derive(Debug, Clone)]
pub struct MergeCostEstimate {
    pub segment_reduction: usize,
    pub overlap_reduction: f32,
    pub rebuild_cost_ms: u64,
    pub worth_it: bool,
}

/// Trait for pluggable compaction strategies.
pub trait CompactionPolicy: Send + 'static {
    /// Score a segment for compaction urgency (higher = more urgent).
    /// Range: [0.0, 1.0] where 1.0 is "must merge now".
    fn score_segment(&self, stats: &SegmentSummary, all_stats: &[SegmentSummary]) -> f32;

    /// Select which segments to merge.
    /// Returns Some(vec of segment IDs) if a merge should happen, None otherwise.
    fn select_merge_candidates(&self, all_stats: &[SegmentSummary]) -> Option<Vec<u64>>;
}

/// Size-tiered compaction: merge when too many small segments.
/// This is the simplest and most predictable strategy.
pub struct SizeTieredPolicy {
    pub max_segments_per_tier: usize,
    pub tier_size_ratio: usize, // L1 is 10x L0, L2 is 100x L0, etc.
    pub min_segment_count: usize,
}

impl SizeTieredPolicy {
    pub fn new(
        max_segments_per_tier: usize,
        tier_size_ratio: usize,
        min_segment_count: usize,
    ) -> Self {
        Self {
            max_segments_per_tier,
            tier_size_ratio,
            min_segment_count,
        }
    }
}

impl Default for SizeTieredPolicy {
    fn default() -> Self {
        Self {
            max_segments_per_tier: 4,
            tier_size_ratio: 10,
            min_segment_count: 2,
        }
    }
}

impl CompactionPolicy for SizeTieredPolicy {
    fn score_segment(&self, _stats: &SegmentSummary, all_stats: &[SegmentSummary]) -> f32 {
        if all_stats.len() < self.min_segment_count {
            return 0.0;
        }

        // Score based on how many small segments exist
        let small_count = all_stats.iter().filter(|s| s.size_mb() < 50.0).count();

        let pressure = (small_count as f32) / (self.max_segments_per_tier as f32);
        (pressure - 1.0).max(0.0).min(1.0)
    }

    fn select_merge_candidates(&self, all_stats: &[SegmentSummary]) -> Option<Vec<u64>> {
        if all_stats.len() < self.min_segment_count {
            return None;
        }

        // Find the smallest segments
        let mut sorted = all_stats.to_vec();
        sorted.sort_by(|a, b| a.size_bytes().cmp(&b.size_bytes()));

        let to_merge = (self.max_segments_per_tier).min(sorted.len() / 2).max(2);
        let candidates: Vec<u64> = sorted[..to_merge].iter().map(|s| s.segment_id).collect();

        if candidates.len() >= self.min_segment_count {
            return Some(candidates);
        }

        None
    }
}

/// Similarity-aware compaction: merge overlapping segments.
/// Reduces traversal redundancy by consolidating similar vectors.
pub struct SimilarityAwarePolicy {
    // Max cosine distance allowed for merge; lower means stricter merges.
    pub overlap_threshold: f32,
    // Do not merge segments above this heterogeneity level.
    pub max_merge_entropy: f32,
}

impl SimilarityAwarePolicy {
    pub fn new(overlap_threshold: f32) -> Self {
        Self {
            overlap_threshold,
            max_merge_entropy: 0.6,
        }
    }
}

impl Default for SimilarityAwarePolicy {
    fn default() -> Self {
        Self {
            overlap_threshold: 0.1,
            max_merge_entropy: 0.6,
        }
    }
}

impl CompactionPolicy for SimilarityAwarePolicy {
    fn score_segment(&self, stats: &SegmentSummary, all_stats: &[SegmentSummary]) -> f32 {
        if all_stats.len() < 2 {
            return 0.0;
        }

        // Score based on cosine distance to other segment centroids.
        let avg_overlap = all_stats
            .iter()
            .filter(|s| s.segment_id != stats.segment_id)
            .map(|s| centroid_cosine_distance(&stats.centroid, &s.centroid))
            .sum::<f32>()
            / (all_stats.len() as f32 - 1.0);

        // High-entropy segments are risky merge targets: keep urgency low.
        let entropy_factor = (1.0 - stats.entropy).clamp(0.0, 1.0);

        // Lower distance than threshold means stronger overlap => higher urgency.
        if self.overlap_threshold <= 0.0 {
            return 0.0;
        }
        (((self.overlap_threshold - avg_overlap) / self.overlap_threshold) * entropy_factor)
            .max(0.0)
            .min(1.0)
    }

    fn select_merge_candidates(&self, all_stats: &[SegmentSummary]) -> Option<Vec<u64>> {
        if all_stats.len() < 2 {
            return None;
        }

        // Find the most similar low-entropy pair.
        let mut best_pair = None;
        let mut best_distance = f32::INFINITY;

        for i in 0..all_stats.len() {
            for j in (i + 1)..all_stats.len() {
                if all_stats[i].entropy > self.max_merge_entropy
                    || all_stats[j].entropy > self.max_merge_entropy
                {
                    continue;
                }
                let dist = centroid_cosine_distance(&all_stats[i].centroid, &all_stats[j].centroid);
                if dist < best_distance {
                    best_distance = dist;
                    best_pair = Some((i, j));
                }
            }
        }

        if let Some((i, j)) = best_pair {
            if best_distance < self.overlap_threshold {
                return Some(vec![all_stats[i].segment_id, all_stats[j].segment_id]);
            }
        }

        None
    }
}

fn centroid_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }
    let na = math::l2_norm(a);
    let nb = math::l2_norm(b);
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    let cosine = (math::dot(a, b) / (na * nb)).clamp(-1.0, 1.0);
    1.0 - cosine
}

/// Query-driven compaction: merge based on co-access patterns.
/// Tracks which segments are frequently queried together.
pub struct QueryDrivenPolicy {
    pub co_access_threshold: u64, // merge if both queried together >N times
    pub min_segments_to_keep: usize,
}

impl QueryDrivenPolicy {
    pub fn new(co_access_threshold: u64, min_segments_to_keep: usize) -> Self {
        Self {
            co_access_threshold,
            min_segments_to_keep,
        }
    }
}

impl Default for QueryDrivenPolicy {
    fn default() -> Self {
        Self {
            co_access_threshold: 100,
            min_segments_to_keep: 1,
        }
    }
}

impl CompactionPolicy for QueryDrivenPolicy {
    fn score_segment(&self, stats: &SegmentSummary, all_stats: &[SegmentSummary]) -> f32 {
        if all_stats.len() <= self.min_segments_to_keep {
            return 0.0;
        }

        // Score based on visit frequency relative to others
        let avg_visits = all_stats
            .iter()
            .map(|s| s.visit_count() as f32)
            .sum::<f32>()
            / (all_stats.len() as f32);

        let my_visits = stats.visit_count() as f32;

        // Segments with below-average visits are candidates for merging
        if my_visits < avg_visits {
            ((avg_visits - my_visits) / avg_visits).min(1.0)
        } else {
            0.0
        }
    }

    fn select_merge_candidates(&self, all_stats: &[SegmentSummary]) -> Option<Vec<u64>> {
        if all_stats.len() <= self.min_segments_to_keep {
            return None;
        }

        // Find the least-visited segment and merge with its closest neighbor
        let mut least_visited_idx = 0;
        let mut min_visits = u64::MAX;

        for (i, stats) in all_stats.iter().enumerate() {
            if stats.visit_count() < min_visits {
                min_visits = stats.visit_count();
                least_visited_idx = i;
            }
        }

        // Find closest neighbor (most similar centroid)
        let least_visited = &all_stats[least_visited_idx];
        let mut closest_idx = 0;
        let mut closest_distance = f32::INFINITY;

        for (i, stats) in all_stats.iter().enumerate() {
            if i == least_visited_idx {
                continue;
            }
            let dist = math::l2_distance_sq(&least_visited.centroid, &stats.centroid);
            if dist < closest_distance {
                closest_distance = dist;
                closest_idx = i;
            }
        }

        Some(vec![
            all_stats[least_visited_idx].segment_id,
            all_stats[closest_idx].segment_id,
        ])
    }
}

/// Estimate the cost of merging specific segments.
pub fn estimate_merge_cost(
    segments_to_merge: &[SegmentSummary],
    all_segments: &[SegmentSummary],
) -> MergeCostEstimate {
    let before_count = all_segments.len();
    let after_count = before_count - segments_to_merge.len() + 1;
    let segment_reduction = before_count - after_count;

    // Estimate overlap reduction (simplified)
    let avg_overlap_before = if before_count > 1 {
        (0..segments_to_merge.len())
            .map(|i| {
                all_segments
                    .iter()
                    .filter(|s| {
                        !segments_to_merge
                            .iter()
                            .any(|m| m.segment_id == s.segment_id)
                    })
                    .map(|s| math::l2_distance_sq(&segments_to_merge[i].centroid, &s.centroid))
                    .sum::<f32>()
                    / (all_segments.len() as f32)
            })
            .sum::<f32>()
            / segments_to_merge.len() as f32
    } else {
        0.0
    };

    let overlap_reduction = (1.0f32 - avg_overlap_before).max(0.0f32);

    // Estimate rebuild cost (rough: 1ms per 10M vectors)
    let total_vectors: usize = segments_to_merge.iter().map(|s| s.vector_count).sum();
    let rebuild_cost_ms = ((total_vectors / 10_000_000) as u64).max(10);

    let worth_it = segment_reduction > 0 && overlap_reduction > 0.05;

    MergeCostEstimate {
        segment_reduction,
        overlap_reduction,
        rebuild_cost_ms,
        worth_it,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(id: u64, count: usize, dim: usize, radius: f32) -> SegmentSummary {
        SegmentSummary::new(id, count, dim, vec![0.5; dim], radius, 0.2)
    }

    #[test]
    fn test_size_tiered_policy() {
        let policy = SizeTieredPolicy::default();
        let stats = vec![
            make_stats(1, 100, 128, 0.5),
            make_stats(2, 100, 128, 0.5),
            make_stats(3, 100, 128, 0.5),
            make_stats(4, 100, 128, 0.5),
            make_stats(5, 1000, 128, 1.0),
        ];

        let candidates = policy.select_merge_candidates(&stats);
        assert!(candidates.is_some());
    }

    #[test]
    fn test_similarity_aware_policy() {
        let policy = SimilarityAwarePolicy::default();
        let stats = vec![make_stats(1, 100, 128, 0.5), make_stats(2, 100, 128, 0.5)];

        let candidates = policy.select_merge_candidates(&stats);
        assert!(candidates.is_some());
    }
}
