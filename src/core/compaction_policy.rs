use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Per-segment statistics for compaction decisions.
#[derive(Clone)]
pub struct SegmentStats {
    pub segment_id: u64,
    pub vector_count: usize,
    pub dimension: usize,
    pub created_at: u64, // unix timestamp in seconds
    pub query_visit_count: Arc<AtomicU64>,
    pub centroid: Vec<f32>, // mean of all vectors for overlap detection
}

impl SegmentStats {
    pub fn new(segment_id: u64, vector_count: usize, dimension: usize, centroid: Vec<f32>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            segment_id,
            vector_count,
            dimension,
            created_at: now,
            query_visit_count: Arc::new(AtomicU64::new(0)),
            centroid,
        }
    }

    pub fn age_secs(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now.saturating_sub(self.created_at)
    }

    pub fn size_bytes(&self) -> usize {
        self.vector_count * self.dimension * std::mem::size_of::<f32>()
    }

    pub fn size_mb(&self) -> f32 {
        self.size_bytes() as f32 / (1024.0 * 1024.0)
    }

    pub fn visit_count(&self) -> u64 {
        self.query_visit_count.load(Ordering::Relaxed)
    }

    pub fn record_visit(&self) {
        self.query_visit_count.fetch_add(1, Ordering::Relaxed);
    }
}

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
    fn score_segment(&self, stats: &SegmentStats, all_stats: &[SegmentStats]) -> f32;

    /// Select which segments to merge.
    /// Returns Some(vec of segment IDs) if a merge should happen, None otherwise.
    fn select_merge_candidates(&self, all_stats: &[SegmentStats]) -> Option<Vec<u64>>;
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
    fn score_segment(&self, _stats: &SegmentStats, all_stats: &[SegmentStats]) -> f32 {
        if all_stats.len() < self.min_segment_count {
            return 0.0;
        }

        // Score based on how many small segments exist
        let small_count = all_stats.iter().filter(|s| s.size_mb() < 50.0).count();

        let pressure = (small_count as f32) / (self.max_segments_per_tier as f32);
        (pressure - 1.0).max(0.0).min(1.0)
    }

    fn select_merge_candidates(&self, all_stats: &[SegmentStats]) -> Option<Vec<u64>> {
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
    pub overlap_threshold: f32, // 0.0-1.0; merge if overlap > threshold
    pub min_segments_to_keep: usize,
}

impl SimilarityAwarePolicy {
    pub fn new(overlap_threshold: f32, min_segments_to_keep: usize) -> Self {
        Self {
            overlap_threshold,
            min_segments_to_keep,
        }
    }
}

impl Default for SimilarityAwarePolicy {
    fn default() -> Self {
        Self {
            overlap_threshold: 0.7,
            min_segments_to_keep: 1,
        }
    }
}

impl CompactionPolicy for SimilarityAwarePolicy {
    fn score_segment(&self, stats: &SegmentStats, all_stats: &[SegmentStats]) -> f32 {
        if all_stats.len() <= self.min_segments_to_keep {
            return 0.0;
        }

        // Score based on overlap with other segments
        let avg_overlap = all_stats
            .iter()
            .filter(|s| s.segment_id != stats.segment_id)
            .map(|s| centroid_distance(&stats.centroid, &s.centroid))
            .sum::<f32>()
            / (all_stats.len() as f32 - 1.0);

        // Higher overlap = higher merge score
        (avg_overlap - self.overlap_threshold).max(0.0).min(1.0)
    }

    fn select_merge_candidates(&self, all_stats: &[SegmentStats]) -> Option<Vec<u64>> {
        if all_stats.len() <= self.min_segments_to_keep {
            return None;
        }

        // Find the most similar pair
        let mut best_pair = None;
        let mut best_distance = f32::INFINITY;

        for i in 0..all_stats.len() {
            for j in (i + 1)..all_stats.len() {
                let dist = centroid_distance(&all_stats[i].centroid, &all_stats[j].centroid);
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
    fn score_segment(&self, stats: &SegmentStats, all_stats: &[SegmentStats]) -> f32 {
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

    fn select_merge_candidates(&self, all_stats: &[SegmentStats]) -> Option<Vec<u64>> {
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
            let dist = centroid_distance(&least_visited.centroid, &stats.centroid);
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
    segments_to_merge: &[SegmentStats],
    all_segments: &[SegmentStats],
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
                    .map(|s| centroid_distance(&segments_to_merge[i].centroid, &s.centroid))
                    .sum::<f32>()
                    / (all_segments.len() as f32)
            })
            .sum::<f32>()
            / segments_to_merge.len() as f32
    } else {
        0.0
    };

    let overlap_reduction = (1.0 - avg_overlap_before).max(0.0);

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

/// Compute centroid distance (L2 norm between centroids).
fn centroid_distance(c1: &[f32], c2: &[f32]) -> f32 {
    if c1.is_empty() || c2.is_empty() {
        return f32::INFINITY;
    }

    let mut sum = 0.0;
    for (v1, v2) in c1.iter().zip(c2.iter()) {
        let diff = v1 - v2;
        sum += diff * diff;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(id: u64, count: usize, dim: usize) -> SegmentStats {
        SegmentStats::new(id, count, dim, vec![0.5; dim])
    }

    #[test]
    fn test_size_tiered_policy() {
        let policy = SizeTieredPolicy::default();
        let stats = vec![
            make_stats(1, 100, 128),
            make_stats(2, 100, 128),
            make_stats(3, 100, 128),
            make_stats(4, 100, 128),
            make_stats(5, 1000, 128),
        ];

        let candidates = policy.select_merge_candidates(&stats);
        assert!(candidates.is_some());
    }

    #[test]
    fn test_similarity_aware_policy() {
        let policy = SimilarityAwarePolicy::default();
        let stats = vec![make_stats(1, 100, 128), make_stats(2, 100, 128)];

        let candidates = policy.select_merge_candidates(&stats);
        assert!(candidates.is_some());
    }
}
