use super::compaction_policy::{
    CompactionPolicy, QueryDrivenPolicy, SimilarityAwarePolicy, SizeTieredPolicy,
};
use super::segment::{Segment, SegmentBuilder, SegmentSummary};
use super::task_system::TaskSystem;
use crate::err::DendraError;
use crate::io::read_u8_le;
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const STORE_MAGIC: &[u8; 4] = b"VSTR";
const STORE_VERSION: u8 = 2;
const DEFAULT_MAX_MB_PER_SEGMENT: usize = 100;
const DEFAULT_ASYNC_SEAL_QUEUE_CAPACITY: usize = 2;

/// Serializable compaction policy selection.
#[derive(Serialize, Deserialize, Clone)]
pub enum CompactionPolicyType {
    SizeTiered {
        max_segments_per_tier: usize,
        tier_size_ratio: usize,
    },
    SimilarityAware {
        overlap_threshold: f32,
    },
    QueryDriven {
        co_access_threshold: u64,
    },
}

impl Default for CompactionPolicyType {
    fn default() -> Self {
        Self::SizeTiered {
            max_segments_per_tier: 4,
            tier_size_ratio: 10,
        }
    }
}

/// Serializable query routing selection.
#[derive(Serialize, Deserialize, Clone)]
pub enum RoutingPolicyType {
    /// Disable routing and search all segments.
    Disabled,
    /// Search only the top-N most promising segments.
    FlatTopK {
        max_segments: usize,
        min_segments: usize,
    },
}

impl Default for RoutingPolicyType {
    fn default() -> Self {
        Self::FlatTopK {
            max_segments: 8,
            min_segments: 2,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EngineConfig {
    pub leaf_size: usize,
    pub num_trees: usize,
    pub dimension: usize,
    pub seed: u64,
    pub max_segment_capacity: usize,
    pub async_seal_queue_capacity: usize,
    pub num_workers: usize,
    pub compaction_policy: CompactionPolicyType,
    pub routing_policy: RoutingPolicyType,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            leaf_size: 64,
            num_trees: 2,
            dimension: 128,
            seed: 42,
            max_segment_capacity: DEFAULT_MAX_MB_PER_SEGMENT,
            async_seal_queue_capacity: DEFAULT_ASYNC_SEAL_QUEUE_CAPACITY,
            num_workers: 4,
            compaction_policy: CompactionPolicyType::default(),
            routing_policy: RoutingPolicyType::default(),
        }
    }
}

impl EngineConfig {
    pub fn new(
        leaf_size: usize,
        num_trees: usize,
        dimension: usize,
        seed: u64,
        max_segment_capacity: usize,
        async_seal_queue_capacity: usize,
    ) -> Self {
        Self {
            leaf_size,
            num_trees,
            dimension,
            seed,
            max_segment_capacity,
            async_seal_queue_capacity,
            ..Self::default()
        }
    }

    pub fn with_compaction_policy(mut self, policy: CompactionPolicyType) -> Self {
        self.compaction_policy = policy;
        self
    }

    pub fn with_routing_policy(mut self, policy: RoutingPolicyType) -> Self {
        self.routing_policy = policy;
        self
    }
}

struct SealTask {
    segment_id: u64,
    builder: SegmentBuilder,
    segment_path: PathBuf,
    leaf_size: usize,
    num_trees: usize,
    seed: u64,
}

struct CompactTask {
    old_segment_ids: Vec<u64>,
    new_segment_id: u64,
    dir: PathBuf,
    dimension: usize,
    leaf_size: usize,
    num_trees: usize,
    seed: u64,
}

enum StorageTask {
    Seal(SealTask),
    Compact(CompactTask),
}

enum StorageTaskResult {
    Sealed {
        segment_id: u64,
        segment_path: PathBuf,
    },
    Compacted {
        old_segment_ids: Vec<u64>,
        new_segment_id: u64,
        segment_path: PathBuf,
    },
}

pub(crate) struct Engine {
    dir: PathBuf,
    segment_builder: SegmentBuilder,
    next_segment_id: u64,
    pub config: EngineConfig,

    segments: Vec<Segment>,
    segment_ids: Vec<u64>,
    segment_stats: Vec<SegmentSummary>,

    tasks: TaskSystem<StorageTask, Result<StorageTaskResult, DendraError>>,
    in_flight_tasks: usize,

    compaction_in_flight: bool,
    compaction_policy: Box<dyn CompactionPolicy>,
}

impl Engine {
    pub(crate) fn new(dir: PathBuf, config: EngineConfig) -> Self {
        let segment_builder = SegmentBuilder::new(config.dimension, config.max_segment_capacity);
        let tasks = TaskSystem::new(
            config.async_seal_queue_capacity,
            config.num_workers,
            |task: StorageTask| -> Result<StorageTaskResult, DendraError> {
                match task {
                    StorageTask::Seal(t) => {
                        let mut builder = t.builder;
                        let segment = builder.build(t.leaf_size, t.num_trees, t.seed)?;
                        segment.flush(&t.segment_path)?;
                        Ok(StorageTaskResult::Sealed {
                            segment_id: t.segment_id,
                            segment_path: t.segment_path,
                        })
                    }
                    StorageTask::Compact(t) => {
                        let merged = Engine::compact_segments_into_one(
                            &t.dir,
                            &t.old_segment_ids,
                            t.dimension,
                            t.leaf_size,
                            t.num_trees,
                            t.seed,
                        )?;

                        let out_path = t.dir.join(format!("segment{}", t.new_segment_id));
                        merged.flush(&out_path)?;

                        Ok(StorageTaskResult::Compacted {
                            old_segment_ids: t.old_segment_ids,
                            new_segment_id: t.new_segment_id,
                            segment_path: out_path,
                        })
                    }
                }
            },
        );
        let compaction_policy = Engine::create_compaction_policy(&config.compaction_policy);

        Self {
            dir,
            segment_builder,
            next_segment_id: 0,
            config,
            segments: Vec::new(),
            segment_ids: Vec::new(),
            segment_stats: Vec::new(),
            tasks,
            in_flight_tasks: 0,
            compaction_in_flight: false,
            compaction_policy,
        }
    }

    fn create_compaction_policy(policy_type: &CompactionPolicyType) -> Box<dyn CompactionPolicy> {
        match policy_type {
            CompactionPolicyType::SizeTiered {
                max_segments_per_tier,
                tier_size_ratio,
            } => Box::new(SizeTieredPolicy {
                max_segments_per_tier: *max_segments_per_tier,
                tier_size_ratio: *tier_size_ratio,
                min_segment_count: 2,
            }),
            CompactionPolicyType::SimilarityAware { overlap_threshold } => {
                Box::new(SimilarityAwarePolicy {
                    overlap_threshold: *overlap_threshold,
                    max_merge_entropy: 0.6,
                })
            }
            CompactionPolicyType::QueryDriven {
                co_access_threshold,
            } => Box::new(QueryDrivenPolicy {
                co_access_threshold: *co_access_threshold,
                min_segments_to_keep: 1,
            }),
        }
    }

    pub(crate) fn insert(&mut self, vector: &[f32], id: u32) -> Result<(), DendraError> {
        if !self.segment_builder.try_insert(vector, id)? {
            let old_builder = std::mem::replace(
                &mut self.segment_builder,
                SegmentBuilder::new(self.config.dimension, self.config.max_segment_capacity),
            );
            self.enqueue_builder_for_seal(old_builder)?;
            self.drain_completed_tasks()?;
            self.maybe_schedule_compaction()?;
            self.segment_builder.try_insert(vector, id)?;
        }
        Ok(())
    }

    pub(crate) fn flush(&mut self) -> Result<(), DendraError> {
        if self.segment_builder.current_capacity > 0 {
            let old_builder = std::mem::replace(
                &mut self.segment_builder,
                SegmentBuilder::new(self.config.dimension, self.config.max_segment_capacity),
            );
            self.enqueue_builder_for_seal(old_builder)?;
        }

        self.maintenance_tick()?;

        self.await_all_tasks()
    }

    pub(crate) fn save(&mut self) -> Result<(), DendraError> {
        self.flush()?;
        fs::create_dir_all(&self.dir)?;

        let meta_path = self.dir.join("store.bin");
        let mut w = BufWriter::new(File::create(meta_path)?);

        w.write_all(STORE_MAGIC)?;
        w.write_all(&STORE_VERSION.to_le_bytes())?;

        let metadata = (
            self.config.dimension as u32,
            self.config.leaf_size as u32,
            self.config.num_trees as u32,
            self.config.seed,
            self.config.max_segment_capacity as u64,
            self.next_segment_id,
        );
        let config = bincode::config::standard();
        let encoded = bincode::encode_to_vec(metadata, config)
            .map_err(|e| DendraError::Serialization(e.to_string()))?;
        w.write_all(&encoded)?;
        Ok(())
    }

    pub(crate) fn load(dir: &Path, num_workers: usize) -> Result<Self, DendraError> {
        let meta_path = dir.join("store.bin");
        let mut r = BufReader::new(File::open(meta_path)?);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != STORE_MAGIC {
            return Err(DendraError::InvalidHeader {
                expected: String::from_utf8_lossy(STORE_MAGIC).to_string(),
                received: String::from_utf8_lossy(&magic).to_string(),
            });
        }

        let version = read_u8_le(&mut r)?;
        if version != STORE_VERSION {
            return Err(DendraError::InvalidHeader {
                expected: STORE_VERSION.to_string(),
                received: version.to_string(),
            });
        }

        let mut metadata_bytes = Vec::new();
        r.read_to_end(&mut metadata_bytes)?;
        let config = bincode::config::standard();
        let (dimension, leaf_size, num_trees, seed, max_segment_capacity, next_segment_id): (
            u32,
            u32,
            u32,
            u64,
            u64,
            u64,
        ) = bincode::decode_from_slice(&metadata_bytes, config)
            .map(|(v, _)| v)
            .map_err(|e| DendraError::Deserialization(e.to_string()))?;

        let cfg = EngineConfig {
            leaf_size: leaf_size as usize,
            num_trees: num_trees as usize,
            dimension: dimension as usize,
            seed,
            max_segment_capacity: max_segment_capacity as usize,
            async_seal_queue_capacity: DEFAULT_ASYNC_SEAL_QUEUE_CAPACITY,
            num_workers,
            compaction_policy: CompactionPolicyType::default(),
            routing_policy: RoutingPolicyType::default(),
        };

        let mut engine = Self::new(dir.to_path_buf(), cfg);
        engine.next_segment_id = next_segment_id;

        let mut found = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(id_str) = name.strip_prefix("segment") else {
                continue;
            };
            let Some(id) = id_str.parse::<u64>().ok() else {
                continue;
            };
            found.push((id, entry.path()));
        }

        found.sort_by_key(|(id, _)| *id);
        for (id, path) in found {
            let seg = Segment::open(&path)?;
            if seg.dim != engine.config.dimension {
                return Err(DendraError::InvalidVectorDimension {
                    expected: engine.config.dimension,
                    received: seg.dim,
                });
            }
            engine.segment_ids.push(id);
            engine.segments.push(seg);
        }

        Ok(engine)
    }

    pub(crate) fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub(crate) fn segment_summaries(&self) -> &[SegmentSummary] {
        &self.segment_stats
    }

    pub(crate) fn maintenance_tick(&mut self) -> Result<(), DendraError> {
        self.drain_completed_tasks()?;
        self.maybe_schedule_compaction()?;
        Ok(())
    }

    fn compact_segments_into_one(
        dir: &Path,
        old_segment_ids: &[u64],
        dimension: usize,
        leaf_size: usize,
        num_trees: usize,
        seed: u64,
    ) -> Result<Segment, DendraError> {
        let mut total_rows = 0usize;
        for id in old_segment_ids {
            let p = dir.join(format!("segment{}", id));
            let seg = Segment::open(&p)?;
            total_rows = total_rows.saturating_add(seg.count);
        }

        let total_bytes = total_rows
            .saturating_mul(dimension)
            .saturating_mul(std::mem::size_of::<f32>());
        let capacity_mb = (total_bytes / (1024 * 1024)).saturating_add(1);

        let mut builder = SegmentBuilder::new(dimension, capacity_mb);

        for id in old_segment_ids {
            let p = dir.join(format!("segment{}", id));
            let seg = Segment::open(&p)?;
            if seg.dim != dimension {
                return Err(DendraError::InvalidVectorDimension {
                    expected: dimension,
                    received: seg.dim,
                });
            }

            for row in 0..seg.count {
                let v = seg.vector_at(row)?;
                let ext_id = seg.id_at(row)?;
                let ok = builder.try_insert(v, ext_id)?;
                if !ok {
                    return Err(DendraError::UnsupportedOperation(
                        "compaction builder capacity underestimated".to_string(),
                    ));
                }
            }
        }

        builder.build(leaf_size, num_trees, seed)
    }

    fn enqueue_builder_for_seal(&mut self, builder: SegmentBuilder) -> Result<(), DendraError> {
        let segment_id = self.next_segment_id;
        self.next_segment_id += 1;
        let segment_path = self.dir.join(format!("segment{}", segment_id));

        let task = StorageTask::Seal(SealTask {
            segment_id,
            builder,
            segment_path,
            leaf_size: self.config.leaf_size,
            num_trees: self.config.num_trees,
            seed: self.config.seed,
        });
        info!("Scheduling seal for segment {}", segment_id);
        self.tasks.submit(task).map_err(DendraError::TaskSystem)?;
        self.in_flight_tasks += 1;
        Ok(())
    }

    fn maybe_schedule_compaction(&mut self) -> Result<(), DendraError> {
        if self.compaction_in_flight {
            return Ok(());
        }

        if self.segments.is_empty() {
            return Ok(());
        }

        // Stats are updated in integrate_task_result, so no need to refresh here

        // Let the policy decide if we should compact
        if let Some(merge_candidates) = self
            .compaction_policy
            .select_merge_candidates(&self.segment_stats)
        {
            let new_segment_id = self.next_segment_id;
            self.next_segment_id += 1;

            info!(
                "Scheduling compaction: merging {} segments into segment {}",
                merge_candidates.len(),
                new_segment_id
            );

            let task = StorageTask::Compact(CompactTask {
                old_segment_ids: merge_candidates,
                new_segment_id,
                dir: self.dir.clone(),
                dimension: self.config.dimension,
                leaf_size: self.config.leaf_size,
                num_trees: self.config.num_trees,
                seed: self.config.seed,
            });

            self.tasks.submit(task).map_err(DendraError::TaskSystem)?;
            self.in_flight_tasks += 1;
            self.compaction_in_flight = true;
        }

        Ok(())
    }

    fn integrate_task_result(
        &mut self,
        result: Result<StorageTaskResult, DendraError>,
    ) -> Result<(), DendraError> {
        self.in_flight_tasks = self.in_flight_tasks.saturating_sub(1);

        match result? {
            StorageTaskResult::Sealed {
                segment_id,
                segment_path,
            } => {
                info!("Seal completed for segment {}", segment_id);
                let seg = Segment::open(&segment_path)?;

                // Compute centroid and radius for this segment
                let mut centroid = vec![0.0; seg.dim];
                let mut radius = 0.0;
                seg.describe(&mut centroid, &mut radius)?;
                let entropy = seg.estimate_entropy()?;

                let stats =
                    SegmentSummary::new(segment_id, seg.count, seg.dim, centroid, radius, entropy);
                self.segment_ids.push(segment_id);
                self.segments.push(seg);
                self.segment_stats.push(stats);

                debug!(
                    "Segment {} now loaded and available for queries",
                    segment_id
                );
            }
            StorageTaskResult::Compacted {
                old_segment_ids,
                new_segment_id,
                segment_path,
            } => {
                info!(
                    "Compaction completed: {} segments merged into segment {}",
                    old_segment_ids.len(),
                    new_segment_id
                );
                let old_set = old_segment_ids.iter().copied().collect::<HashSet<_>>();

                let new_seg = Segment::open(&segment_path)?;
                let mut centroid = vec![0.0; new_seg.dim];
                let mut radius = 0.0;
                new_seg.describe(&mut centroid, &mut radius)?;
                let entropy = new_seg.estimate_entropy()?;
                let new_stats = SegmentSummary::new(
                    new_segment_id,
                    new_seg.count,
                    new_seg.dim,
                    centroid,
                    radius,
                    entropy,
                );

                let mut new_ids = Vec::with_capacity(self.segment_ids.len());
                let mut new_segments = Vec::with_capacity(self.segments.len());
                let mut new_stats_list = Vec::with_capacity(self.segment_stats.len());

                for i in 0..self.segment_ids.len() {
                    if !old_set.contains(&self.segment_ids[i]) {
                        new_ids.push(self.segment_ids[i]);
                        new_segments.push(self.segments.remove(0));
                        new_stats_list.push(self.segment_stats.remove(0));
                    } else {
                        // Skip this segment (it will be compacted)
                        self.segments.remove(0);
                        self.segment_stats.remove(0);
                    }
                }

                new_ids.push(new_segment_id);
                new_segments.push(new_seg);
                new_stats_list.push(new_stats);

                self.segment_ids = new_ids;
                self.segments = new_segments;
                self.segment_stats = new_stats_list;

                for old_id in old_segment_ids {
                    let old_path = self.dir.join(format!("segment{}", old_id));
                    let _ = fs::remove_dir_all(old_path);
                    debug!("Deleted compacted segment {}", old_id);
                }

                self.compaction_in_flight = false;
            }
        }

        Ok(())
    }

    fn drain_completed_tasks(&mut self) -> Result<(), DendraError> {
        loop {
            match self
                .tasks
                .try_recv_result()
                .map_err(DendraError::TaskSystem)?
            {
                Some(result) => self.integrate_task_result(result)?,
                None => return Ok(()),
            }
        }
    }
    fn await_all_tasks(&mut self) -> Result<(), DendraError> {
        while self.in_flight_tasks > 0 {
            let result = self.tasks.recv_result().map_err(DendraError::TaskSystem)?;
            self.integrate_task_result(result)?;
            self.maybe_schedule_compaction()?;
        }
        Ok(())
    }
}
