use crate::core::config::EngineConfig;
use crate::core::task_system::TaskSystem;
use crate::err::EngramError;
use crate::index::{NigStats, RouteScratch};
use crate::math::{MetricFn, l2_distance_sq_prefix};
use crate::query::Query;
use crate::segment::{ActiveState, SealedState, Segment, SegmentTelemetry};
use arc_swap::ArcSwap;
use log::{debug, info};

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

const COMPACTION_SEGMENT_TRIGGER: usize = 8;
const SEGMENT_DIR_PREFIX: &str = "segment_";

#[derive(Clone, Debug)]
pub struct CompactionExplanation {
    pub left_segment_id: u64,
    pub right_segment_id: u64,
    pub left_depth: u8,
    pub right_depth: u8,
    pub left_vectors: usize,
    pub right_vectors: usize,
    pub left_query_visits: u64,
    pub right_query_visits: u64,
    pub centroid_distance_l2: f32,
    pub bic_gain: f64,
    pub drift_proxy_l2: f32,
    pub estimated_vectors_rewritten: usize,
    pub estimated_bytes_rewritten: usize,
}

#[derive(Clone, Debug)]
pub struct RoutingSegmentExplanation {
    pub segment_id: u64,
    pub depth: u8,
    pub vector_count: usize,
    pub query_visits: u64,
    pub root_score: f64,
    pub posterior_mass: f64,
    pub cumulative_mass: f64,
    pub selected: bool,
}

#[derive(Clone, Debug)]
pub struct RoutingExplanation {
    pub delta: f32,
    pub target_mass: f64,
    pub selected_segment_indices: Vec<usize>,
    pub segments: Vec<RoutingSegmentExplanation>,
}

#[derive(Default)]
struct RoutingScratch {
    scores: Vec<(usize, f64)>,
    probs: Vec<f64>,
}

#[derive(Default)]
pub struct QueryScratch {
    selected_segments: Vec<usize>,
    routing: RoutingScratch,
    candidate_rows: Vec<u32>,
    lsh_route: RouteScratch,
    best_by_id: HashMap<u32, f32>,
    decoded: Vec<f32>,
    heap: BinaryHeap<Reverse<(OrderedF32, u32)>>,
}

#[derive(Clone)]
struct PublishedSegment {
    segment: Arc<Segment<SealedState>>,
    query_visits: Arc<AtomicU64>,
}

impl PublishedSegment {
    fn new(segment: Segment<SealedState>) -> Result<Self, EngramError> {
        let query_visits = segment.telemetry().map(|t| t.query_visits).unwrap_or(0);
        Ok(Self {
            segment: Arc::new(segment),
            query_visits: Arc::new(AtomicU64::new(query_visits)),
        })
    }

    fn current_query_visits(&self) -> u64 {
        self.query_visits.load(Ordering::Relaxed)
    }

    fn telemetry_snapshot(&self) -> Result<SegmentTelemetry, EngramError> {
        let mut telemetry = self.segment.telemetry_snapshot()?;
        telemetry.query_visits = self.current_query_visits();
        Ok(telemetry)
    }

    fn persist(&self, dir: &Path) -> Result<(), EngramError> {
        let mut segment = (*self.segment).clone();
        segment.set_query_visits(self.current_query_visits())?;
        segment.persist(dir)
    }
}

struct EngineSnapshot {
    sealed: Vec<PublishedSegment>,
}

impl EngineSnapshot {
    fn from_segments(sealed: &[PublishedSegment]) -> Self {
        Self {
            sealed: sealed.to_vec(),
        }
    }
}

enum EngineTask {
    Seal {
        active: Box<Segment<ActiveState>>,
        config: EngineConfig,
    },
    Compact {
        left: Arc<Segment<SealedState>>,
        right: Arc<Segment<SealedState>>,
        merged_query_visits: u64,
        config: EngineConfig,
    },
}

enum EngineTaskResult {
    Sealed(Segment<SealedState>),
    Compacted {
        merged: Segment<SealedState>,
        old_ids: [u64; 2],
        bic_gain: f64,
        drift_proxy_l2: f32,
        depth_before: u8,
        depth_after: u8,
    },
}

struct EngineRuntime {
    dir: PathBuf,
    active: Segment<ActiveState>,
    sealed: Vec<PublishedSegment>,
    next_segment_id: u64,
    tasks: TaskSystem<EngineTask, Result<EngineTaskResult, EngramError>>,
    pending_seals: VecDeque<Segment<ActiveState>>,
    seal_in_flight: bool,
    in_flight: usize,
    compaction_in_flight: bool,
}

pub(crate) struct Engine {
    config: EngineConfig,
    snapshot: ArcSwap<EngineSnapshot>,
    runtime: Mutex<EngineRuntime>,
}

impl EngineRuntime {
    fn new(dir: PathBuf, config: &EngineConfig) -> Self {
        let _ = std::fs::create_dir_all(&dir);

        let task_runner = TaskSystem::new(
            config.seal_queue_capacity,
            config.num_workers,
            |task: EngineTask| match task {
                EngineTask::Seal { active, config } => {
                    let active = *active;
                    let seg_id = active.id();
                    let vecs = active.len();
                    let start = Instant::now();
                    info!(
                        "seal task start: id={} vectors={} tables={} bits={} dims_per_bit={} probe_radius={} expert_dims={} min_candidates={} max_candidates={} gamma={:.2}",
                        seg_id,
                        vecs,
                        config.lsh_num_tables,
                        config.lsh_bits_per_table,
                        config.lsh_dims_per_bit,
                        config.lsh_probe_hamming_radius,
                        config.lsh_bucket_expert_dims,
                        config.lsh_min_candidates,
                        config.lsh_max_candidates,
                        config.lsh_adaptive_gamma
                    );

                    let out = active
                        .seal(
                            config.lsh_bits_per_table,
                            config.lsh_num_tables,
                            config.lsh_dims_per_bit,
                            config.lsh_probe_hamming_radius,
                            config.lsh_bucket_expert_dims,
                            config.lsh_min_candidates,
                            config.lsh_max_candidates,
                            config.lsh_adaptive_gamma,
                            config.seed,
                        )
                        .map(EngineTaskResult::Sealed);

                    match &out {
                        Ok(_) => info!(
                            "seal task done: id={} vectors={} elapsed_ms={:.3}",
                            seg_id,
                            vecs,
                            start.elapsed().as_secs_f64() * 1000.0
                        ),
                        Err(e) => info!(
                            "seal task failed: id={} vectors={} elapsed_ms={:.3} err={}",
                            seg_id,
                            vecs,
                            start.elapsed().as_secs_f64() * 1000.0,
                            e
                        ),
                    }

                    out
                }
                EngineTask::Compact {
                    left,
                    right,
                    merged_query_visits,
                    config,
                } => {
                    let old_ids = [left.id(), right.id()];
                    let depth_before = left.depth().max(right.depth());
                    let bic_gain = bic_gain(left.root_stats()?, right.root_stats()?);
                    compact_segments(&left, &right, merged_query_visits, &config).map(
                        |(merged, drift_proxy_l2)| {
                            let depth_after = merged.depth();
                            EngineTaskResult::Compacted {
                                merged,
                                old_ids,
                                bic_gain,
                                drift_proxy_l2,
                                depth_before,
                                depth_after,
                            }
                        },
                    )
                }
            },
        );

        Self {
            dir,
            active: Segment::new_active(0, config.dimension),
            sealed: Vec::new(),
            next_segment_id: 1,
            tasks: task_runner,
            pending_seals: VecDeque::new(),
            seal_in_flight: false,
            in_flight: 0,
            compaction_in_flight: false,
        }
    }

    fn rotate_and_seal_active(&mut self, config: &EngineConfig) -> Result<(), EngramError> {
        if self.active.is_empty() {
            return Ok(());
        }

        let sealed_id = self.active.id();
        let sealed_vectors = self.active.len();

        let current = mem::replace(
            &mut self.active,
            Segment::new_active(self.next_segment_id, config.dimension),
        );
        self.next_segment_id += 1;

        info!(
            "enqueue seal: id={} vectors={} next_active_id={} in_flight={} pending_seals={}",
            sealed_id,
            sealed_vectors,
            self.next_segment_id - 1,
            self.in_flight,
            self.pending_seals.len()
        );

        self.pending_seals.push_back(current);
        self.schedule_next_seal_if_possible(config)?;
        Ok(())
    }

    fn schedule_next_seal_if_possible(&mut self, config: &EngineConfig) -> Result<(), EngramError> {
        if self.seal_in_flight {
            return Ok(());
        }

        let Some(active) = self.pending_seals.pop_front() else {
            return Ok(());
        };

        self.tasks.submit(EngineTask::Seal {
            active: Box::new(active),
            config: config.clone(),
        })?;
        self.seal_in_flight = true;
        self.in_flight += 1;
        Ok(())
    }

    fn maybe_schedule_compaction(&mut self, config: &EngineConfig) -> Result<(), EngramError> {
        if self.compaction_in_flight || self.sealed.len() < COMPACTION_SEGMENT_TRIGGER {
            return Ok(());
        }

        let Some((left_idx, right_idx)) =
            select_compaction_pair(&self.sealed, config.compaction_depth_cap)
        else {
            return Ok(());
        };

        let (hi, lo) = if left_idx > right_idx {
            (left_idx, right_idx)
        } else {
            (right_idx, left_idx)
        };

        let hi_seg = self.sealed.swap_remove(hi);
        let lo_seg = self.sealed.swap_remove(lo);
        let merged_query_visits = lo_seg
            .current_query_visits()
            .saturating_add(hi_seg.current_query_visits());

        self.tasks.submit(EngineTask::Compact {
            left: Arc::clone(&lo_seg.segment),
            right: Arc::clone(&hi_seg.segment),
            merged_query_visits,
            config: config.clone(),
        })?;

        self.in_flight += 1;
        self.compaction_in_flight = true;
        debug!(
            "compaction scheduled: pair=({}, {}) depth_cap={}",
            lo, hi, config.compaction_depth_cap
        );
        Ok(())
    }

    fn poll_tasks(&mut self, config: &EngineConfig) -> Result<bool, EngramError> {
        let mut snapshot_changed = false;
        while let Some(result) = self.tasks.try_recv_result()? {
            self.integrate_task_result(result)?;
            self.in_flight = self.in_flight.saturating_sub(1);
            self.schedule_next_seal_if_possible(config)?;
            self.maybe_schedule_compaction(config)?;
            snapshot_changed = true;
        }
        Ok(snapshot_changed)
    }

    fn integrate_task_result(
        &mut self,
        result: Result<EngineTaskResult, EngramError>,
    ) -> Result<(), EngramError> {
        match result? {
            EngineTaskResult::Sealed(segment) => {
                self.seal_in_flight = false;
                let entry = PublishedSegment::new(segment)?;
                let seg_dir = self.segment_dir(entry.segment.id());
                let persist_start = Instant::now();
                info!(
                    "persist sealed segment start: id={} path={}",
                    entry.segment.id(),
                    seg_dir.display()
                );
                entry.persist(&seg_dir)?;
                info!(
                    "persist sealed segment done: id={} path={} elapsed_ms={:.3}",
                    entry.segment.id(),
                    seg_dir.display(),
                    persist_start.elapsed().as_secs_f64() * 1000.0
                );
                info!(
                    "segment sealed: id={} vectors={} depth={}",
                    entry.segment.id(),
                    entry.segment.len(),
                    entry.segment.depth()
                );
                self.sealed.push(entry);
            }
            EngineTaskResult::Compacted {
                merged,
                old_ids,
                bic_gain,
                drift_proxy_l2,
                depth_before,
                depth_after,
            } => {
                let entry = PublishedSegment::new(merged)?;
                let merged_dir = self.segment_dir(entry.segment.id());
                let merged_name =
                    merged_dir
                        .file_name()
                        .and_then(|n| n.to_str())
                        .ok_or_else(|| EngramError::InvalidHeader {
                            expected: "valid segment directory name".to_string(),
                            received: merged_dir.display().to_string(),
                        })?;
                let temp_dir = self.dir.join(format!("{}_tmp", merged_name));
                entry.persist(&temp_dir)?;

                if merged_dir.exists() {
                    std::fs::remove_dir_all(&merged_dir)?;
                }
                std::fs::rename(&temp_dir, &merged_dir)?;

                for old in old_ids {
                    let old_dir = self.segment_dir(old);
                    if old_dir.exists() {
                        let _ = std::fs::remove_dir_all(old_dir);
                    }
                }

                self.sealed.push(entry);
                self.compaction_in_flight = false;
                info!(
                    "compaction complete: old_ids={:?} bic_gain={:.6} drift_proxy_l2={:.6} depth {}->{}",
                    old_ids, bic_gain, drift_proxy_l2, depth_before, depth_after
                );
            }
        }
        Ok(())
    }

    fn segment_dir(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{}{}", SEGMENT_DIR_PREFIX, id))
    }
}

impl Engine {
    pub(crate) fn new(dir: PathBuf, config: EngineConfig) -> Self {
        let runtime = EngineRuntime::new(dir, &config);
        Self {
            config,
            snapshot: ArcSwap::from_pointee(EngineSnapshot::from_segments(&runtime.sealed)),
            runtime: Mutex::new(runtime),
        }
    }

    pub(crate) fn load(dir: &Path, config: EngineConfig) -> Result<Self, EngramError> {
        if !dir.is_dir() {
            return Err(EngramError::InvariantViolation(
                "load requires an existing database directory".to_string(),
            ));
        }

        let engine = Self::new(dir.to_path_buf(), config);
        let mut runtime = engine.lock_runtime()?;

        let mut max_id = 0u64;
        let mut found_segment = false;
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with(SEGMENT_DIR_PREFIX) {
                continue;
            }

            let id = name[SEGMENT_DIR_PREFIX.len()..]
                .parse::<u64>()
                .map_err(|e| EngramError::Codec(e.to_string()))?;

            let seg = Segment::<SealedState>::load(&entry.path())?;
            max_id = max_id.max(id);
            runtime.sealed.push(PublishedSegment::new(seg)?);
            found_segment = true;
        }

        if !found_segment {
            return Err(EngramError::InvariantViolation(
                "load requires an existing populated database (no sealed segments found)"
                    .to_string(),
            ));
        }

        runtime.next_segment_id = max_id + 1;
        let active_dim = runtime.sealed[0].segment.dim();
        runtime.active = Segment::new_active(runtime.next_segment_id, active_dim);
        engine.publish_snapshot(&runtime);
        drop(runtime);
        Ok(engine)
    }

    pub(crate) fn insert(&mut self, vector: &[f32], id: u32) -> Result<(), EngramError> {
        let mut runtime = self.lock_runtime()?;
        let snapshot_changed = runtime.poll_tasks(&self.config)?;
        runtime.active.insert(vector, id)?;

        let max_active = self.config.max_active_vectors().max(1);
        if runtime.active.len() >= max_active {
            runtime.rotate_and_seal_active(&self.config)?;
        }
        runtime.maybe_schedule_compaction(&self.config)?;

        if snapshot_changed {
            self.publish_snapshot(&runtime);
        }
        Ok(())
    }

    pub(crate) fn flush(&mut self) -> Result<(), EngramError> {
        let mut runtime = self.lock_runtime()?;
        let mut snapshot_changed = runtime.poll_tasks(&self.config)?;

        if !runtime.active.is_empty() {
            runtime.rotate_and_seal_active(&self.config)?;
        }

        while runtime.in_flight > 0 {
            let result = runtime.tasks.recv_result()?;
            runtime.integrate_task_result(result)?;
            runtime.in_flight = runtime.in_flight.saturating_sub(1);
            runtime.schedule_next_seal_if_possible(&self.config)?;
            runtime.maybe_schedule_compaction(&self.config)?;
            snapshot_changed = true;
        }

        if snapshot_changed {
            self.publish_snapshot(&runtime);
        }
        Ok(())
    }

    pub(crate) fn save(&mut self) -> Result<(), EngramError> {
        self.flush()?;
        let runtime = self.lock_runtime()?;
        for seg in &runtime.sealed {
            seg.persist(&runtime.segment_dir(seg.segment.id()))?;
        }
        Ok(())
    }

    pub(crate) fn query(
        &self,
        query: &Query,
        scratch: &mut QueryScratch,
        results: &mut Vec<(u32, f32)>,
    ) -> Result<bool, EngramError> {
        self.query_raw(
            query.vector(),
            query.k(),
            query.metric(),
            query.threshold(),
            query.delta(),
            scratch,
            results,
        )
    }

    pub(crate) fn query_raw(
        &self,
        vector: &[f32],
        k: usize,
        metric: Option<MetricFn>,
        threshold: Option<f32>,
        delta: f32,
        scratch: &mut QueryScratch,
        results: &mut Vec<(u32, f32)>,
    ) -> Result<bool, EngramError> {
        let snapshot = self.snapshot.load_full();
        let delta = delta.clamp(0.0, 0.99);
        select_segments_by_routing_mass_into(
            &snapshot.sealed,
            vector,
            delta,
            &mut scratch.selected_segments,
            &mut scratch.routing,
        );

        debug!(
            "query routing: delta={} selected={} total={}",
            delta,
            scratch.selected_segments.len(),
            snapshot.sealed.len()
        );

        scratch.best_by_id.clear();
        for &seg_idx in &scratch.selected_segments {
            let published = &snapshot.sealed[seg_idx];
            let seg = published.segment.as_ref();
            if seg.dim() != vector.len() || seg.is_empty() {
                continue;
            }

            scratch.candidate_rows.clear();
            seg.lsh()?.route_candidate_rows_into(
                vector,
                delta,
                &mut scratch.candidate_rows,
                &mut scratch.lsh_route,
            );

            for &row in &scratch.candidate_rows {
                let row_idx = row as usize;
                if row_idx >= seg.len() {
                    continue;
                }

                let dist = match metric {
                    Some(metric) => {
                        scratch.decoded.resize(seg.dim(), 0.0);
                        seg.decode_row(row_idx, &mut scratch.decoded)?;
                        metric(vector, &scratch.decoded)?
                    }
                    None => seg.store()?.adc_l2_sq(vector, row_idx).sqrt(),
                };

                if let Some(threshold) = threshold
                    && dist > threshold
                {
                    continue;
                }

                if let Some(id) = seg.external_id(row_idx)? {
                    let entry = scratch.best_by_id.entry(id).or_insert(f32::INFINITY);
                    if dist < *entry {
                        *entry = dist;
                    }
                }
            }
        }

        for &seg_idx in &scratch.selected_segments {
            snapshot.sealed[seg_idx]
                .query_visits
                .fetch_add(1, Ordering::Relaxed);
        }

        scratch.heap.clear();
        for (&id, &dist) in &scratch.best_by_id {
            scratch.heap.push(Reverse((OrderedF32(dist), id)));
        }

        results.clear();
        for _ in 0..k.min(scratch.heap.len()) {
            if let Some(Reverse((OrderedF32(d), id))) = scratch.heap.pop() {
                results.push((id, d));
            }
        }

        Ok(!results.is_empty())
    }

    pub(crate) fn config(&self) -> &EngineConfig {
        &self.config
    }

    pub(crate) fn num_sealed_segments(&self) -> usize {
        self.snapshot.load().sealed.len()
    }

    pub(crate) fn sealed_segment_summaries(&self) -> Vec<SegmentTelemetry> {
        self.snapshot
            .load()
            .sealed
            .iter()
            .filter_map(|segment| segment.telemetry_snapshot().ok())
            .collect()
    }

    pub(crate) fn explain_next_compaction(
        &self,
    ) -> Result<Option<CompactionExplanation>, EngramError> {
        let snapshot = self.snapshot.load();
        let Some((left_idx, right_idx)) =
            select_compaction_pair(&snapshot.sealed, self.config.compaction_depth_cap)
        else {
            return Ok(None);
        };

        let left = &snapshot.sealed[left_idx];
        let right = &snapshot.sealed[right_idx];
        let left_telemetry = left.telemetry_snapshot()?;
        let right_telemetry = right.telemetry_snapshot()?;
        let bic_gain = bic_gain(left.segment.root_stats()?, right.segment.root_stats()?);
        let (_, drift_proxy_l2) = compact_segments(
            left.segment.as_ref(),
            right.segment.as_ref(),
            left_telemetry
                .query_visits
                .saturating_add(right_telemetry.query_visits),
            &self.config,
        )?;
        let estimated_vectors_rewritten =
            left_telemetry.vector_count + right_telemetry.vector_count;

        Ok(Some(CompactionExplanation {
            left_segment_id: left.segment.id(),
            right_segment_id: right.segment.id(),
            left_depth: left.segment.depth(),
            right_depth: right.segment.depth(),
            left_vectors: left_telemetry.vector_count,
            right_vectors: right_telemetry.vector_count,
            left_query_visits: left_telemetry.query_visits,
            right_query_visits: right_telemetry.query_visits,
            centroid_distance_l2: centroid_distance_l2(&left_telemetry, &right_telemetry),
            bic_gain,
            drift_proxy_l2,
            estimated_vectors_rewritten,
            estimated_bytes_rewritten: estimated_vectors_rewritten
                .saturating_mul(self.config.dimension)
                .saturating_mul(std::mem::size_of::<f32>()),
        }))
    }

    pub(crate) fn explain_query_routing(&self, query: &Query) -> RoutingExplanation {
        let snapshot = self.snapshot.load();
        let delta = query.delta().clamp(0.0, 0.99);
        explain_routing_mass_budget(&snapshot.sealed, query.vector(), delta)
    }

    fn lock_runtime(&self) -> Result<MutexGuard<'_, EngineRuntime>, EngramError> {
        self.runtime.lock().map_err(|_| {
            EngramError::InvariantViolation("engine runtime mutex poisoned".to_string())
        })
    }

    fn publish_snapshot(&self, runtime: &EngineRuntime) {
        self.snapshot
            .store(Arc::new(EngineSnapshot::from_segments(&runtime.sealed)));
    }
}

fn compact_segments(
    left: &Segment<SealedState>,
    right: &Segment<SealedState>,
    merged_query_visits: u64,
    config: &EngineConfig,
) -> Result<(Segment<SealedState>, f32), EngramError> {
    let left_store = left.store()?;
    let right_store = right.store()?;

    let mut vectors = left_store.decode_all();
    vectors.extend_from_slice(&right_store.decode_all());

    let mut ids = left_store.ids_vec();
    ids.extend_from_slice(&right_store.ids_vec());

    let mut active = Segment::new_active(left.id().max(right.id()) + 1, left.dim());
    for (i, row) in vectors.chunks_exact(left.dim()).enumerate() {
        active.insert(row, ids[i])?;
    }

    let depth = left.depth().max(right.depth()).saturating_add(1);
    let mut merged = active
        .seal(
            config.lsh_bits_per_table,
            config.lsh_num_tables,
            config.lsh_dims_per_bit,
            config.lsh_probe_hamming_radius,
            config.lsh_bucket_expert_dims,
            config.lsh_min_candidates,
            config.lsh_max_candidates,
            config.lsh_adaptive_gamma,
            config.seed,
        )
        .map(|seg| seg.with_depth(depth))?;
    merged.set_query_visits(merged_query_visits)?;

    let before = {
        let mut v = left_store.decode_all();
        v.extend_from_slice(&right_store.decode_all());
        v
    };
    let after = merged.store()?.decode_all();
    let sample = before.len().min(after.len()).min(config.dimension * 1024);
    let drift_proxy_l2 = if sample == 0 {
        0.0
    } else {
        l2_distance_sq_prefix(&before, &after, sample).sqrt()
    };

    Ok((merged, drift_proxy_l2))
}

#[cfg(test)]
fn select_segments_by_routing_mass(
    segments: &[Segment<SealedState>],
    query: &[f32],
    delta: f32,
) -> Vec<usize> {
    let published = segments
        .iter()
        .cloned()
        .map(PublishedSegment::new)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut selected = Vec::new();
    let mut scratch = RoutingScratch::default();
    select_segments_by_routing_mass_into(&published, query, delta, &mut selected, &mut scratch);
    selected
}

fn select_segments_by_routing_mass_into(
    segments: &[PublishedSegment],
    query: &[f32],
    delta: f32,
    out: &mut Vec<usize>,
    scratch: &mut RoutingScratch,
) {
    out.clear();

    if segments.is_empty() {
        return;
    }

    let target_mass = 1.0 - delta.clamp(0.0, 0.99) as f64;
    scratch.scores.clear();
    scratch.probs.clear();
    scratch.probs.resize(segments.len(), 0.0);

    scratch.scores.extend(
        segments
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.segment.lsh().ok().map(|lsh| (i, lsh.root_score(query)))),
    );

    if scratch.scores.is_empty() {
        return;
    }

    let max_score = scratch
        .scores
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut exp_sum = 0.0f64;
    for (idx, score) in &scratch.scores {
        let e = (*score - max_score).exp();
        scratch.probs[*idx] = e;
        exp_sum += e;
    }

    if exp_sum <= 0.0 {
        out.extend(0..segments.len());
        return;
    }

    for p in &mut scratch.probs {
        *p /= exp_sum;
    }

    scratch.scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut cumulative = 0.0f64;
    for (idx, _) in &scratch.scores {
        out.push(*idx);
        cumulative += scratch.probs[*idx];
        if cumulative >= target_mass {
            break;
        }
    }
}

fn explain_routing_mass_budget(
    segments: &[PublishedSegment],
    query: &[f32],
    delta: f32,
) -> RoutingExplanation {
    let delta = delta.clamp(0.0, 0.99);
    let target_mass = 1.0 - delta as f64;

    if segments.is_empty() {
        return RoutingExplanation {
            delta,
            target_mass,
            selected_segment_indices: Vec::new(),
            segments: Vec::new(),
        };
    }

    let mut scores = segments
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.segment.lsh().ok().map(|lsh| (i, lsh.root_score(query))))
        .collect::<Vec<_>>();

    if scores.is_empty() {
        return RoutingExplanation {
            delta,
            target_mass,
            selected_segment_indices: Vec::new(),
            segments: Vec::new(),
        };
    }

    let max_score = scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut exp_sum = 0.0f64;
    let mut probs = vec![0.0f64; segments.len()];
    for (idx, score) in &scores {
        let e = (*score - max_score).exp();
        probs[*idx] = e;
        exp_sum += e;
    }

    if exp_sum <= 0.0 {
        let selected_segment_indices = (0..segments.len()).collect::<Vec<_>>();
        let mut cumulative = 0.0f64;
        let uniform = 1.0 / segments.len() as f64;
        let segment_explanations = selected_segment_indices
            .iter()
            .map(|&idx| {
                cumulative += uniform;
                let telemetry = segments[idx].telemetry_snapshot().ok();
                RoutingSegmentExplanation {
                    segment_id: segments[idx].segment.id(),
                    depth: segments[idx].segment.depth(),
                    vector_count: telemetry
                        .as_ref()
                        .map_or(segments[idx].segment.len(), |t| t.vector_count),
                    query_visits: telemetry.as_ref().map_or(0, |t| t.query_visits),
                    root_score: 0.0,
                    posterior_mass: uniform,
                    cumulative_mass: cumulative,
                    selected: true,
                }
            })
            .collect();
        return RoutingExplanation {
            delta,
            target_mass,
            selected_segment_indices,
            segments: segment_explanations,
        };
    }

    for p in &mut probs {
        *p /= exp_sum;
    }

    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut selected = Vec::new();
    let mut cumulative = 0.0f64;
    let mut segment_explanations = Vec::with_capacity(scores.len());

    for (idx, score) in scores {
        selected.push(idx);
        cumulative += probs[idx];
        let telemetry = segments[idx].telemetry_snapshot().ok();
        segment_explanations.push(RoutingSegmentExplanation {
            segment_id: segments[idx].segment.id(),
            depth: segments[idx].segment.depth(),
            vector_count: telemetry
                .as_ref()
                .map_or(segments[idx].segment.len(), |t| t.vector_count),
            query_visits: telemetry.as_ref().map_or(0, |t| t.query_visits),
            root_score: score,
            posterior_mass: probs[idx],
            cumulative_mass: cumulative,
            selected: true,
        });
        if cumulative >= target_mass {
            break;
        }
    }

    let selected_set = selected
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    for (idx, score) in segments
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.segment.lsh().ok().map(|lsh| (i, lsh.root_score(query))))
    {
        if selected_set.contains(&idx) {
            continue;
        }
        let telemetry = segments[idx].telemetry_snapshot().ok();
        segment_explanations.push(RoutingSegmentExplanation {
            segment_id: segments[idx].segment.id(),
            depth: segments[idx].segment.depth(),
            vector_count: telemetry
                .as_ref()
                .map_or(segments[idx].segment.len(), |t| t.vector_count),
            query_visits: telemetry.as_ref().map_or(0, |t| t.query_visits),
            root_score: score,
            posterior_mass: probs[idx],
            cumulative_mass: cumulative,
            selected: false,
        });
    }

    RoutingExplanation {
        delta,
        target_mass,
        selected_segment_indices: selected,
        segments: segment_explanations,
    }
}

fn select_compaction_pair(segments: &[PublishedSegment], depth_cap: u8) -> Option<(usize, usize)> {
    let mut best_pair = None;
    let mut best_gain = f64::NEG_INFINITY;

    for i in 0..segments.len() {
        if segments[i].segment.depth() >= depth_cap {
            continue;
        }
        for j in (i + 1)..segments.len() {
            if segments[j].segment.depth() >= depth_cap {
                continue;
            }

            let (Ok(left_stats), Ok(right_stats)) = (
                segments[i].segment.root_stats(),
                segments[j].segment.root_stats(),
            ) else {
                continue;
            };
            let gain = bic_gain(left_stats, right_stats);
            if gain > 0.0 && gain > best_gain {
                best_gain = gain;
                best_pair = Some((i, j));
            }
        }
    }

    best_pair
}

fn centroid_distance_l2(left: &SegmentTelemetry, right: &SegmentTelemetry) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..left.centroid.len().min(right.centroid.len()) {
        let diff = left.centroid[i] - right.centroid[i];
        sum = diff.mul_add(diff, sum);
    }
    sum.sqrt()
}

fn bic_gain(a: &[NigStats], b: &[NigStats]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NEG_INFINITY;
    }

    let bic_a = root_bic(a);
    let bic_b = root_bic(b);

    let merged = a
        .iter()
        .zip(b.iter())
        .map(|(sa, sb)| NigStats::combined(sa, sb))
        .collect::<Vec<_>>();

    let bic_m = root_bic(&merged);
    bic_m - (bic_a + bic_b)
}

fn root_bic(stats: &[NigStats]) -> f64 {
    let mut ll = 0.0f64;
    let mut n_total = 0.0f64;

    for s in stats {
        if s.n > 0.0 {
            n_total += s.n;
            let var = (s.m2 / s.n.max(1.0)).max(1e-12);
            ll += -0.5 * s.n * (2.0 * std::f64::consts::PI * var).ln();
        }
    }

    let k = (stats.len() * 4) as f64;
    ll - 0.5 * k * n_total.max(2.0).ln()
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(id: u64, dim: usize, center: f32, depth: u8) -> Segment<SealedState> {
        let mut active = Segment::new_active(id, dim);
        for i in 0..32u32 {
            let vec = vec![center + (i as f32 * 0.0001); dim];
            active.insert(&vec, id as u32 * 1000 + i).unwrap();
        }
        active
            .seal(8, 4, 8, 0, 4, 512, 2048, 1.5, 42 + id)
            .unwrap()
            .with_depth(depth)
    }

    fn publish_segments(segments: Vec<Segment<SealedState>>) -> Vec<PublishedSegment> {
        segments
            .into_iter()
            .map(PublishedSegment::new)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    #[test]
    fn routing_mass_budget_is_delta_sensitive() {
        let dim = 16;
        let segments = vec![
            make_segment(1, dim, 0.1, 0),
            make_segment(2, dim, 1.0, 0),
            make_segment(3, dim, 2.0, 0),
            make_segment(4, dim, 3.0, 0),
        ];

        let query = vec![0.1; dim];
        let strict = select_segments_by_routing_mass(&segments, &query, 0.01);
        let loose = select_segments_by_routing_mass(&segments, &query, 0.5);

        assert!(!strict.is_empty());
        assert!(!loose.is_empty());
        assert!(strict.len() >= loose.len());
    }

    #[test]
    fn compaction_pair_respects_depth_cap() {
        let dim = 8;
        let segments = publish_segments(vec![
            make_segment(1, dim, 0.0, 3),
            make_segment(2, dim, 0.1, 3),
            make_segment(3, dim, 0.2, 2),
        ]);

        let pair = select_compaction_pair(&segments, 3);
        assert!(pair.is_none());
    }
}
