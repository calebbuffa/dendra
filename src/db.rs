use crate::err::FvdbError;
use crate::io::read_u8_le;
use crate::query::Query;
use crate::segment::{Segment, SegmentBuilder, SegmentQueryContext};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread;

const STORE_MAGIC: &[u8; 4] = b"VSTR";
const STORE_VERSION: u8 = 2; // Bumped for quantization support
const DEFAULT_MAX_MB_PER_SEGMENT: usize = 100; // 100MB
const DEFAULT_ASYNC_SEAL_QUEUE_CAPACITY: usize = 2;

#[derive(Serialize, Deserialize, Clone)]
pub struct VectorDBConfig {
    pub leaf_size: usize,
    pub num_trees: usize,
    pub dimension: usize,
    pub seed: u64,
    pub max_segment_capacity: usize,
    pub async_seal_queue_capacity: usize,
}

impl VectorDBConfig {
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
        }
    }
}

impl Default for VectorDBConfig {
    fn default() -> Self {
        Self {
            leaf_size: 32,
            num_trees: 4,
            dimension: 128,
            seed: 42,
            max_segment_capacity: DEFAULT_MAX_MB_PER_SEGMENT,
            async_seal_queue_capacity: DEFAULT_ASYNC_SEAL_QUEUE_CAPACITY,
        }
    }
}

pub struct VectorDB {
    pub(crate) segments: Vec<Segment>,

    dir: PathBuf,
    segment_builder: SegmentBuilder,
    next_segment_id: u64,
    pub config: VectorDBConfig,
    seal_tx: SyncSender<SealTask>,
    seal_result_rx: Receiver<Result<PathBuf, FvdbError>>,
    in_flight_seals: usize,
}

struct SealTask {
    builder: SegmentBuilder,
    segment_path: PathBuf,
    leaf_size: usize,
    num_trees: usize,
    seed: u64,
    dimension: usize,
}

#[derive(Debug)]
struct Scored {
    id: u32,
    distance: f32,
}

// BinaryHeap is a max-heap and we want the heap's top to be the worse (largest) distance so we
// can evict the worst when mainitng size k
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

impl VectorDB {
    fn spawn_seal_worker(
        queue_capacity: usize,
    ) -> (SyncSender<SealTask>, Receiver<Result<PathBuf, FvdbError>>) {
        let (seal_tx, seal_rx) = mpsc::sync_channel::<SealTask>(queue_capacity);
        let (result_tx, result_rx) = mpsc::channel::<Result<PathBuf, FvdbError>>();

        thread::spawn(move || {
            while let Ok(task) = seal_rx.recv() {
                let result = (|| {
                    let mut builder = task.builder;
                    let segment = builder.build(task.leaf_size, task.num_trees, task.seed)?;
                    let _ = segment.flush(&task.segment_path)?;
                    Ok(task.segment_path)
                })();

                if result_tx.send(result).is_err() {
                    break;
                }
            }
        });

        (seal_tx, result_rx)
    }

    fn enqueue_builder_for_seal(&mut self, builder: SegmentBuilder) -> Result<(), FvdbError> {
        let segment_path = self.dir.join(format!("segment{}", self.next_segment_id));
        self.next_segment_id += 1;

        let task = SealTask {
            builder,
            segment_path,
            leaf_size: self.config.leaf_size,
            num_trees: self.config.num_trees,
            seed: self.config.seed,
            dimension: self.config.dimension,
        };

        self.seal_tx
            .send(task)
            .map_err(|_| FvdbError::UnsupportedOperation("seal worker unavailable".to_string()))?;

        self.in_flight_seals += 1;
        Ok(())
    }

    fn integrate_seal_result(
        &mut self,
        result: Result<PathBuf, FvdbError>,
    ) -> Result<(), FvdbError> {
        self.in_flight_seals = self.in_flight_seals.saturating_sub(1);
        let segment_path = result?;
        let segment = Segment::open(&segment_path)?;
        self.segments.push(segment);
        Ok(())
    }

    fn drain_completed_seals(&mut self) -> Result<(), FvdbError> {
        loop {
            match self.seal_result_rx.try_recv() {
                Ok(result) => self.integrate_seal_result(result)?,
                Err(TryRecvError::Empty) => return Ok(()),
                Err(TryRecvError::Disconnected) => {
                    return Err(FvdbError::UnsupportedOperation(
                        "seal worker disconnected".to_string(),
                    ));
                }
            }
        }
    }

    fn await_all_seals(&mut self) -> Result<(), FvdbError> {
        while self.in_flight_seals > 0 {
            let result = self.seal_result_rx.recv().map_err(|_| {
                FvdbError::UnsupportedOperation("seal worker disconnected".to_string())
            })?;
            self.integrate_seal_result(result)?;
        }
        Ok(())
    }

    pub fn new(dir: PathBuf, config: VectorDBConfig) -> Self {
        let segment_builder = SegmentBuilder::new(config.dimension, config.max_segment_capacity);
        let (seal_tx, seal_result_rx) = Self::spawn_seal_worker(config.async_seal_queue_capacity);

        Self {
            segments: Vec::new(),
            dir,
            segment_builder,
            next_segment_id: 0,
            config,
            seal_tx,
            seal_result_rx,
            in_flight_seals: 0,
        }
    }

    pub fn insert(&mut self, vector: &[f32], id: u32) -> Result<(), FvdbError> {
        if !self.segment_builder.try_insert(vector, id)? {
            let old_builder = std::mem::replace(
                &mut self.segment_builder,
                SegmentBuilder::new(self.config.dimension, self.config.max_segment_capacity),
            );
            self.enqueue_builder_for_seal(old_builder)?;
            self.drain_completed_seals()?;
            self.segment_builder.try_insert(vector, id)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), FvdbError> {
        if self.segment_builder.current_capacity > 0 {
            let old_builder = std::mem::replace(
                &mut self.segment_builder,
                SegmentBuilder::new(self.config.dimension, self.config.max_segment_capacity),
            );
            self.enqueue_builder_for_seal(old_builder)?;
        }
        self.await_all_seals()?;
        Ok(())
    }

    pub fn save(&mut self) -> Result<(), FvdbError> {
        self.flush()?;

        fs::create_dir_all(&self.dir)?;
        let meta_path = self.dir.join("store.bin");
        let mut w = BufWriter::new(File::create(meta_path)?);

        // Write magic and version
        w.write_all(STORE_MAGIC)?;
        w.write_all(&STORE_VERSION.to_le_bytes())?;

        // Serialize config and metadata together
        let metadata = (
            self.config.dimension as u32,
            self.config.leaf_size as u32,
            self.config.num_trees as u32,
            self.config.seed,
            self.config.max_segment_capacity as u64,
            self.next_segment_id,
        );
        let config = bincode::config::standard();
        let encoded = bincode::encode_to_vec(&metadata, config)
            .map_err(|e| FvdbError::SerializationError(e.to_string()))?;
        w.write_all(&encoded)?;

        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self, FvdbError> {
        let meta_path = dir.join("store.bin");
        let mut r = BufReader::new(File::open(meta_path)?);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != STORE_MAGIC {
            return Err(FvdbError::InvalidHeader {
                expected: String::from_utf8_lossy(STORE_MAGIC).to_string(),
                received: String::from_utf8_lossy(&magic).to_string(),
            });
        }

        let version = read_u8_le(&mut r)?;
        if version != STORE_VERSION {
            return Err(FvdbError::InvalidHeader {
                expected: STORE_VERSION.to_string(),
                received: version.to_string(),
            });
        }

        // Deserialize metadata with bincode 2.0
        let mut metadata_bytes = Vec::new();
        r.read_to_end(&mut metadata_bytes)?;
        let config = bincode::config::standard();
        let (
            dimension,
            leaf_size,
            num_trees,
            seed,
            max_segment_capacity,
            next_segment_id,
        ): (u32, u32, u32, u64, u64, u64) =
            bincode::decode_from_slice(&metadata_bytes, config)
                .map(|(v, _)| v)
                .map_err(|e| FvdbError::SerializationError(e.to_string()))?;

        let dimension = dimension as usize;
        let leaf_size = leaf_size as usize;
        let num_trees = num_trees as usize;
        let max_segment_capacity = max_segment_capacity as usize;

        let config = VectorDBConfig {
            leaf_size,
            num_trees,
            dimension,
            seed,
            max_segment_capacity,
            async_seal_queue_capacity: DEFAULT_ASYNC_SEAL_QUEUE_CAPACITY,
        };

        // Create store with quantizer initialized
        let mut store = Self::new(dir.to_path_buf(), config);
        store.next_segment_id = next_segment_id;

        // Load segments from disk
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

        for (_, path) in found {
            let segment = Segment::open(&path)?;
            if segment.dim != store.config.dimension {
                return Err(FvdbError::InvalidVectorDimension {
                    expected: store.config.dimension,
                    received: segment.dim,
                });
            }
            store.segments.push(segment);
        }

        Ok(store)
    }

    pub fn query(&self, query: &Query, results: &mut Vec<(u32, f32)>) -> Result<bool, FvdbError> {
        let candidates_per_segment = std::cmp::max(10, query.k * 2);
        let mut context = SegmentQueryContext::new(candidates_per_segment);

        for segment in &self.segments {
            context.candidates.clear();
            context.queue.clear();

            let _ = segment.query(query, candidates_per_segment, query.metric, &mut context)?;
        }

        let k = query.k;
        if context.best_map.is_empty() {
            return Ok(false); // no candidates found, return empty results
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
