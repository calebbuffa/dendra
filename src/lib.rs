mod core;
mod err;
mod index;
mod math;
mod query;
mod segment;
mod storage;

pub use core::EngineConfig as VectorDBConfig;
pub use core::{
    CompactionExplanation, QueryScratch, RoutingExplanation, RoutingSegmentExplanation,
};
pub use err::EngramError;
pub use math::{MetricFn, cosine_distance, l2_distance};
pub use query::Query;
pub use segment::SegmentTelemetry;

use crate::core::Engine;
use std::path::{Path, PathBuf};

pub struct VectorDB {
    engine: Engine,
}

impl VectorDB {
    pub fn new(dir: PathBuf, config: VectorDBConfig) -> Self {
        Self {
            engine: Engine::new(dir, config),
        }
    }

    pub fn insert(&mut self, vector: &[f32], id: u32) -> Result<(), EngramError> {
        self.engine.insert(vector, id)
    }

    pub fn flush(&mut self) -> Result<(), EngramError> {
        self.engine.flush()
    }

    pub fn save(&mut self) -> Result<(), EngramError> {
        self.engine.save()
    }

    pub fn load(dir: &Path, config: VectorDBConfig) -> Result<Self, EngramError> {
        Ok(Self {
            engine: Engine::load(dir, config)?,
        })
    }

    pub fn num_sealed_segments(&self) -> usize {
        self.engine.num_sealed_segments()
    }

    pub fn config(&self) -> &VectorDBConfig {
        self.engine.config()
    }

    pub fn sealed_segment_summaries(&self) -> Vec<SegmentTelemetry> {
        self.engine.sealed_segment_summaries()
    }

    pub fn explain_next_compaction(&self) -> Result<Option<CompactionExplanation>, EngramError> {
        self.engine.explain_next_compaction()
    }

    pub fn explain_query_routing(&self, query: &Query) -> RoutingExplanation {
        self.engine.explain_query_routing(query)
    }

    pub fn query(
        &self,
        query: &Query,
        scratch: &mut QueryScratch,
        results: &mut Vec<(u32, f32)>,
    ) -> Result<bool, EngramError> {
        self.engine.query(query, scratch, results)
    }

    pub fn query_raw(
        &self,
        vector: &[f32],
        k: usize,
        metric: Option<MetricFn>,
        threshold: Option<f32>,
        delta: f32,
        scratch: &mut QueryScratch,
        results: &mut Vec<(u32, f32)>,
    ) -> Result<bool, EngramError> {
        self.engine
            .query_raw(vector, k, metric, threshold, delta, scratch, results)
    }
}
