use crate::core::{Engine, EngineConfig};
use crate::err::DendraError;
use crate::index::VectorIndex;
use crate::query::Query;
use std::path::{Path, PathBuf};

pub type VectorDBConfig = EngineConfig;

pub struct VectorDB {
    engine: Engine,
    index: VectorIndex,
}

impl VectorDB {
    pub fn new(dir: PathBuf, config: VectorDBConfig) -> Self {
        Self {
            engine: Engine::new(dir, config),
            index: VectorIndex::new(),
        }
    }

    pub fn insert(&mut self, vector: &[f32], id: u32) -> Result<(), DendraError> {
        self.engine.insert(vector, id)
    }

    pub fn flush(&mut self) -> Result<(), DendraError> {
        self.engine.flush()
    }

    pub fn save(&mut self) -> Result<(), DendraError> {
        self.engine.save()
    }

    pub fn load(dir: &Path) -> Result<Self, DendraError> {
        Ok(Self {
            engine: Engine::load(dir)?,
            index: VectorIndex::new(),
        })
    }

    pub fn query(&self, query: &Query, results: &mut Vec<(u32, f32)>) -> Result<bool, DendraError> {
        self.index.query(self.engine.segments(), query, results)
    }
}
