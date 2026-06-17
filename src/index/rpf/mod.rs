use rand::{self, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::{
    collections::{HashSet, VecDeque},
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use crate::{
    DendraError,
    io::{read_u32_le, read_u64_le},
};

const FOREST_MAGIC: &[u8; 4] = b"RPFI";
const FOREST_VERSION: u32 = 1;

mod node;
mod tree;

pub(crate) use node::Node;
pub use tree::{Candidate, Tree};

pub struct ForestBuilder {
    dim: usize,
    leaf_size: usize,
    num_trees: usize,
    seed: u64,
}

impl ForestBuilder {
    pub fn new(dim: usize, leaf_size: usize, num_trees: usize, seed: u64) -> Self {
        Self {
            dim,
            leaf_size,
            num_trees,
            seed,
        }
    }

    pub fn build(&self, vectors: &[f32], ids: &[u32]) -> Result<Forest, DendraError> {
        let config = ForestConfig {
            seed: self.seed,
            leaf_size: self.leaf_size,
            num_trees: self.num_trees,
            dim: self.dim,
        };
        let mut forest = Forest::new(config);
        // let mut rng = StdRng::seed_from_u64(self.seed);
        forest.build(vectors, ids, self.dim)?;
        Ok(forest)
    }
}

pub struct ForestConfig {
    pub seed: u64,
    pub leaf_size: usize,
    pub num_trees: usize,
    pub dim: usize,
}

pub struct Forest {
    trees: Vec<Tree>,
    config: ForestConfig,
}

impl Forest {
    pub fn new(config: ForestConfig) -> Self {
        Self {
            trees: Vec::with_capacity(config.num_trees),
            config,
        }
    }

    pub fn dim(&self) -> usize {
        self.config.dim
    }

    pub fn leaf_size(&self) -> usize {
        self.config.leaf_size
    }

    pub fn save(&self, path: &Path) -> Result<(), DendraError> {
        let start = std::time::Instant::now();
        eprintln!(
            "[Forest::save] Saving {} trees to {:?}",
            self.trees.len(),
            path
        );
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(FOREST_MAGIC)?;
        w.write_all(&FOREST_VERSION.to_le_bytes())?;
        w.write_all(&(self.config.dim as u32).to_le_bytes())?;
        w.write_all(&(self.trees.len() as u32).to_le_bytes())?;
        w.write_all(&(self.config.leaf_size as u32).to_le_bytes())?;
        w.write_all(&self.config.seed.to_le_bytes())?;

        for (i, tree) in self.trees.iter().enumerate() {
            let tree_start = std::time::Instant::now();
            tree.write(&mut w)?;
            if (i + 1) % 2 == 0 {
                eprintln!(
                    "  tree {}: {:.1}ms",
                    i,
                    tree_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        w.flush()?;
        eprintln!(
            "[Forest::save] Complete in {:.3}s",
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, DendraError> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != FOREST_MAGIC {
            return Err(DendraError::InvalidHeader {
                expected: String::from_utf8_lossy(FOREST_MAGIC).to_string(),
                received: String::from_utf8_lossy(&magic).to_string(),
            });
        }
        let version = read_u32_le(&mut r)?;
        if version != FOREST_VERSION {
            return Err(DendraError::InvalidHeader {
                expected: FOREST_VERSION.to_string(),
                received: version.to_string(),
            });
        }
        let dim = read_u32_le(&mut r)? as usize;
        let num_trees = read_u32_le(&mut r)? as usize;
        let leaf_size = read_u32_le(&mut r)? as usize;
        let seed = read_u64_le(&mut r)?;
        let config = ForestConfig {
            seed,
            leaf_size,
            num_trees,
            dim,
        };
        let mut forest = Self::new(config);
        for _ in 0..num_trees {
            let tree = Tree::read(&mut r, dim, leaf_size)?;
            forest.trees.push(tree);
        }
        Ok(forest)
    }

    pub fn len(&self) -> usize {
        self.trees.len()
    }

    pub fn tree(&self, index: usize) -> Option<&Tree> {
        self.trees.get(index)
    }

    pub fn builder(dim: usize, leaf_size: usize, num_trees: usize, seed: u64) -> ForestBuilder {
        ForestBuilder::new(dim, leaf_size, num_trees, seed)
    }

    pub fn build(&mut self, vectors: &[f32], ids: &[u32], dim: usize) -> Result<(), DendraError> {
        self.trees.clear();
        let leaf_size = self.config.leaf_size;
        let num_trees = self.config.num_trees;
        let seed = self.config.seed;

        // Build trees in parallel; each thread gets its own seeded RNG
        let trees: Vec<_> = (0..num_trees)
            .into_par_iter()
            .map(|i| {
                let mut thread_rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
                Tree::builder(dim, leaf_size).build(vectors, ids, &mut thread_rng)
            })
            .collect::<Result<_, _>>()?;

        self.trees = trees;
        Ok(())
    }
    pub fn generate_candidates(
        &self,
        vector: &[f32],
        max_candidates: usize,
        candidates: &mut Vec<Candidate>,
        queue: &mut VecDeque<usize>,
    ) -> usize {
        candidates.clear();
        if self.trees.is_empty() || max_candidates == 0 {
            return 0;
        }
        let max_candidates_per_tree = max_candidates.div_ceil(self.trees.len());

        let mut seen: HashSet<(usize, usize, usize)> = HashSet::with_capacity(max_candidates);
        let mut candidates_per_tree: Vec<Candidate> = Vec::with_capacity(max_candidates_per_tree);

        for (index, tree) in self.trees.iter().enumerate() {
            candidates_per_tree.clear();
            let _ = tree.generate_candidates(
                vector,
                max_candidates_per_tree,
                index,
                queue,
                &mut candidates_per_tree,
            );
            let remaining_capacity = max_candidates.saturating_sub(candidates.len());
            for candidate in candidates_per_tree.drain(..).take(remaining_capacity) {
                let key = (candidate.tree_index, candidate.start, candidate.end);
                if seen.insert(key) {
                    candidates.push(candidate);
                }
            }
            if candidates.len() >= max_candidates {
                break;
            }
        }
        candidates.len()
    }

    pub fn size(&self) -> usize {
        self.trees.iter().map(|tree| tree.nodes.len()).sum()
    }
}
