mod ivf;
mod rpf;
mod vector;

use std::{cell::RefCell, collections::VecDeque, path::Path};

use crate::DendraError;

pub use ivf::{Candidate as IvfCandidate, IvfIndex};
pub use rpf::{
    Candidate as RpfCandidate, Forest as RpfIndex, TreeBuildPolicy as RpfTreeBuildPolicy,
};
pub use vector::VectorIndex;

thread_local! {
    static RPF_CANDIDATE_SCRATCH: RefCell<Vec<RpfCandidate>> = const { RefCell::new(Vec::new()) };
    static IVF_CANDIDATE_SCRATCH: RefCell<Vec<IvfCandidate>> = const { RefCell::new(Vec::new()) };
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum IndexCandidate {
    Rpf(RpfCandidate),
    Ivf(IvfCandidate),
}

/// Segment-level ANN index abstraction.
///
/// This keeps `Segment` decoupled from a concrete backend implementation
/// while preserving the current RPF behavior.
pub trait SegmentIndex: Send + Sync {
    fn search(
        &self,
        vector: &[f32],
        max_candidates: usize,
        candidates: &mut Vec<IndexCandidate>,
        queue: &mut VecDeque<usize>,
    );

    fn candidate_lookups<'a>(
        &'a self,
        candidate: &IndexCandidate,
    ) -> Result<&'a [u32], DendraError>;

    fn save(&self, path: &Path) -> Result<(), DendraError>;

    fn len(&self) -> usize;
}

impl SegmentIndex for RpfIndex {
    fn search(
        &self,
        vector: &[f32],
        max_candidates: usize,
        candidates: &mut Vec<IndexCandidate>,
        queue: &mut VecDeque<usize>,
    ) {
        RPF_CANDIDATE_SCRATCH.with(|scratch| {
            let mut raw = scratch.borrow_mut();
            raw.clear();
            self.generate_candidates(vector, max_candidates, &mut raw, queue);
            candidates.reserve(raw.len());
            candidates.extend(raw.drain(..).map(IndexCandidate::Rpf));
        });
    }

    fn candidate_lookups<'a>(
        &'a self,
        candidate: &IndexCandidate,
    ) -> Result<&'a [u32], DendraError> {
        let candidate = match candidate {
            IndexCandidate::Rpf(candidate) => candidate,
            _ => {
                return Err(DendraError::UnsupportedOperation(
                    "candidate kind does not match index backend".to_string(),
                ));
            }
        };

        let tree = self
            .tree(candidate.tree_index)
            .ok_or(DendraError::InvalidTreeIndex {
                index: candidate.tree_index,
                max: self.len(),
            })?;

        if candidate.end > tree.look_up.len() {
            return Err(DendraError::IndexOutOfBounds {
                index: candidate.end,
                length: tree.look_up.len(),
            });
        }

        Ok(&tree.look_up[candidate.start..candidate.end])
    }

    fn save(&self, path: &Path) -> Result<(), DendraError> {
        self.save(path)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl SegmentIndex for IvfIndex {
    fn search(
        &self,
        vector: &[f32],
        max_candidates: usize,
        candidates: &mut Vec<IndexCandidate>,
        queue: &mut VecDeque<usize>,
    ) {
        candidates.clear();
        IVF_CANDIDATE_SCRATCH.with(|scratch| {
            let mut raw = scratch.borrow_mut();
            raw.clear();
            self.generate_candidates(vector, max_candidates, &mut raw, queue);
            candidates.reserve(raw.len());
            candidates.extend(raw.drain(..).map(IndexCandidate::Ivf));
        });
    }

    fn candidate_lookups<'a>(
        &'a self,
        candidate: &IndexCandidate,
    ) -> Result<&'a [u32], DendraError> {
        match candidate {
            IndexCandidate::Ivf(candidate) => self.candidate_lookups(candidate),
            _ => Err(DendraError::UnsupportedOperation(
                "candidate kind does not match index backend".to_string(),
            )),
        }
    }

    fn save(&self, path: &Path) -> Result<(), DendraError> {
        self.save(path)
    }

    fn len(&self) -> usize {
        self.len()
    }
}
