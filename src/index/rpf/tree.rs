use rand::{self, Rng};
use std::{
    collections::VecDeque,
    io::{Read, Write},
};

use crate::{
    DendraError,
    index::rpf::Node,
    io::read_u32_le,
    math::{dot, random_unit_vector},
};

const MAX_SPLIT_TRIES: usize = 5;

pub struct TreeBuilder {
    dim: usize,
    leaf_size: usize,
}

impl TreeBuilder {
    pub fn new(dim: usize, leaf_size: usize) -> Self {
        Self { dim, leaf_size }
    }

    /// Build modifies `positions` in-place to produce contiguous ranges for leaves.
    /// `position` is a buffer of positions (0..N-1) mapping to the flat `vectors` buffer.
    pub fn build(
        &self,
        vectors: &[f32],
        ids: &[u32],
        rng: &mut impl Rng,
    ) -> Result<Tree, DendraError> {
        let mut tree = Tree::new(self.dim, self.leaf_size);
        tree.build(vectors, ids, rng)?;
        Ok(tree)
    }
}

fn compute_median_split(
    vectors: &[f32],
    positions: &mut [u32], // working permutation buffer
    dp_pairs: &mut Vec<(f32, u32)>,
    projection: &[f32],
    start: usize,
    end: usize,
    dim: usize,
) -> Result<(f32, usize), DendraError> {
    // compute dot products for ids[start..end) and collect (dp, id)
    dp_pairs.clear();
    for i in start..end {
        let id = positions[i];
        let idx = id as usize;
        let off = idx * dim;
        let slice = &vectors[off..off + dim];
        let dp = dot(projection, slice);
        dp_pairs.push((dp, id));
    }

    let len = dp_pairs.len();
    if len == 0 {
        return Err(DendraError::EmptyNode);
    }

    let mid = dp_pairs.len() / 2;
    // nth-element to get median pivot (linear time)
    dp_pairs.select_nth_unstable_by(mid, |a, b| a.0.total_cmp(&b.0));
    let pivot = dp_pairs[mid].0;

    // count how many < pivot. (Because select_nth_unstable_by does a partition,
    // this will be a contiguous prefix of dp_pairs.)
    let left_count = dp_pairs.iter().filter(|p| p.0 < pivot).count();

    // fallback if degenerate (all or none on one side)
    if left_count == 0 || left_count == dp_pairs.len() {
        return Err(DendraError::DegenerateSplit);
    }

    // Write partitioned ids back into ids[start..end):
    // first those with dp < pivot, then the rest (dp >= pivot).
    let mut wi = start;
    for &(dp, id) in dp_pairs.iter() {
        if dp < pivot {
            positions[wi] = id;
            wi += 1;
        }
    }
    for &(dp, id) in dp_pairs.iter() {
        if dp >= pivot {
            positions[wi] = id;
            wi += 1;
        }
    }
    Ok((pivot, left_count))
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub look_up: Vec<u32>, // tree-local permutation; leaf ranges are contiguous in this permuted order
    dim: usize,
    leaf_size: usize,
}

impl Tree {
    pub fn new(dim: usize, leaf_size: usize) -> Self {
        Self {
            nodes: Vec::new(),
            look_up: Vec::new(),
            dim,
            leaf_size,
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<(), DendraError> {
        let start = std::time::Instant::now();
        w.write_all(&(self.nodes.len() as u32).to_le_bytes())?;

        let nodes_start = std::time::Instant::now();
        for node in self.nodes.iter() {
            node.write(w)?;
        }
        eprintln!(
            "    write {} nodes: {:.1}ms",
            self.nodes.len(),
            nodes_start.elapsed().as_secs_f64() * 1000.0
        );

        let lookup_start = std::time::Instant::now();
        w.write_all(&(self.look_up.len() as u32).to_le_bytes())?;
        for &id in self.look_up.iter() {
            w.write_all(&id.to_le_bytes())?;
        }
        eprintln!(
            "    write {} lookups: {:.1}ms",
            self.look_up.len(),
            lookup_start.elapsed().as_secs_f64() * 1000.0
        );
        eprintln!(
            "    tree total: {:.1}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R, dim: usize, leaf_size: usize) -> Result<Self, DendraError> {
        let num_nodes = read_u32_le(r)? as usize;
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let node = Node::read(r)?;
            nodes.push(node);
        }
        let lookup_len = read_u32_le(r)? as usize;
        let mut look_up = Vec::with_capacity(lookup_len);
        for _ in 0..lookup_len {
            let id = read_u32_le(r)?;
            look_up.push(id);
        }
        Ok(Tree {
            nodes,
            look_up,
            dim,
            leaf_size,
        })
    }

    pub fn builder(dim: usize, leaf_size: usize) -> TreeBuilder {
        TreeBuilder::new(dim, leaf_size)
    }

    /// vector must be normalized before passing in
    pub fn generate_candidates(
        &self,
        vector: &[f32],
        max_candidates: usize,
        index: usize,
        queue: &mut VecDeque<usize>,
        candidates: &mut Vec<Candidate>,
    ) -> usize {
        queue.clear();
        if self.nodes.is_empty() {
            return 0;
        }
        let mut total_candidates = 0;
        queue.push_back(0); // start with root

        while let Some(node_idx) = queue.pop_front() {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                let start = node.start;
                let mut end = node.end;
                let n = end.saturating_sub(start);
                if n > max_candidates {
                    end = start + max_candidates;
                }
                candidates.push(Candidate {
                    start,
                    end,
                    tree_index: index,
                });

                total_candidates += end - start;

                if candidates.len() >= max_candidates {
                    break;
                }
            } else {
                let dot_product = dot(&node.projection, vector);
                if dot_product < node.threshold {
                    queue.push_back(node.left as usize);
                } else {
                    queue.push_back(node.right as usize);
                }
            }
        }
        total_candidates
    }

    pub fn build(
        &mut self,
        vectors: &[f32],
        ids: &[u32],
        rng: &mut impl Rng,
    ) -> Result<usize, DendraError> {
        let n = ids.len() as u32;
        self.look_up = (0u32..n).collect(); // initialize to identity permutation

        self.nodes.clear();
        let n = ids.len();

        self.nodes.push(Node::default()); // placeholder for root
        if n == 0 {
            self.nodes[0].is_leaf = true;
            self.nodes[0].start = 0;
            self.nodes[0].end = 0;
            return Ok(0);
        }

        // heuristic reserve: roughly 2*(n/leaf) nodes
        self.nodes.reserve(2 * (n / (self.leaf_size.max(1) + 1)));

        // BFS queue of (node_idx, start, end)
        let mut queue = VecDeque::new();
        queue.push_back((0usize, 0usize, n));

        // dot product buffer reused across nodes
        let mut dp_pairs: Vec<(f32, u32)> = Vec::with_capacity(n);

        while let Some((node_idx, start, end)) = queue.pop_front() {
            let len = end - start;
            if len <= self.leaf_size {
                // make leaf
                self.nodes[node_idx] = Node {
                    left: 0,
                    right: 0,
                    projection: Vec::new(),
                    threshold: 0.0,
                    is_leaf: true,
                    start,
                    end,
                };
                continue;
            }

            let mut projection = random_unit_vector(self.dim, rng)?;

            // Try to compute a median split. If we get a degenerate split (all points on one side),
            // retry with a new random projection up to `MAX_SPLIT_TRIES`. If we still fail, fall back
            // to a balanced split (i.e. split the ids in half)
            let mut split_attempt = 0usize;
            let split_result = loop {
                match compute_median_split(
                    vectors,
                    &mut self.look_up,
                    &mut dp_pairs,
                    &projection,
                    start,
                    end,
                    self.dim,
                ) {
                    Ok((p, lc)) => break Ok((p, lc)),
                    Err(DendraError::DegenerateSplit) => {
                        if split_attempt >= MAX_SPLIT_TRIES {
                            break Err(DendraError::DegenerateSplit);
                        }
                        projection = random_unit_vector(self.dim, rng)?;
                        split_attempt += 1;
                    }
                    Err(e) => break Err(e),
                }
            };

            match split_result {
                Ok((pivot, left_count)) => {
                    let left_end = start + left_count;

                    // create child placeholders and set parent
                    let left_child_idx = self.nodes.len();
                    self.nodes.push(Node::default());
                    let right_child_idx = self.nodes.len();
                    self.nodes.push(Node::default());

                    self.nodes[node_idx].left = left_child_idx as u32;
                    self.nodes[node_idx].right = right_child_idx as u32;
                    self.nodes[node_idx].projection = projection;
                    self.nodes[node_idx].threshold = pivot;
                    self.nodes[node_idx].is_leaf = false;
                    self.nodes[node_idx].start = 0;
                    self.nodes[node_idx].end = 0;

                    // enqueue children
                    queue.push_back((left_child_idx, start, left_end));
                    queue.push_back((right_child_idx, left_end, end));
                }
                Err(DendraError::DegenerateSplit) => {
                    // Couldn't find a safe split: make this node a leaf
                    self.nodes[node_idx] = Node::leaf(start, end);
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(0)
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Candidate {
    /// half-open [start, end) range of positions (indices into the segment's vectors/ids)
    pub start: usize,
    pub end: usize,
    pub tree_index: usize, // index of the tree that produced this candidate
}
