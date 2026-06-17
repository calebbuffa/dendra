use log::debug;
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

#[derive(Clone, Debug)]
pub struct TreeBuildPolicy {
    pub sparse_dims: usize,
    pub sparse_min_points: usize,
    pub sample_size: usize,
    pub sampled_min_points: usize,
    pub sampled_min_sample_mult: usize,
    pub min_split_balance_ratio: f32,
    pub max_split_retries: usize,
}

impl Default for TreeBuildPolicy {
    fn default() -> Self {
        Self {
            sparse_dims: 24,
            sparse_min_points: 2048,
            sample_size: 256,
            sampled_min_points: 1024,
            sampled_min_sample_mult: 8,
            min_split_balance_ratio: 0.20,
            max_split_retries: 4,
        }
    }
}

struct NodeBuildParams {
    use_sparse: bool,
    sparse_dims: usize,
    use_sampled_pivot: bool,
    sample_size: usize,
    min_split_balance_ratio: f32,
    max_retries: usize,
}

fn compute_node_params(
    policy: &TreeBuildPolicy,
    node_points: usize,
    dim: usize,
) -> NodeBuildParams {
    let sparse_dims = policy.sparse_dims.clamp(1, dim);
    let use_sparse = node_points >= policy.sparse_min_points && sparse_dims < dim;

    let sample = policy.sample_size.clamp(1, node_points.max(1));
    let use_sampled_pivot = node_points >= policy.sampled_min_points
        && node_points >= sample.saturating_mul(policy.sampled_min_sample_mult);
    let max_retries = policy.max_split_retries;

    NodeBuildParams {
        use_sparse,
        sparse_dims,
        use_sampled_pivot,
        sample_size: sample,
        min_split_balance_ratio: policy.min_split_balance_ratio,
        max_retries,
    }
}

#[derive(Default)]
struct SplitPhaseTiming {
    calls: usize,
    points: usize,
    dot_time: f64,
    select_time: f64,
    count_time: f64,
    write_time: f64,
}

pub struct TreeBuilder {
    dim: usize,
    leaf_size: usize,
    policy: TreeBuildPolicy,
}

impl TreeBuilder {
    pub fn new(dim: usize, leaf_size: usize) -> Self {
        Self {
            dim,
            leaf_size,
            policy: TreeBuildPolicy::default(),
        }
    }

    pub fn with_policy(mut self, policy: TreeBuildPolicy) -> Self {
        self.policy = policy;
        self
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
        tree.build(vectors, ids, rng, &self.policy)?;
        Ok(tree)
    }
}

fn compute_median_split(
    vectors: &[f32],
    positions: &mut [u32], // working permutation buffer
    tmp_ids: &mut [u32],   // temporary partition buffer
    dp_pairs: &mut Vec<(f32, u32)>,
    projection: &[f32],
    selected_dims: Option<&[usize]>,
    use_sampled_pivot: bool,
    sample_size: usize,
    min_split_balance_ratio: f32,
    start: usize,
    end: usize,
    dim: usize,
    timing: &mut SplitPhaseTiming,
) -> Result<(f32, usize), DendraError> {
    let len = end.saturating_sub(start);
    if len == 0 {
        return Err(DendraError::EmptyNode);
    }
    timing.calls += 1;
    timing.points += len;

    // Sampled-pivot fast path when sample_size is smaller than node size.
    if use_sampled_pivot && sample_size > 0 && sample_size < len {
        let select_start = std::time::Instant::now();
        let sample_count = sample_size.min(len);
        let step = len / sample_count;
        let mut sample_vals: Vec<f32> = Vec::with_capacity(sample_count);
        for s in 0..sample_count {
            let pos = start + s * step;
            let id = positions[pos];
            let idx = id as usize;
            let off = idx * dim;
            let slice = &vectors[off..off + dim];
            sample_vals.push(dot_for_split(projection, slice, selected_dims));
        }
        let smid = sample_vals.len() / 2;
        sample_vals.select_nth_unstable_by(smid, |a, b| a.total_cmp(b));
        let sampled_pivot = sample_vals[smid];
        timing.select_time += select_start.elapsed().as_secs_f64();

        let dot_start = std::time::Instant::now();
        let mut left_w = start;
        let mut right_w = end;
        for i in start..end {
            let id = positions[i];
            let idx = id as usize;
            let off = idx * dim;
            let slice = &vectors[off..off + dim];
            let dp = dot_for_split(projection, slice, selected_dims);
            if dp < sampled_pivot {
                tmp_ids[left_w] = id;
                left_w += 1;
            } else {
                right_w -= 1;
                tmp_ids[right_w] = id;
            }
        }
        timing.dot_time += dot_start.elapsed().as_secs_f64();

        let count_start = std::time::Instant::now();
        let left_count = left_w - start;
        timing.count_time += count_start.elapsed().as_secs_f64();

        if left_count != 0 && left_count != len {
            let left_ratio = left_count as f32 / len as f32;
            let is_balanced = left_ratio >= min_split_balance_ratio
                && (1.0 - left_ratio) >= min_split_balance_ratio;
            if !is_balanced {
                // Skewed sampled split: fall back to exact median path.
            } else {
                let write_start = std::time::Instant::now();
                positions[start..end].copy_from_slice(&tmp_ids[start..end]);
                timing.write_time += write_start.elapsed().as_secs_f64();
                return Ok((sampled_pivot, left_count));
            }
        }
        // Degenerate sampled split; fall through to exact median fallback.
    }

    // Exact path for smaller nodes or sampled-pivot degeneracy fallback.
    let dot_start = std::time::Instant::now();
    dp_pairs.clear();
    for i in start..end {
        let id = positions[i];
        let idx = id as usize;
        let off = idx * dim;
        let slice = &vectors[off..off + dim];
        let dp = dot_for_split(projection, slice, selected_dims);
        dp_pairs.push((dp, id));
    }
    timing.dot_time += dot_start.elapsed().as_secs_f64();

    let select_start = std::time::Instant::now();
    let mid = len / 2;
    dp_pairs.select_nth_unstable_by(mid, |a, b| a.0.total_cmp(&b.0));
    let pivot = dp_pairs[mid].0;
    timing.select_time += select_start.elapsed().as_secs_f64();

    let write_start = std::time::Instant::now();
    let mut left_w = start;
    let mut right_w = end;
    for &(dp, id) in dp_pairs.iter() {
        if dp < pivot {
            tmp_ids[left_w] = id;
            left_w += 1;
        } else {
            right_w -= 1;
            tmp_ids[right_w] = id;
        }
    }
    let left_count = left_w - start;
    timing.write_time += write_start.elapsed().as_secs_f64();

    let count_start = std::time::Instant::now();
    let is_degenerate = left_count == 0 || left_count == len;
    timing.count_time += count_start.elapsed().as_secs_f64();
    if is_degenerate {
        return Err(DendraError::DegenerateSplit);
    }

    let copy_start = std::time::Instant::now();
    positions[start..end].copy_from_slice(&tmp_ids[start..end]);
    timing.write_time += copy_start.elapsed().as_secs_f64();
    Ok((pivot, left_count))
}

#[inline(always)]
fn dot_for_split(projection: &[f32], vector: &[f32], selected_dims: Option<&[usize]>) -> f32 {
    if let Some(dims) = selected_dims {
        let mut total = 0.0f32;
        for &d in dims {
            total = projection[d].mul_add(vector[d], total);
        }
        total
    } else {
        dot(projection, vector)
    }
}

fn sparsify_projection_in_place(projection: &mut [f32], keep_dims: usize) -> Vec<usize> {
    let dim = projection.len();
    if keep_dims >= dim {
        return (0..dim).collect();
    }

    let mut dims: Vec<usize> = (0..dim).collect();
    dims.sort_unstable_by(|&a, &b| projection[b].abs().total_cmp(&projection[a].abs()));
    dims.truncate(keep_dims);

    let mut keep_mask = vec![false; dim];
    for &d in &dims {
        keep_mask[d] = true;
    }

    for i in 0..dim {
        if !keep_mask[i] {
            projection[i] = 0.0;
        }
    }

    let mut norm_sq = 0.0f32;
    for &d in &dims {
        norm_sq = projection[d].mul_add(projection[d], norm_sq);
    }
    if norm_sq > 0.0 {
        let inv_norm = norm_sq.sqrt().recip();
        for &d in &dims {
            projection[d] *= inv_norm;
        }
    }

    dims.sort_unstable();
    dims
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
        debug!(
            "    write {} nodes: {:.1}ms",
            self.nodes.len(),
            nodes_start.elapsed().as_secs_f64() * 1000.0
        );

        let lookup_start = std::time::Instant::now();
        w.write_all(&(self.look_up.len() as u32).to_le_bytes())?;
        for &id in self.look_up.iter() {
            w.write_all(&id.to_le_bytes())?;
        }
        debug!(
            "    write {} lookups: {:.1}ms",
            self.look_up.len(),
            lookup_start.elapsed().as_secs_f64() * 1000.0
        );
        debug!(
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
        policy: &TreeBuildPolicy,
    ) -> Result<usize, DendraError> {
        let build_start = std::time::Instant::now();
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

        // DFS stack of (node_idx, start, end) to improve subtree locality.
        let mut stack: Vec<(usize, usize, usize)> = Vec::new();
        stack.push((0usize, 0usize, n));

        // dot product buffer reused across nodes
        let mut dp_pairs: Vec<(f32, u32)> = Vec::with_capacity(n);
        // Temporary ID buffer reused for partition write-back.
        let mut tmp_ids: Vec<u32> = vec![0u32; n];

        let mut internal_nodes = 0usize;
        let mut leaf_nodes = 0usize;
        let mut degenerate_fallbacks = 0usize;
        let mut split_retries_total = 0usize;
        let mut sparse_splits = 0usize;
        let mut projection_time = 0.0f64;
        let mut split_time = 0.0f64;
        let mut split_phase_timing = SplitPhaseTiming::default();

        while let Some((node_idx, start, end)) = stack.pop() {
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
                leaf_nodes += 1;
                continue;
            }

            let projection_start = std::time::Instant::now();
            let mut projection = random_unit_vector(self.dim, rng)?;
            let mut selected_dims = Vec::new();

            let node_params = compute_node_params(policy, len, self.dim);

            if node_params.use_sparse {
                selected_dims =
                    sparsify_projection_in_place(&mut projection, node_params.sparse_dims);
                sparse_splits += 1;
            }
            projection_time += projection_start.elapsed().as_secs_f64();

            // Try to compute a median split. If we get a degenerate split (all points on one side),
            // retry with a new random projection up to `MAX_SPLIT_TRIES`. If we still fail, fall back
            // to a balanced split (i.e. split the ids in half)
            let mut split_attempt = 0usize;
            let split_result = loop {
                let split_start = std::time::Instant::now();
                match compute_median_split(
                    vectors,
                    &mut self.look_up,
                    &mut tmp_ids,
                    &mut dp_pairs,
                    &projection,
                    if selected_dims.is_empty() {
                        None
                    } else {
                        Some(&selected_dims)
                    },
                    node_params.use_sampled_pivot,
                    node_params.sample_size,
                    node_params.min_split_balance_ratio,
                    start,
                    end,
                    self.dim,
                    &mut split_phase_timing,
                ) {
                    Ok((p, lc)) => {
                        split_time += split_start.elapsed().as_secs_f64();
                        break Ok((p, lc));
                    }
                    Err(DendraError::DegenerateSplit) => {
                        split_time += split_start.elapsed().as_secs_f64();
                        if split_attempt >= node_params.max_retries {
                            break Err(DendraError::DegenerateSplit);
                        }
                        let projection_retry_start = std::time::Instant::now();
                        projection = random_unit_vector(self.dim, rng)?;
                        selected_dims.clear();
                        if node_params.use_sparse {
                            selected_dims = sparsify_projection_in_place(
                                &mut projection,
                                node_params.sparse_dims,
                            );
                        }
                        projection_time += projection_retry_start.elapsed().as_secs_f64();
                        split_attempt += 1;
                    }
                    Err(e) => {
                        split_time += split_start.elapsed().as_secs_f64();
                        break Err(e);
                    }
                }
            };
            split_retries_total += split_attempt;

            match split_result {
                Ok((pivot, left_count)) => {
                    internal_nodes += 1;
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

                    // Push right first so left subtree is processed next (LIFO stack).
                    stack.push((right_child_idx, left_end, end));
                    stack.push((left_child_idx, start, left_end));
                }
                Err(DendraError::DegenerateSplit) => {
                    // Couldn't find a safe split: make this node a leaf
                    self.nodes[node_idx] = Node::leaf(start, end);
                    leaf_nodes += 1;
                    degenerate_fallbacks += 1;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        let total_ms = build_start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "[Tree::build] points={} dim={} leaf_size={} nodes={} internal={} leaves={} retries={} degenerate={} sparse_splits={} total={:.1}ms split={:.1}ms proj={:.1}ms split_calls={} split_points={} split_dot={:.1}ms split_select={:.1}ms split_count={:.1}ms split_write={:.1}ms",
            ids.len(),
            self.dim,
            self.leaf_size,
            self.nodes.len(),
            internal_nodes,
            leaf_nodes,
            split_retries_total,
            degenerate_fallbacks,
            sparse_splits,
            total_ms,
            split_time * 1000.0,
            projection_time * 1000.0,
            split_phase_timing.calls,
            split_phase_timing.points,
            split_phase_timing.dot_time * 1000.0,
            split_phase_timing.select_time * 1000.0,
            split_phase_timing.count_time * 1000.0,
            split_phase_timing.write_time * 1000.0
        );
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
