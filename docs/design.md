# engram: Design Thesis

## Bayesian LSH Routing with Disciplined Approximations

### Design Statement

engram is a segmented vector search system that concentrates Bayesian rigor in routing, while using bounded approximations for throughput-critical paths.

The system is not a single global Bayesian posterior over all database structure. It is an approximation stack with explicit cost control:

- exact conjugate updates where they matter for routing decisions,
- cheap heuristics where exact inference is too expensive,
- hard operational bounds for latency predictability.

---

## 1. Cost Budget

| Phase | Allowed Cost | Why |
|---|---|---|
| Query-time routing | O(T *P* d_e) + O(C) bookkeeping | Must be low-latency |
| Query-time rerank | O(C * d) ADC | Dominant per-query math cost |
| Insert-time | O(d) append | Must stay synchronous |
| Background seal | O(N *T* b *s) + O(N* d) SQ8 encode | Async worker path |
| Background compaction | O(S^2 * d') proxy scoring | Must remain periodic |

Notation:

- T = number of LSH tables
- P = probed signatures per table (radius dependent)
- d_e = bucket expert projection dimensions
- C = routed candidate count after adaptive cap
- d = original vector dimension
- N = vectors in a sealing batch
- b = bits per table
- s = sampled dims per bit
- S = number of sealed segments

---

## 2. Architecture Overview

Database layout:

- Active segment (mutable, in-memory vectors)
- Sealed segments (immutable, persisted)
- Per-segment Bayesian LSH index
- Per-segment SQ8 payload store
- Background workers for seal and compaction

Operational write path:

1. Insert appends to active segment.
2. Active segment hits capacity.
3. Background seal builds Bayesian LSH + SQ8.
4. Segment persists as segment.bin + ids.bin + codes.bin.
5. New active segment is created.

Operational read path:

1. Score sealed segments by root Bayesian score.
2. Select segments by routing-mass policy.
3. For each selected segment, run Bayesian LSH routing.
4. Apply adaptive candidate cap from confidence function.
5. Rerank candidates with ADC and return top-k.

---

## 3. Routing Model

### 3.1 Bucket Expert Predictive Model

For each table-signature bucket expert and each expert dimension j, use a Normal-Inverse-Gamma prior:

$$(\mu_j, \sigma_j^2) \sim \text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$$

Given bucket data D, posterior parameters are closed-form:

$$
\begin{aligned}
\kappa_n &= \kappa_0 + n \\
\mu_n &= \frac{\kappa_0 \mu_0 + n\bar{x}}{\kappa_n} \\
\alpha_n &= \alpha_0 + \frac{n}{2} \\
\beta_n &= \beta_0 + \frac{1}{2}\sum_i (x_i-\bar{x})^2 + \frac{\kappa_0 n(\bar{x}-\mu_0)^2}{2\kappa_n}
\end{aligned}
$$

Predictive scoring uses Student-t log-likelihood summed across expert dimensions:

$$
\log p(q \mid D) = \sum_{j=1}^{d_e} \log t_{2\alpha_{n,j}}\!\left(q_j \mid \mu_{n,j}, \frac{\beta_{n,j}(\kappa_{n,j}+1)}{\alpha_{n,j}\kappa_{n,j}}\right)
$$

Status: exact conjugate updates and predictive evaluation per expert dimension.

### 3.2 LSH Candidate Generation

Each table computes a signature from sparse random projections. Query probing uses bounded Hamming neighborhoods.

Per table:

1. Compute query signature.
2. Enumerate signatures within probe radius.
3. For each matching bucket, evaluate Bayesian expert score.
4. Add score to each row in that bucket and accumulate table votes.

This produces row-level evidence tuples (row, score, votes).

Status: heuristic retrieval skeleton with Bayesian bucket scoring.

### 3.3 Smooth Adaptive Fanout

Let confidence c be in [0, 1], built from:

- vote concentration (table support),
- soft evidence concentration among top rows,
- top score margin.

Candidate budget is computed as:

$$
K(c) = K_{\min} + (1-c)^{\gamma}(K_{\max}-K_{\min})
$$

and clamped to [K_min, K_max].

Status: motivated approximation with explicit hard bounds for latency.

---

## 4. Segment Selection and Compaction

### 4.1 Segment Selection

Sealed segments are scored by root-level Bayesian predictive statistics. Selection uses a routing-mass budget parameter delta to balance breadth and speed.

Status: heuristic mass-budget policy with operational meaning.

### 4.2 Compaction Proxy

Compaction uses cheap BIC-style proxies over root statistics instead of full merged-index reconstruction at decision time.

Status: acknowledged approximation chosen for asynchronous throughput.

---

## 5. Random Projection Justification

Sparse Gaussian-style random projections are used for both signature construction and expert-space projection.

Guidance comes from JL-style distance preservation:

$$d' = \Omega\!\left(\epsilon^{-2}\log N\right)$$

The implementation uses practical low-dimensional projections for speed; no claim is made that all information divergences are preserved.

Status: theory-guided engineering choice.

---

## 6. Storage Layer

### 6.1 Separation of Concerns

Storage is intentionally decoupled from routing inference.

- Routing decides where to look.
- SQ8 codec optimizes memory and rerank cost.

### 6.2 SQ8 Codec

Per-dimension min/max scalar quantization:

$$
\text{code}_i(x) = \left\lfloor 255\cdot\frac{x_i-\min_i}{\max_i-\min_i}+0.5\right\rfloor
$$

ADC reranking with per-dimension scale adaptation:

$$
\|q-\tilde{x}\|_2^2 = \sum_i \alpha_i^2\,(q'_i-\text{code}_i)^2, \quad \alpha_i = \frac{\max_i-\min_i}{255}
$$

Status: practical codec with known trade-offs.

### 6.3 Persistence Format

Per sealed segment directory:

- segment.bin: versioned metadata payload
- ids.bin: external id sidecar
- codes.bin: SQ8 code sidecar

Read path attempts memory mapping for sidecars.

---

## 7. Operational Semantics

### 7.1 Write Path

insert(vector, id):

1. Validate dimensionality.
2. Append to active segment.
3. If capacity reached, enqueue async seal.
4. Seal builds Bayesian LSH + SQ8 and persists sidecars.

### 7.2 Read Path

query(q, k):

1. Route across sealed segments by root score and delta policy.
2. Run Bayesian LSH routing in selected segments.
3. Apply smooth adaptive candidate cap.
4. ADC rerank and merge top-k results.

Freshness:

- Active segment entries are query-visible before sealing.
- Sealed segments are immutable after persistence.

---

## 8. Current Default Knobs

| Parameter | Default | Rationale |
|---|---|---|
| lsh_num_tables | 8 | Enough diversity without high build cost |
| lsh_bits_per_table | 16 | Balanced bucket granularity |
| lsh_dims_per_bit | 8 | Sparse fast signature planes |
| lsh_probe_hamming_radius | 1 | Recall/latency balance |
| lsh_bucket_expert_dims | 4 | Cheap Bayesian expert scoring |
| lsh_min_candidates | 384 | Lower fanout floor |
| lsh_max_candidates | 2048 | Hard latency ceiling |
| lsh_adaptive_gamma | 2.2 | More aggressive shrink at high confidence |
| delta | 0.05 | Typical routing mass budget |

All values are workload-dependent and should be validated with recall-latency sweeps.

---

## 9. Claims Summary

| Claim | Status |
|---|---|
| Bucket expert updates are exact conjugate Bayesian | Exact |
| Bucket predictive log-likelihood is exact under NIG model | Exact |
| Global database posterior inference is exact | Not claimed |
| LSH probing and candidate union are exact nearest-neighbor search | Not claimed |
| Smooth adaptive cap is Bayes-optimal | Not claimed |
| Adaptive cap provides operational latency control | True |
| Root-stat compaction criterion is exact model evidence | Not claimed |
| SQ8 is mathematically optimal compression | Not claimed |
