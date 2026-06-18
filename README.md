# engram

`engram` is an LSM-inspired vector database that uses Locality-Sensitive Hashing (LSH) to route queries into sparse Bayesian experts for probabilistic candidate generation.

The current engine is built for high ingest throughput, predictable query latency, and simplicity:

- Mutable active segment for fast inserts.
- Background sealing to immutable segments.
- Segment-local Bayesian LSH index for candidate generation.
- SQ8 payload store with sidecar files and memory-mapped read path.

## Architecture

At a high level:

1. Inserts append to the active in-memory segment.
2. When the segment reaches capacity, it is sealed in the background.
3. Sealing builds a Bayesian LSH index and SQ8 store.
4. Sealed segments are persisted and reopened through memory maps.
5. Query-time routing selects segments, then candidates, then ADC reranks top-k.

### Indexing model

Each immutable segment owns its own Bayesian LSH index.

- Each table hashes vectors into signature buckets.
- Each bucket stores Bayesian expert statistics (Normal-Inverse-Gamma per expert dimension).
- Querying probes exact and nearby signatures (bounded Hamming radius).
- Candidate rows are scored by bucket Bayesian evidence plus vote support.
- A smooth confidence function maps uncertainty to an adaptive candidate budget.

Candidate fanout is bounded by min and max limits for latency control.

### Query routing model

Since the DB relies on immutable sealed segments, naive segment selection would degrade performance as segments grow. To mitigate this `engram` uses a two-stage process:

1. **Route to relevant sealed segments**
   - For each sealed segment, compute a segment-level Bayesian score from its root statistics.
   - Convert scores to normalized routing mass.
   - Select the smallest set of segments whose cumulative mass reaches the target `(1 - delta)`.

2. **Route within each selected segment**
   - Run that segment’s Bayesian LSH index to generate candidate rows.
   - Probe exact + nearby signatures (bounded Hamming radius).
   - Score candidates with bucket Bayesian evidence plus table-vote support.
   - Apply smooth adaptive fanout (bounded by `lsh_min_candidates` and `lsh_max_candidates`).

3. **Rerank and return**
   - Rerank selected candidates with distance/ADC.
   - Merge across selected segments and return top-k.

4. **Fresh data path**
   - The active in-memory segment is scanned directly, so newly inserted vectors are query-visible before sealing.

### Storage model

Each sealed segment directory contains:

- segment.bin: versioned segment metadata payload.
- ids.bin: row-aligned external ids sidecar.
- codes.bin: SQ8 codes sidecar.

The ids and codes sidecars are used through memory maps when possible.

## Current status

Implemented:

- Bayesian LSH index build and query path.
- Bucket-level Bayesian experts.
- Confidence-based smooth adaptive candidate budgeting.
- SQ8 segment storage with sidecar persistence.
- Memory-mapped sealed-segment read path.
- Background sealing and compaction workflow.

Not yet implemented:

- Deletes/tombstones.
- Full WAL-backed crash recovery for active in-memory state.
