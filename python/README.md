# FVDB Python Bindings

High-performance vector search library for Python with Random Projection Forests.

## Installation

```bash
pip install fvdb
```

Or build from source:

```bash
cd python
pip install maturin
maturin develop
```

## Quick Start

```python
import numpy as np
from fvdb import VectorDB

# Create a database
db = VectorDB(
    dir="./my_vectors",
    dimension=128,
    leaf_size=32,
    num_trees=4,
)

# Insert vectors
vectors = np.random.randn(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.uint32)

db.insert_batch(vectors, ids)
db.flush()
db.save()

# Query
query = np.random.randn(128).astype(np.float32)
results = db.query(query, k=10, metric="cosine")

for vec_id, distance in results:
    print(f"ID: {vec_id}, Distance: {distance}")

# Load from disk
db2 = VectorDB.load("./my_vectors")
```

## Metrics

- `"cosine"` - Cosine distance (default)
- `"l2"` - Euclidean (L2) distance

## API

### VectorDB(dir, dimension, leaf_size=32, num_trees=4, seed=42, max_segment_capacity=100)

Create or open a vector database.

**Parameters:**

- `dir` (str): Directory to store database files
- `dimension` (int): Vector dimension
- `leaf_size` (int): RPF leaf size for index granularity
- `num_trees` (int): Number of random projection trees
- `seed` (int): Random seed for reproducibility
- `max_segment_capacity` (int): Max MB per sealed segment

### Methods

- `insert(vector: np.ndarray, id: int)` - Insert a single vector
- `insert_batch(vectors: np.ndarray, ids: np.ndarray)` - Insert multiple vectors
- `query(vector: np.ndarray, k: int, metric: str = "cosine")` - Query the database
- `query_batch(vectors: np.ndarray, k: int, metric: str = "cosine")` - Batch query
- `flush()` - Flush pending writes to disk
- `save()` - Persist database metadata
- `load(dir: str)` - Static method to load from disk
- `num_segments()` - Get number of sealed segments
- `get_config()` - Get database configuration

## Performance

Typical performance on 1M vectors (128-dim, cosine distance):

- Insert: ~5-6k ns/vector
- Query k=100: ~10ms over 1M vectors
