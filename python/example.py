"""Simple example of using engram with NumPy."""

import tempfile

import numpy as np
from engram import VectorDB


def main():
    # Create temporary directory for database
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Creating database in {tmpdir}")

        # Create a VectorDB instance
        db = VectorDB(
            dir=tmpdir,
            dimension=128,
            lsh_tables=8,
            lsh_bits=16,
            seed=42,
        )

        print("Database created")
        print(f"Config: {db.get_config()}")

        # Generate random vectors
        num_vectors = 10000
        dimension = 128

        print(f"\nGenerating {num_vectors} random vectors...")
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        ids = np.arange(num_vectors, dtype=np.uint32)

        # Normalize for cosine distance
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-8)

        # Insert vectors in batches
        batch_size = 1000
        for i in range(0, num_vectors, batch_size):
            batch_vectors = vectors[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            db.insert_batch(batch_vectors, batch_ids)
            print(f"  Inserted {i + len(batch_ids)}/{num_vectors} vectors")

        print("\nFlushing and saving...")
        db.flush()
        db.save()

        print(f"Segments: {db.num_segments()}")

        # Query
        print("\nQuerying...")
        query_vector = np.random.randn(dimension).astype(np.float32)
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)

        results = db.query(query_vector, k=10, metric="cosine")
        print("Top 10 results for cosine metric:")
        for idx, (vec_id, distance) in enumerate(results, 1):
            print(f"  {idx}. ID: {vec_id:6d}, Distance: {distance:.6f}")

        # L2 distance query
        results_l2 = db.query(query_vector, k=10, metric="l2")
        print("\nTop 10 results for L2 metric:")
        for idx, (vec_id, distance) in enumerate(results_l2, 1):
            print(f"  {idx}. ID: {vec_id:6d}, Distance: {distance:.6f}")

        # Batch query
        print("\nBatch querying with 5 queries...")
        batch_queries = np.random.randn(5, dimension).astype(np.float32)
        batch_queries = batch_queries / (
            np.linalg.norm(batch_queries, axis=1, keepdims=True) + 1e-8
        )

        batch_results = db.query_batch(batch_queries, k=5, metric="cosine")
        for query_idx, results in enumerate(batch_results):
            print(f"\n  Query {query_idx}:")
            for vec_id, distance in results:
                print(f"    ID: {vec_id}, Distance: {distance:.6f}")


if __name__ == "__main__":
    main()
