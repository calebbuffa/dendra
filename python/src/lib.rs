use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use ndarray::s;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};

use fvdb::{cosine_distance, l2_distance, Query, VectorDB, VectorDBConfig};

/// Python wrapper for VectorDB
#[pyclass(name = "VectorDB")]
pub struct PyVectorDB {
    db: VectorDB,
}

#[pymethods]
impl PyVectorDB {
    /// Create a new VectorDB instance
    ///
    /// Args:
    ///     dir: Directory path to store database files
    ///     dimension: Vector dimension (e.g., 128)
    ///     leaf_size: RPF leaf size (default 32)
    ///     num_trees: Number of RPF trees (default 4)
    ///     seed: Random seed for reproducibility
    ///     max_segment_capacity: Max MB per segment (default 100)
    ///     quantize: Enable quantization (default True)
    ///     bit_width: Bit width for quantization (default 4)
    #[new]
    #[pyo3(signature = (dir, dimension, leaf_size=32, num_trees=4, seed=42, max_segment_capacity=100, quantize=true, bit_width=4))]
    fn new(
        dir: String,
        dimension: usize,
        leaf_size: usize,
        num_trees: usize,
        seed: u64,
        max_segment_capacity: usize,
        _quantize: bool,
        _bit_width: u32,
    ) -> PyResult<Self> {
        let config = VectorDBConfig::new(
            leaf_size,
            num_trees,
            dimension,
            seed,
            max_segment_capacity,
            2, // async_seal_queue_capacity
        );

        let db = VectorDB::new(std::path::PathBuf::from(dir), config);

        Ok(Self { db })
    }

    /// Insert a single vector
    ///
    /// Args:
    ///     vector: 1D NumPy array of f32
    ///     id: Vector ID (u32)
    fn insert(&mut self, vector: PyReadonlyArray1<f32>, id: u32) -> PyResult<()> {
        let vec_slice = vector.as_array();
        self.db
            .insert(vec_slice.as_slice().unwrap(), id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Insert multiple vectors
    ///
    /// Args:
    ///     vectors: 2D NumPy array of shape (n_vectors, dimension)
    ///     ids: 1D NumPy array of u32 IDs
    fn insert_batch(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u32>,
    ) -> PyResult<()> {
        let vectors_arr = vectors.as_array();
        let ids_arr = ids.as_array();

        if vectors_arr.nrows() != ids_arr.len() {
            return Err(PyValueError::new_err(
                "Number of vectors must match number of IDs",
            ));
        }

        for (i, id) in ids_arr.iter().enumerate() {
            let vector = vectors_arr.row(i).to_vec();
            self.db
                .insert(&vector, *id)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        Ok(())
    }

    /// Flush pending writes to disk
    fn flush(&mut self) -> PyResult<()> {
        self.db
            .flush()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Save database metadata and persist to disk
    fn save(&mut self) -> PyResult<()> {
        self.db
            .save()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Query the database
    ///
    /// Args:
    ///     query_vector: 1D NumPy array of f32
    ///     k: Number of results to return
    ///     metric: Distance metric ("cosine" or "l2")
    ///
    /// Returns:
    ///     List of tuples (id, distance)
    #[pyo3(signature = (query_vector, k, metric="cosine"))]
    fn query(
        &self,
        query_vector: PyReadonlyArray1<f32>,
        k: usize,
        metric: &str,
    ) -> PyResult<Vec<(u32, f32)>> {
        let vec_slice = query_vector.as_array();
        let metric_fn = match metric {
            "cosine" => cosine_distance,
            "l2" => l2_distance,
            _ => return Err(PyValueError::new_err("Metric must be 'cosine' or 'l2'")),
        };

        let query = Query {
            vector: vec_slice.to_vec(),
            k,
            metric: metric_fn,
        };

        let mut results = Vec::new();
        self.db
            .query(&query, &mut results)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(results)
    }

    /// Batch query the database
    ///
    /// Args:
    ///     queries: 2D NumPy array of shape (n_queries, dimension)
    ///     k: Number of results to return per query
    ///     metric: Distance metric ("cosine" or "l2")
    ///
    /// Returns:
    ///     List of lists, each containing (id, distance) tuples
    #[pyo3(signature = (queries, k, metric="cosine"))]
    fn query_batch(
        &self,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        metric: &str,
    ) -> PyResult<Vec<Vec<(u32, f32)>>> {
        let queries_arr = queries.as_array();
        let metric_fn = match metric {
            "cosine" => cosine_distance,
            "l2" => l2_distance,
            _ => return Err(PyValueError::new_err("Metric must be 'cosine' or 'l2'")),
        };

        let mut batch_results = Vec::new();

        for query_row in queries_arr.rows() {
            let mut results = Vec::with_capacity(k);
            let query = Query {
                vector: query_row.to_vec(),
                k,
                metric: metric_fn,
            };

            self.db
                .query(&query, &mut results)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            batch_results.push(results);
        }

        Ok(batch_results)
    }

    /// Load an existing database from disk
    #[staticmethod]
    fn load(dir: String) -> PyResult<Self> {
        let db = VectorDB::load(std::path::Path::new(&dir))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { db })
    }

    /// Get number of segments
    fn num_segments(&self) -> usize {
        self.db.segments.len()
    }

    /// Get database configuration
    fn get_config(&self) -> PyResult<Vec<(String, String)>> {
        Ok(vec![
            (
                "dimension".to_string(),
                self.db.config.dimension.to_string(),
            ),
            (
                "leaf_size".to_string(),
                self.db.config.leaf_size.to_string(),
            ),
            (
                "num_trees".to_string(),
                self.db.config.num_trees.to_string(),
            ),
            ("seed".to_string(), self.db.config.seed.to_string()),
            (
                "max_segment_capacity".to_string(),
                self.db.config.max_segment_capacity.to_string(),
            ),
        ])
    }

    fn __repr__(&self) -> String {
        format!(
            "VectorDB(dimension={}, segments={}, capacity={}MB)",
            self.db.config.dimension,
            self.db.segments.len(),
            self.db.config.max_segment_capacity
        )
    }
}

/// Python module initialization
#[pymodule]
fn fvdb(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyVectorDB>()?;

    // Add module docstring
    m.add(
        "__doc__",
        "Fast Vector Database (FVDB) - High-performance vector search with Random Projection Forests",
    )?;

    Ok(())
}
