use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use engram::{cosine_distance, l2_distance, QueryScratch, VectorDB, VectorDBConfig};

/// Python wrapper for VectorDB
#[pyclass(name = "VectorDB")]
pub struct PyVectorDB {
    db: VectorDB,
    scratch: QueryScratch,
    results_buf: Vec<(u32, f32)>,
}

#[pymethods]
impl PyVectorDB {
    fn pack_query_result(py: Python<'_>, results: &[(u32, f32)]) -> PyResult<PyObject> {
        let mut ids = Vec::with_capacity(results.len());
        let mut distances = Vec::with_capacity(results.len());
        for &(id, distance) in results {
            ids.push(id);
            distances.push(distance);
        }

        let out = PyDict::new(py);
        out.set_item("ids", PyArray1::from_vec(py, ids))?;
        out.set_item("distances", PyArray1::from_vec(py, distances))?;
        Ok(out.into())
    }

    /// Create a new VectorDB instance
    ///
    /// Args:
    ///     dir: Directory path to store database files
    ///     dimension: Vector dimension (e.g., 128)
    ///     lsh_tables: Number of LSH tables (default 8)
    ///     lsh_bits: Bits per LSH table signature (default 16)
    ///     seed: Random seed for reproducibility
    ///     max_segment_capacity: Max MB per segment (default 100)
    #[new]
    #[pyo3(signature = (dir, dimension, lsh_tables=8, lsh_bits=16, seed=42, max_segment_capacity=100))]
    fn new(
        dir: String,
        dimension: usize,
        lsh_tables: usize,
        lsh_bits: usize,
        seed: u64,
        max_segment_capacity: usize,
    ) -> PyResult<Self> {
        let config = VectorDBConfig::new(dimension, seed, max_segment_capacity, 2)
            .with_lsh_tables(lsh_tables)
            .with_lsh_bits(lsh_bits);

        let db = VectorDB::new(std::path::PathBuf::from(dir), config);

        Ok(Self {
            db,
            scratch: QueryScratch::default(),
            results_buf: Vec::new(),
        })
    }

    /// Insert a single vector
    ///
    /// Args:
    ///     vector: 1D NumPy array of f32
    ///     id: Vector ID (u32)
    fn insert(&mut self, vector: PyReadonlyArray1<f32>, id: u32) -> PyResult<()> {
        let vec_slice = vector.as_array();
        let Some(slice) = vec_slice.as_slice() else {
            return Err(PyValueError::new_err(
                "insert expects a contiguous float32 NumPy array",
            ));
        };
        self.db
            .insert(slice, id)
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

        let Some(vectors_slice) = vectors_arr.as_slice_memory_order() else {
            return Err(PyValueError::new_err(
                "insert_batch expects a C-contiguous float32 NumPy array; use np.ascontiguousarray",
            ));
        };

        let Some(ids_slice) = ids_arr.as_slice_memory_order() else {
            return Err(PyValueError::new_err(
                "insert_batch expects a contiguous uint32 ids array; use np.ascontiguousarray",
            ));
        };

        let dim = vectors_arr.ncols();
        if dim == 0 {
            return Ok(());
        }

        for (vector, id) in vectors_slice.chunks_exact(dim).zip(ids_slice.iter()) {
            self.db
                .insert(vector, *id)
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
        &mut self,
        py: Python<'_>,
        query_vector: PyReadonlyArray1<f32>,
        k: usize,
        metric: &str,
    ) -> PyResult<PyObject> {
        let vec_slice = query_vector.as_array();
        let Some(query_slice) = vec_slice.as_slice() else {
            return Err(PyValueError::new_err(
                "query expects a contiguous float32 NumPy array",
            ));
        };
        let metric_fn = match metric {
            "cosine" => cosine_distance,
            "l2" => l2_distance,
            _ => return Err(PyValueError::new_err("Metric must be 'cosine' or 'l2'")),
        };

        self.results_buf.clear();
        self.results_buf.reserve(k);
        let delta = self.db.config().delta;

        self.db
            .query_raw(
                query_slice,
                k,
                Some(metric_fn),
                None,
                delta,
                &mut self.scratch,
                &mut self.results_buf,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let results = std::mem::take(&mut self.results_buf);
        Self::pack_query_result(py, &results)
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
        &mut self,
        py: Python<'_>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        metric: &str,
    ) -> PyResult<PyObject> {
        let queries_arr = queries.as_array();
        let metric_fn = match metric {
            "cosine" => cosine_distance,
            "l2" => l2_distance,
            _ => return Err(PyValueError::new_err("Metric must be 'cosine' or 'l2'")),
        };

        let Some(queries_slice) = queries_arr.as_slice_memory_order() else {
            return Err(PyValueError::new_err(
                "query_batch expects a C-contiguous float32 NumPy array; use np.ascontiguousarray",
            ));
        };

        let dim = queries_arr.ncols();
        let delta = self.db.config().delta;

        let n_queries = queries_arr.nrows();
        let mut ids_rows = vec![vec![0u32; k]; n_queries];
        let mut distance_rows = vec![vec![f32::INFINITY; k]; n_queries];
        let mut counts = vec![0u32; n_queries];

        for (row_idx, query_row) in queries_slice.chunks_exact(dim).enumerate() {
            self.results_buf.clear();
            self.results_buf.reserve(k);
            self.db
                .query_raw(
                    query_row,
                    k,
                    Some(metric_fn),
                    None,
                    delta,
                    &mut self.scratch,
                    &mut self.results_buf,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let n = self.results_buf.len().min(k);
            counts[row_idx] = n as u32;
            for j in 0..n {
                let (id, distance) = self.results_buf[j];
                ids_rows[row_idx][j] = id;
                distance_rows[row_idx][j] = distance;
            }
        }

        let out = PyDict::new(py);
        out.set_item("ids", PyArray2::from_vec2(py, &ids_rows)?)?;
        out.set_item("distances", PyArray2::from_vec2(py, &distance_rows)?)?;
        out.set_item("counts", PyArray1::from_vec(py, counts))?;
        Ok(out.into())
    }

    /// Load an existing database from disk
    #[staticmethod]
    #[pyo3(signature = (dir, lsh_tables=8, lsh_bits=16, seed=42, max_segment_capacity=100))]
    fn load(
        dir: String,
        lsh_tables: usize,
        lsh_bits: usize,
        seed: u64,
        max_segment_capacity: usize,
    ) -> PyResult<Self> {
        let config = VectorDBConfig::new(1, seed, max_segment_capacity, 2)
            .with_lsh_tables(lsh_tables)
            .with_lsh_bits(lsh_bits);

        let db = VectorDB::load(std::path::Path::new(&dir), config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            db,
            scratch: QueryScratch::default(),
            results_buf: Vec::new(),
        })
    }

    /// Get number of segments
    fn num_segments(&self) -> usize {
        self.db.num_sealed_segments()
    }

    /// Get database configuration
    fn get_config(&self) -> PyResult<Vec<(String, String)>> {
        let cfg = self.db.config();
        Ok(vec![
            ("dimension".to_string(), cfg.dimension.to_string()),
            ("lsh_tables".to_string(), cfg.lsh_num_tables.to_string()),
            ("lsh_bits".to_string(), cfg.lsh_bits_per_table.to_string()),
            (
                "max_segment_capacity_mb".to_string(),
                cfg.segment_capacity_mb.to_string(),
            ),
            ("seed".to_string(), cfg.seed.to_string()),
        ])
    }

    fn __repr__(&self) -> String {
        let cfg = self.db.config();
        format!(
            "VectorDB(dimension={}, segments={}, capacity={}MB)",
            cfg.dimension,
            self.db.num_sealed_segments(),
            cfg.segment_capacity_mb
        )
    }
}

/// Python module initialization
#[pymodule]
fn engram(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyVectorDB>()?;

    // Add module docstring
    m.add(
        "__doc__",
        "engram: Bayesian LSH vector database with adaptive probabilistic routing",
    )?;
    m.add("__python_version__", py.version())?;

    Ok(())
}
