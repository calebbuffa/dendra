from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

MetricName = Literal["cosine", "l2"]
ArrayF32: TypeAlias = npt.NDArray[np.float32]
Array2DF32: TypeAlias = npt.NDArray[np.float32]
ArrayU32: TypeAlias = npt.NDArray[np.uint32]


class QueryResult(TypedDict):
    ids: npt.NDArray[np.uint32]
    distances: npt.NDArray[np.float32]


class QueryBatchResult(TypedDict):
    ids: npt.NDArray[np.uint32]
    distances: npt.NDArray[np.float32]
    counts: npt.NDArray[np.uint32]


class VectorDB:
    """Bayesian LSH vector database.

    This class provides insert, persistence, and nearest-neighbor query
    operations over float32 vectors.
    """

    def __init__(
        self,
        dir: str,
        dimension: int,
        lsh_tables: int = 8,
        lsh_bits: int = 16,
        seed: int = 42,
        max_segment_capacity: int = 100,
    ) -> None:
        """Create a new database instance.

        Parameters
        ----------
        dir : str
            Directory used for on-disk data.
        dimension : int
            Vector dimensionality.
        lsh_tables : int, default=8
            Number of LSH tables.
        lsh_bits : int, default=16
            Number of signature bits per table.
        seed : int, default=42
            Random seed used for deterministic index construction.
        max_segment_capacity : int, default=100
            Maximum sealed segment capacity in megabytes.

        Returns
        -------
        None
        """
        ...

    def insert(self, vector: ArrayF32, id: int) -> None:
        """Insert a single vector.

        Parameters
        ----------
        vector : numpy.typing.NDArray[numpy.float32]
            C-contiguous float32 vector of shape (dimension,).
        id : int
            External identifier for the vector.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If input shape/dtype is invalid or insertion fails.
        """
        ...

    def insert_batch(
        self,
        vectors: Array2DF32,
        ids: ArrayU32,
    ) -> None:
        """Insert a batch of vectors.

        Parameters
        ----------
        vectors : numpy.typing.NDArray[numpy.float32]
            C-contiguous float32 array with shape (n_vectors, dimension).
        ids : numpy.typing.NDArray[numpy.uint32]
            Contiguous uint32 array with shape (n_vectors,).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If vector/id counts differ, arrays are not contiguous,
            or insertion fails.
        """
        ...

    def flush(self) -> None:
        """Flush pending writes.

        Returns
        -------
        None
        """
        ...

    def save(self) -> None:
        """Persist database metadata and sealed segments.

        Returns
        -------
        None
        """
        ...

    def query(
        self,
        query_vector: ArrayF32,
        k: int,
        metric: MetricName = "cosine",
    ) -> QueryResult:
        """Query nearest neighbors for one vector.

        Parameters
        ----------
        query_vector : numpy.typing.NDArray[numpy.float32]
            Contiguous float32 vector with shape (dimension,).
        k : int
            Number of neighbors to return.
        metric : {"cosine", "l2"}, default="cosine"
            Distance metric.

        Returns
        -------
        QueryResult
            Dictionary containing NumPy arrays:
            - ids: uint32 array of shape (n,)
            - distances: float32 array of shape (n,)

        Raises
        ------
        ValueError
            If metric is unsupported, array is not contiguous,
            or query fails.
        """
        ...

    def query_batch(
        self,
        queries: Array2DF32,
        k: int,
        metric: MetricName = "cosine",
    ) -> QueryBatchResult:
        """Query nearest neighbors for multiple vectors.

        Parameters
        ----------
        queries : numpy.typing.NDArray[numpy.float32]
            C-contiguous float32 matrix with shape (n_queries, dimension).
        k : int
            Number of neighbors per query.
        metric : {"cosine", "l2"}, default="cosine"
            Distance metric.

        Returns
        -------
        QueryBatchResult
            Dictionary containing NumPy arrays:
            - ids: uint32 array of shape (n_queries, k)
            - distances: float32 array of shape (n_queries, k)
            - counts: uint32 array of shape (n_queries,), valid results per row
        """
        ...

    @staticmethod
    def load(
        dir: str,
        lsh_tables: int = 8,
        lsh_bits: int = 16,
        seed: int = 42,
        max_segment_capacity: int = 100,
    ) -> "VectorDB":
        """Load a database from disk.

        Parameters
        ----------
        dir : str
            Database directory path.
        lsh_tables : int, default=8
            Number of LSH tables.
        lsh_bits : int, default=16
            Number of signature bits per table.
        seed : int, default=42
            Random seed for deterministic index construction.
        max_segment_capacity : int, default=100
            Maximum active segment capacity in megabytes.

        Returns
        -------
        VectorDB
            Loaded database handle.

        Raises
        ------
        ValueError
            If directory does not exist or does not contain a populated database.
        """
        ...

    def num_segments(self) -> int:
        """Return number of sealed segments.

        Returns
        -------
        int
            Number of sealed immutable segments.
        """
        ...

    def get_config(self) -> list[tuple[str, str]]:
        """Return runtime configuration as key/value strings.

        Returns
        -------
        list[tuple[str, str]]
            Configuration entries serialized as strings.
        """
        ...
