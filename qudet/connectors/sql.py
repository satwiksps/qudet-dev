"""
SQL-to-Quantum data bridge for enterprise database integration.

Streams data directly from SQL databases into quantum circuits with
automatic pagination and connection pooling.
"""

import logging
from typing import Iterator, Optional, Tuple

import pandas as pd

from .loader import QuantumDataLoader

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import create_engine, text
    HAS_SQL = True
except ImportError:
    HAS_SQL = False


class QuantumSQLLoader:
    """Stream data from a SQL database into quantum circuits.

    In the real world data lives in databases, not CSVs.  This loader:

    * Connects to any SQL database (PostgreSQL, MySQL, SQLite, etc.).
    * Handles connection pooling automatically via SQLAlchemy.
    * Fetches data in batches (pagination).
    * Converts each batch to quantum circuits.

    Args:
        connection_string: SQLAlchemy connection string, e.g.
            ``"sqlite:///data.db"``,
            ``"postgresql://user:pass@localhost/mydb"``,
            ``"mysql+pymysql://user:pass@localhost/mydb"``.
        query: SQL query to fetch data.
        batch_size: Rows per batch.
        encoder_type: Quantum encoder type — ``'angle'`` or
            ``'amplitude'``.

    Attributes:
        engine: SQLAlchemy database engine.
        query (str): SQL query to execute.
        batch_size (int): Number of rows per batch.
        encoder_type (str): Active encoder type.

    Raises:
        ImportError: If SQLAlchemy is not installed.

    Example:
        >>> loader = QuantumSQLLoader(
        ...     connection_string="postgresql://user:pass@localhost/db",
        ...     query="SELECT feature1, feature2 FROM data",
        ...     batch_size=100,
        ... )
        >>> for data_batch, circuits in loader:
        ...     predictions = model.predict(circuits)
    """

    def __init__(
        self,
        connection_string: str,
        query: str,
        batch_size: int = 100,
        encoder_type: str = "angle",
    ) -> None:
        if not HAS_SQL:
            raise ImportError(
                "SQLAlchemy not installed. Run 'pip install sqlalchemy'."
            )

        if not connection_string:
            raise ValueError("'connection_string' must not be empty.")
        if not query or not query.strip():
            raise ValueError("'query' must not be empty.")

        self.engine = create_engine(connection_string)
        self.query = query
        self.batch_size = batch_size
        self.encoder_type = encoder_type

        logger.info(
            "QuantumSQLLoader initialised: batch_size=%d, encoder=%s",
            batch_size,
            encoder_type,
        )

    def __iter__(self) -> Iterator[Tuple[pd.DataFrame, list]]:
        """Yield batches of ``(DataFrame, circuits)``.

        Each batch is automatically converted to quantum circuits using
        the configured encoder.

        Yields:
            Tuple of ``(batch_data, quantum_circuits)``.
        """
        with self.engine.connect() as conn:
            chunk_iterator = pd.read_sql_query(
                text(self.query),
                conn,
                chunksize=self.batch_size,
            )

            for chunk_df in chunk_iterator:
                mini_loader = QuantumDataLoader(
                    chunk_df,
                    batch_size=self.batch_size,
                    encoder_type=self.encoder_type,
                )

                try:
                    data_batch, circuits = next(iter(mini_loader))
                    yield data_batch, circuits
                except StopIteration:
                    continue

    def execute_query(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Execute the query and return all results as a ``DataFrame``.

        Warning:
            This loads all data into memory.  For large datasets, use the
            iterator interface instead.

        Args:
            limit: Maximum rows to fetch.  ``None`` means no limit.

        Returns:
            Query results as a ``DataFrame``.
        """
        query = self.query
        if limit is not None:
            query = f"SELECT * FROM ({self.query}) AS _sub LIMIT {int(limit)}"

        with self.engine.connect() as conn:
            return pd.read_sql_query(text(query), conn)

    def get_batch_count(self) -> int:
        """Estimate the number of batches in the result set.

        Returns:
            Approximate batch count (ceiling division).
        """
        with self.engine.connect() as conn:
            count_query = f"SELECT COUNT(*) AS cnt FROM ({self.query}) AS _sub"
            result = conn.execute(text(count_query)).fetchone()
            row_count = result[0] if result else 0

        return max(1, (row_count + self.batch_size - 1) // self.batch_size)

    def close(self) -> None:
        """Close the database connection pool."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine disposed.")

    def __enter__(self) -> "QuantumSQLLoader":
        """Enter the runtime context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context, disposing of the engine."""
        self.close()
