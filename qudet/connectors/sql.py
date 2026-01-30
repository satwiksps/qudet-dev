"""
SQL-to-Quantum data bridge for enterprise database integration.

Streams data directly from SQL databases into quantum circuits with
automatic pagination and connection pooling.
"""

import pandas as pd
from typing import Iterator, Tuple, Optional
from .loader import QuantumDataLoader

try:
    from sqlalchemy import create_engine, text
    HAS_SQL = True
except ImportError:
    HAS_SQL = False


class QuantumSQLLoader:
    """
    Streams data directly from a SQL Database into Quantum Circuits.
    
    In the real world, data lives in databases, not CSVs. This loader:
    - Connects to any SQL database (PostgreSQL, MySQL, SQLite, etc.)
    - Handles connection pooling automatically
    - Fetches data in batches (pagination)
    - Converts each batch to quantum circuits
    
    Example:
        loader = QuantumSQLLoader(
            connection_string="postgresql://user:pass@localhost/db",
            query="SELECT feature1, feature2, feature3 FROM data",
            batch_size=100
        )
        
        for data_batch, circuits in loader:
            # Process batch
            predictions = model.predict(circuits)
    
    Attributes:
        engine: SQLAlchemy database engine
        query (str): SQL query to execute
        batch_size (int): Number of rows per batch
        encoder_type (str): Type of encoder ('angle', 'amplitude', 'iqp')
    """
    
    def __init__(
        self, 
        connection_string: str, 
        query: str, 
        batch_size: int = 100, 
        encoder_type: str = 'angle'
    ):
        """
        Initialize QuantumSQLLoader.
        
        Args:
            connection_string (str): SQLAlchemy connection string
                Examples:
                - "sqlite:///data.db"
                - "postgresql://user:pass@localhost/mydb"
                - "mysql+pymysql://user:pass@localhost/mydb"
            query (str): SQL query to fetch data
            batch_size (int): Rows per batch. Default: 100
            encoder_type (str): Quantum encoder type. Default: 'angle'
            
        Raises:
            ImportError: If SQLAlchemy not installed
        """
        if not HAS_SQL:
            raise ImportError("SQLAlchemy not installed. Run 'pip install sqlalchemy'.")
            
        self.engine = create_engine(connection_string)
        self.query = query
        self.batch_size = batch_size
        self.encoder_type = encoder_type

    def __iter__(self) -> Iterator[Tuple[pd.DataFrame, list]]:
        """
        Yields batches of (DataFrame, Circuits).
        
        Each batch is automatically converted to quantum circuits
        using the specified encoder.
        
        Yields:
            Tuple[pd.DataFrame, list]: (batch_data, quantum_circuits)
        """
        with self.engine.connect() as conn:
            chunk_iterator = pd.read_sql_query(
                text(self.query), 
                conn, 
                chunksize=self.batch_size
            )
            
            for chunk_df in chunk_iterator:
                mini_loader = QuantumDataLoader(
                    chunk_df, 
                    batch_size=self.batch_size, 
                    encoder_type=self.encoder_type
                )
                
                try:
                    data_batch, circuits = next(iter(mini_loader))
                    yield data_batch, circuits
                except StopIteration:
                    continue

    def execute_query(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Execute the query and return all results as DataFrame.
        
        WARNING: This loads all data into memory. For large datasets,
        use the iterator interface instead.
        
        Args:
            limit (int, optional): Maximum rows to fetch. Default: None (no limit)
            
        Returns:
            pd.DataFrame: Query results
        """
        query = self.query
        if limit:
            query = f"({self.query}) LIMIT {limit}"
            
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(query), conn)

    def get_batch_count(self) -> int:
        """
        Estimate number of batches in the result set.
        
        Returns:
            int: Approximate batch count
        """
        with self.engine.connect() as conn:
            count_query = f"SELECT COUNT(*) as cnt FROM ({self.query}) t"
            result = conn.execute(text(count_query)).fetchone()
            row_count = result[0] if result else 0
            
        return max(1, (row_count + self.batch_size - 1) // self.batch_size)

    def close(self):
        """Close the database connection pool."""
        if self.engine:
            self.engine.dispose()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
