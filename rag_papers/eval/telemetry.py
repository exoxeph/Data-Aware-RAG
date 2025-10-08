"""
Telemetry and logging infrastructure for evaluation harness.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# Configure structured logging if available
if STRUCTLOG_AVAILABLE:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger()
else:
    # Fallback to stdlib logging with JSON formatting
    logging.basicConfig(
        level=logging.INFO,
        format='{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(__name__)


def new_run_id() -> str:
    """Generate a new unique run ID."""
    return str(uuid.uuid4())


def log_step(
    run_id: str,
    step_name: str,
    ms: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a DAG step execution with timing.
    
    Args:
        run_id: Unique run identifier
        step_name: Name of the step (e.g., "retrieve", "generate")
        ms: Duration in milliseconds
        extra: Optional extra fields to log
    """
    log_data = {
        "run_id": run_id,
        "step": step_name,
        "duration_ms": ms,
        **(extra or {}),
    }
    
    if STRUCTLOG_AVAILABLE:
        logger.info("step_executed", **log_data)
    else:
        logger.info(json.dumps(log_data))


def persist_duckdb(
    db_path: str | Path,
    table: str,
    records: List[Dict[str, Any]],
) -> None:
    """
    Persist records to a DuckDB table.
    
    Args:
        db_path: Path to DuckDB database file
        table: Table name
        records: List of record dictionaries
        
    Raises:
        ImportError: If DuckDB is not installed
        ValueError: If records list is empty
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError(
            "DuckDB is not installed. Install with: poetry add duckdb"
        )
    
    if not records:
        raise ValueError("Cannot persist empty records list")
    
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect(str(db_path))
    
    try:
        # Convert records to pandas DataFrame if available, otherwise use DuckDB's from_dict
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(records)
            con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df WHERE 1=0")
            con.execute(f"INSERT INTO {table} SELECT * FROM df")
        else:
            # Infer schema from first record and create table if needed
            first_record = records[0]
            columns = list(first_record.keys())
            
            # Try to create table (will fail if exists, which is fine)
            try:
                col_defs = []
                for col in columns:
                    val = first_record[col]
                    if isinstance(val, bool):
                        col_type = "BOOLEAN"
                    elif isinstance(val, int):
                        col_type = "INTEGER"
                    elif isinstance(val, float):
                        col_type = "DOUBLE"
                    else:
                        col_type = "VARCHAR"
                    col_defs.append(f"{col} {col_type}")
                
                create_sql = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(col_defs)})"
                con.execute(create_sql)
            except Exception:
                pass  # Table likely already exists
            
            # Insert records
            placeholders = ", ".join(["?"] * len(columns))
            insert_sql = f"INSERT INTO {table} VALUES ({placeholders})"
            
            for record in records:
                values = [record.get(col) for col in columns]
                con.execute(insert_sql, values)
        
        con.commit()
        logger.info(f"Persisted {len(records)} records to {table} in {db_path}")
        
    finally:
        con.close()


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write records to CSV file.
    
    Args:
        path: Output CSV path
        rows: List of record dictionaries
        
    Raises:
        ImportError: If pandas is not installed
        ValueError: If rows list is empty
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is not installed. Install with: poetry add pandas"
        )
    
    if not rows:
        raise ValueError("Cannot write empty rows list")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"Wrote {len(rows)} rows to {path}")


def write_parquet(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write records to Parquet file.
    
    Args:
        path: Output Parquet path
        rows: List of record dictionaries
        
    Raises:
        ImportError: If pandas is not installed
        ValueError: If rows list is empty
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is not installed. Install with: poetry add pandas"
        )
    
    if not rows:
        raise ValueError("Cannot write empty rows list")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Wrote {len(rows)} rows to {path}")
