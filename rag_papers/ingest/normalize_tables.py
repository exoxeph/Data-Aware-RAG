"""Table extraction and normalization from marker JSON output."""

import json
import pathlib
from typing import Any

import duckdb
import pandas as pd


def _tidy_df(raw_rows: list[list[Any]]) -> pd.DataFrame:
    """Clean and organize raw table rows into a proper DataFrame.
    
    Args:
        raw_rows: List of row data from marker JSON
        
    Returns:
        Cleaned pandas DataFrame with proper column headers
    """
    df = pd.DataFrame(raw_rows)
    
    # If columns are just integer indices and we have data, use first row as headers
    if df.columns.tolist() == list(range(len(df.columns))) and len(df) > 1:
        df.columns = df.iloc[0].astype(str)
        df = df.iloc[1:].reset_index(drop=True)
    
    return df


def normalize_marker_json(json_path: pathlib.Path, parquet_dir: pathlib.Path, db_path: str) -> None:
    """Normalize marker JSON output into DuckDB tables and parquet files.
    
    Args:
        json_path: Path to marker JSON file
        parquet_dir: Directory to store parquet files
        db_path: Path to DuckDB database
    """
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    con = duckdb.connect(db_path)
    con.execute("""CREATE TABLE IF NOT EXISTS tables_meta(
      paper_id TEXT, table_id TEXT, page INT, caption TEXT, parquet_path TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS table_columns(
      paper_id TEXT, table_id TEXT, col_index INT, col_name TEXT)""")
    
    # Parse JSON
    j = json.loads(json_path.read_text())
    paper_id = json_path.stem
    
    # Process each page
    for page in j:
        pg = page.get("metadata", {}).get("page_number", -1)
        
        # Process each block in the page
        for b in page.get("children", []):
            if b.get("block_type") != "Table":
                continue
                
            tbl_id = b.get("id", f"tbl_{pg}")
            caption = b.get("caption", "")
            rows = b.get("data", {}).get("rows", [])
            
            if not rows:
                continue
            
            # Convert to DataFrame and save
            df = _tidy_df(rows)
            pq = parquet_dir / f"{paper_id}__{tbl_id}.parquet"
            df.to_parquet(pq)
            
            # Store metadata
            con.execute("INSERT INTO tables_meta VALUES (?,?,?,?,?)",
                        [paper_id, tbl_id, pg, caption, str(pq)])
            
            # Store column information
            for i, name in enumerate(map(str, df.columns.tolist())):
                con.execute("INSERT INTO table_columns VALUES (?,?,?,?)",
                            [paper_id, tbl_id, i, name])
    
    con.close()