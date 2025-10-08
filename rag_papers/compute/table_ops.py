"""Table operations and structured data queries."""

from typing import Any, Dict, List

import duckdb
import pandas as pd


def find_auc_for_dataset(db_path: str, dataset: str) -> Dict[str, Any]:
    """Find AUC values for a specific dataset from normalized tables.
    
    Args:
        db_path: Path to DuckDB database
        dataset: Dataset name to search for
        
    Returns:
        Dictionary with paper_id, table_id, page, and matching rows
    """
    con = duckdb.connect(db_path)
    
    # Find tables that have AUC columns
    cand = con.execute("""
      SELECT m.paper_id, m.table_id, m.page, m.parquet_path
      FROM tables_meta m
      JOIN table_columns c USING (paper_id, table_id)
      WHERE lower(c.col_name) LIKE '%auc%'
    """).fetchall()
    
    # Search through candidate tables
    for paper_id, table_id, page, pq in cand:
        try:
            df = pd.read_parquet(pq)
            
            # Create case-insensitive column mapping
            cols_lower = {c.lower(): c for c in df.columns}
            
            if "auc" not in cols_lower:
                continue
                
            auc_col = cols_lower["auc"]
            
            # Filter by dataset if dataset column exists
            if "dataset" in cols_lower:
                dataset_col = cols_lower["dataset"]
                rows = df[df[dataset_col].astype(str).str.lower() == dataset.lower()]
            else:
                # If no dataset column, return all rows
                rows = df
            
            if not rows.empty:
                return {
                    "paper_id": paper_id,
                    "table_id": table_id,
                    "page": int(page),
                    "rows": [{"value": float(rows.iloc[0][auc_col]), "col_name": "AUC"}],
                }
        except Exception:
            # Skip tables that can't be read
            continue
    
    con.close()
    return {"paper_id": None, "rows": []}