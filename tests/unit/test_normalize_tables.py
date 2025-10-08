"""Unit test for table normalization from marker JSON."""

import duckdb
from pathlib import Path

from rag_papers.ingest.normalize_tables import normalize_marker_json


def test_normalize_marker_json(tmp_dirs, sample_marker_json):
    """Test that marker JSON is properly normalized into DuckDB."""
    parquet_dir = tmp_dirs["parquet"]
    db_path = str(tmp_dirs["tables_db"])
    
    # Run the normalization
    normalize_marker_json(sample_marker_json, parquet_dir, db_path)
    
    # Check that database was created and tables exist
    con = duckdb.connect(db_path)
    
    # Test tables_meta
    meta_rows = con.execute("SELECT * FROM tables_meta").fetchall()
    assert len(meta_rows) == 1
    
    paper_id, table_id, page, caption, parquet_path = meta_rows[0]
    assert paper_id == "smith_2023"
    assert table_id == "T2"
    assert page == 3
    assert "Table 2: AUC by model on CIFAR-10" in caption
    assert "smith_2023__T2.parquet" in parquet_path
    
    # Test table_columns
    column_rows = con.execute("SELECT * FROM table_columns ORDER BY col_index").fetchall()
    assert len(column_rows) == 3
    
    expected_columns = ["model", "dataset", "AUC"]
    for i, (paper_id, table_id, col_index, col_name) in enumerate(column_rows):
        assert paper_id == "smith_2023"
        assert table_id == "T2"
        assert col_index == i
        assert col_name == expected_columns[i]
    
    # Test that parquet file was created
    parquet_files = list(parquet_dir.glob("*.parquet"))
    assert len(parquet_files) == 1
    assert parquet_files[0].name == "smith_2023__T2.parquet"
    
    con.close()


def test_normalize_marker_json_empty_data(tmp_dirs):
    """Test handling of JSON with no table data."""
    import json
    
    marker_json_dir = tmp_dirs["marker_json"]
    json_file = marker_json_dir / "empty_2023.json"
    
    # Create JSON with no tables
    empty_data = [
        {
            "metadata": {"page_number": 1},
            "children": [
                {
                    "block_type": "Text",
                    "id": "text_1",
                    "content": "Just some text, no tables."
                }
            ]
        }
    ]
    
    json_file.write_text(json.dumps(empty_data))
    
    parquet_dir = tmp_dirs["parquet"]
    db_path = str(tmp_dirs["tables_db"])
    
    # Should not raise an error
    normalize_marker_json(json_file, parquet_dir, db_path)
    
    # Check that tables exist but are empty
    con = duckdb.connect(db_path)
    meta_rows = con.execute("SELECT * FROM tables_meta").fetchall()
    assert len(meta_rows) == 0
    
    column_rows = con.execute("SELECT * FROM table_columns").fetchall()
    assert len(column_rows) == 0
    
    con.close()