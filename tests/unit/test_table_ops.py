"""Unit test for table operations and queries."""

from rag_papers.compute.table_ops import find_auc_for_dataset
from rag_papers.ingest.normalize_tables import normalize_marker_json


def test_find_auc_for_dataset(tmp_dirs, sample_marker_json):
    """Test finding AUC values for a specific dataset."""
    parquet_dir = tmp_dirs["parquet"]
    db_path = str(tmp_dirs["tables_db"])
    
    # First normalize the sample data
    normalize_marker_json(sample_marker_json, parquet_dir, db_path)
    
    # Query for CIFAR-10 dataset
    result = find_auc_for_dataset(db_path, "CIFAR-10")
    
    assert result["paper_id"] == "smith_2023"
    assert "table_id" in result
    assert "page" in result
    assert len(result["rows"]) > 0
    
    # Check that we found AUC values
    auc_row = result["rows"][0]
    assert auc_row["col_name"] == "AUC"
    assert auc_row["value"] == 0.94  # First row should be ResNet-50 with 0.94


def test_find_auc_for_dataset_case_insensitive(tmp_dirs, sample_marker_json):
    """Test that dataset search is case insensitive."""
    parquet_dir = tmp_dirs["parquet"]
    db_path = str(tmp_dirs["tables_db"])
    
    # First normalize the sample data
    normalize_marker_json(sample_marker_json, parquet_dir, db_path)
    
    # Query with different case
    result = find_auc_for_dataset(db_path, "cifar-10")
    
    assert result["paper_id"] == "smith_2023"
    assert len(result["rows"]) > 0


def test_find_auc_for_dataset_not_found(tmp_dirs, sample_marker_json):
    """Test behavior when dataset is not found."""
    parquet_dir = tmp_dirs["parquet"]
    db_path = str(tmp_dirs["tables_db"])
    
    # First normalize the sample data
    normalize_marker_json(sample_marker_json, parquet_dir, db_path)
    
    # Query for non-existent dataset
    result = find_auc_for_dataset(db_path, "ImageNet")
    
    assert result["paper_id"] is None
    assert result["rows"] == []


def test_find_auc_for_dataset_empty_db(tmp_dirs):
    """Test behavior with empty database."""
    db_path = str(tmp_dirs["tables_db"])
    
    # Create empty database structure
    import duckdb
    con = duckdb.connect(db_path)
    con.execute("""CREATE TABLE IF NOT EXISTS tables_meta(
      paper_id TEXT, table_id TEXT, page INT, caption TEXT, parquet_path TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS table_columns(
      paper_id TEXT, table_id TEXT, col_index INT, col_name TEXT)""")
    con.close()
    
    result = find_auc_for_dataset(db_path, "CIFAR-10")
    
    assert result["paper_id"] is None
    assert result["rows"] == []