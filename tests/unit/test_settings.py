"""Unit test for settings configuration."""

from rag_papers.config.settings import Settings


def test_default_settings():
    """Test that default settings are correctly configured."""
    settings = Settings()
    assert settings.parse.paginate_output is True


def test_paths_configuration():
    """Test that default paths are properly set."""
    settings = Settings()
    assert settings.paths.raw_pdfs == "data/raw_pdfs"
    assert settings.paths.md_out == "data/marker/md"
    assert settings.paths.json_out == "data/marker/json"
    assert settings.paths.figures == "data/figures"
    assert settings.paths.parquet_dir == "data/parquet"
    assert settings.paths.tables_meta_db == "data/tables.duckdb"