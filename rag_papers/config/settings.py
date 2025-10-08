"""Application settings and configuration schema."""

from pydantic import BaseModel


class ParseCfg(BaseModel):
    """Configuration for parsing operations."""
    paginate_output: bool = True


class Paths(BaseModel):
    """File and directory paths configuration."""
    raw_pdfs: str = "data/raw_pdfs"
    md_out: str = "data/marker/md"
    json_out: str = "data/marker/json"
    figures: str = "data/figures"
    parquet_dir: str = "data/parquet"
    tables_meta_db: str = "data/tables.duckdb"


class Settings(BaseModel):
    """Main application settings."""
    parse: ParseCfg = ParseCfg()
    paths: Paths = Paths()