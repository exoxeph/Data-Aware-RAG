"""Test configuration and fixtures."""

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def tmp_dirs() -> Generator[dict[str, Path], None, None]:
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dirs = {
            "raw_pdfs": base / "raw_pdfs",
            "marker_md": base / "marker" / "md",
            "marker_json": base / "marker" / "json",
            "parquet": base / "parquet",
            "tables_db": base / "tables.duckdb",
        }
        
        # Create all directories
        for dir_path in dirs.values():
            if isinstance(dir_path, Path) and not str(dir_path).endswith(".duckdb"):
                dir_path.mkdir(parents=True, exist_ok=True)
        
        yield dirs


@pytest.fixture
def sample_marker_json(tmp_dirs) -> Path:
    """Create a fake Marker-style JSON file with test data."""
    marker_json_dir = tmp_dirs["marker_json"]
    json_file = marker_json_dir / "smith_2023.json"
    
    sample_data = [
        {
            "metadata": {"page_number": 3},
            "children": [
                {
                    "block_type": "Text",
                    "id": "text_1",
                    "content": "This is a text block with some content about machine learning models."
                },
                {
                    "block_type": "Table",
                    "id": "T2",
                    "caption": "Table 2: AUC by model on CIFAR-10",
                    "data": {
                        "rows": [
                            ["model", "dataset", "AUC"],
                            ["ResNet-50", "CIFAR-10", "0.94"],
                            ["VGG-16", "CIFAR-10", "0.89"],
                            ["DenseNet", "CIFAR-10", "0.92"]
                        ]
                    }
                }
            ]
        }
    ]
    
    json_file.write_text(json.dumps(sample_data, indent=2))
    return json_file