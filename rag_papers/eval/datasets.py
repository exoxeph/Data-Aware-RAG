"""
Dataset loading utilities for evaluation harness.
"""

import json
from pathlib import Path
from typing import Optional

from .schemas import EvalItem, EvalDataset


def load_dataset(name: str, base_path: Optional[Path] = None) -> EvalDataset:
    """
    Load an evaluation dataset from JSONL file.
    
    Args:
        name: Dataset name (e.g., "qa_small"). Looks for data/eval/{name}.jsonl
        base_path: Optional base path; defaults to repo root detection
        
    Returns:
        EvalDataset with loaded items
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If JSONL is malformed
    """
    if base_path is None:
        # Try to detect repo root by looking for pyproject.toml
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                base_path = parent
                break
        else:
            # Fallback to current working directory
            base_path = Path.cwd()
    
    dataset_path = base_path / "data" / "eval" / f"{name}.jsonl"
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            f"Expected path: data/eval/{name}.jsonl"
        )
    
    items = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(EvalItem(**data))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
            except Exception as e:
                raise ValueError(f"Invalid EvalItem on line {line_num}: {e}") from e
    
    if not items:
        raise ValueError(f"No valid items found in {dataset_path}")
    
    return EvalDataset(name=name, items=items)


def load_dataset_from_path(path: Path) -> EvalDataset:
    """
    Load an evaluation dataset from explicit path.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        EvalDataset with loaded items
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    name = path.stem
    items = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(EvalItem(**data))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
            except Exception as e:
                raise ValueError(f"Invalid EvalItem on line {line_num}: {e}") from e
    
    if not items:
        raise ValueError(f"No valid items found in {path}")
    
    return EvalDataset(name=name, items=items)
