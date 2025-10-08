"""
Path management and corpus versioning.

Provides utilities for organizing corpus data, indexes, caches, and runs.
"""

from dataclasses import dataclass
from pathlib import Path
from .hashing import corpus_hash


@dataclass
class CorpusPaths:
    """Centralized paths for a corpus and its derived data."""
    
    root: Path                 # e.g., data/corpus
    idx_dir: Path              # e.g., data/indexes/{corpus_id}
    cache_dir: Path            # e.g., data/cache
    runs_dir: Path             # e.g., runs/
    
    @property
    def bm25_path(self) -> Path:
        """Path to serialized BM25 index."""
        return self.idx_dir / "bm25.pkl"
    
    @property
    def vectors_path(self) -> Path:
        """Path to vector embeddings."""
        return self.idx_dir / "vectors.npz"
    
    @property
    def texts_path(self) -> Path:
        """Path to corpus texts."""
        return self.idx_dir / "texts.json"
    
    @property
    def meta_path(self) -> Path:
        """Path to index metadata."""
        return self.idx_dir / "meta.json"
    
    @property
    def cache_db_path(self) -> Path:
        """Path to SQLite cache database."""
        return self.cache_dir / "cache.db"


def corpus_id_from_dir(corpus_dir: Path, include_mtime: bool = True) -> str:
    """
    Generate stable corpus ID from directory contents.
    
    Scans all .txt and .md files, computes hash of paths + metadata.
    
    Args:
        corpus_dir: Directory containing corpus files
        include_mtime: Include file modification times in hash
    
    Returns:
        Hex string corpus ID (first 16 chars of hash)
    
    Example:
        >>> corpus_id_from_dir(Path("data/corpus"))
        'a1b2c3d4e5f6g7h8'
    """
    if not corpus_dir.exists():
        return "empty_corpus"
    
    # Scan for text files
    file_paths = []
    for ext in ["*.txt", "*.md"]:
        file_paths.extend(str(p) for p in corpus_dir.rglob(ext))
    
    if not file_paths:
        return "empty_corpus"
    
    # Generate hash
    full_hash = corpus_hash(file_paths, include_mtime=include_mtime)
    
    # Return first 16 chars for readability
    return full_hash[:16]


def ensure_dirs(cp: CorpusPaths) -> None:
    """
    Create all required directories if they don't exist.
    
    Args:
        cp: CorpusPaths instance
    """
    cp.root.mkdir(parents=True, exist_ok=True)
    cp.idx_dir.mkdir(parents=True, exist_ok=True)
    cp.cache_dir.mkdir(parents=True, exist_ok=True)
    cp.runs_dir.mkdir(parents=True, exist_ok=True)


def get_corpus_paths(corpus_dir: Path, base_dir: Path = None) -> CorpusPaths:
    """
    Create CorpusPaths for a given corpus directory.
    
    Args:
        corpus_dir: Directory containing corpus files
        base_dir: Base directory for data (defaults to corpus_dir.parent.parent)
    
    Returns:
        CorpusPaths with appropriate subdirectories
    """
    if base_dir is None:
        base_dir = corpus_dir.parent.parent
    
    corpus_id = corpus_id_from_dir(corpus_dir)
    
    return CorpusPaths(
        root=corpus_dir,
        idx_dir=base_dir / "data" / "indexes" / corpus_id,
        cache_dir=base_dir / "data" / "cache",
        runs_dir=base_dir / "runs",
    )
