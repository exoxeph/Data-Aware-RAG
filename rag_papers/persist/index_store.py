"""
Index persistence - save/load BM25 and vector stores to disk.

Provides versioned serialization with integrity checks.
"""

import hashlib
import json
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Union

import numpy as np

from .paths import CorpusPaths


@dataclass
class IndexMeta:
    """Metadata for saved indexes."""
    
    corpus_id: str              # Hash of corpus content
    doc_count: int              # Number of documents
    built_at: str               # ISO timestamp
    bm25_tokenizer: str         # Tokenizer used for BM25
    embed_model_name: str       # Sentence transformer model
    dim: int                    # Embedding dimension
    version: str = "1.0"        # Serialization format version
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "IndexMeta":
        """Create from dict."""
        return cls(**data)


@dataclass
class SavedIndexes:
    """Paths to saved index files."""
    
    bm25_path: Path
    vectors_path: Path
    texts_path: Path
    meta_path: Path


def _compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def save_indexes(
    idx_dir: Union[Path, CorpusPaths],
    bm25: Any,
    vector_store: Any,
    texts: list[str],
    meta: IndexMeta,
) -> SavedIndexes:
    """
    Save BM25 index, vector store, and texts to disk with metadata.
    
    Args:
        idx_dir: Directory to save indexes (Path or CorpusPaths with .idx_dir)
        bm25: BM25 index object (will be pickled)
        vector_store: Vector store object with .vectors attribute
        texts: List of document texts
        meta: Index metadata
    
    Returns:
        SavedIndexes with paths to all files
    
    File format:
        - bm25.pkl: Pickled BM25 object
        - vectors.npz: Compressed numpy array (float32)
        - texts.json: JSON list of texts
        - meta.json: Metadata with checksums
    """
    # Support both Path and CorpusPaths
    if hasattr(idx_dir, 'idx_dir'):
        idx_dir = idx_dir.idx_dir
    
    idx_dir.mkdir(parents=True, exist_ok=True)
    
    bm25_path = idx_dir / "bm25.pkl"
    vectors_path = idx_dir / "vectors.npz"
    texts_path = idx_dir / "texts.json"
    meta_path = idx_dir / "meta.json"
    
    # Save BM25 (pickle)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save vectors (compressed numpy)
    if hasattr(vector_store, "vectors"):
        vectors = vector_store.vectors
    else:
        # Assume it's already an array
        vectors = vector_store
    
    np.savez_compressed(
        vectors_path,
        vectors=vectors.astype(np.float32)
    )
    
    # Save texts (JSON)
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    
    # Compute checksums
    checksums = {
        "bm25": _compute_sha256(bm25_path),
        "vectors": _compute_sha256(vectors_path),
        "texts": _compute_sha256(texts_path),
    }
    
    # Save metadata with checksums
    meta_dict = meta.to_dict()
    meta_dict["checksums"] = checksums
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    
    return SavedIndexes(
        bm25_path=bm25_path,
        vectors_path=vectors_path,
        texts_path=texts_path,
        meta_path=meta_path,
    )


def load_indexes(
    idx_dir: Union[Path, CorpusPaths],
    verify_checksums: bool = True,
) -> tuple[Optional[Any], Optional[np.ndarray], Optional[list[str]], Optional[IndexMeta]]:
    """
    Load BM25 index, vectors, texts, and metadata from disk.
    
    Args:
        idx_dir: Directory containing saved indexes (Path or CorpusPaths with .idx_dir)
        verify_checksums: Whether to verify SHA256 checksums
    
    Returns:
        Tuple of (bm25, vectors, texts, meta) or (None, None, None, None) if not found
    
    Raises:
        ValueError: If checksums don't match (when verify_checksums=True)
    """
    # Support both Path and CorpusPaths
    if hasattr(idx_dir, 'idx_dir'):
        idx_dir = idx_dir.idx_dir
    
    bm25_path = idx_dir / "bm25.pkl"
    vectors_path = idx_dir / "vectors.npz"
    texts_path = idx_dir / "texts.json"
    meta_path = idx_dir / "meta.json"
    
    # Check if all files exist
    if not all(p.exists() for p in [bm25_path, vectors_path, texts_path, meta_path]):
        return None, None, None, None
    
    # Load metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    
    checksums = meta_dict.pop("checksums", {})
    meta = IndexMeta.from_dict(meta_dict)
    
    # Verify checksums if requested
    if verify_checksums and checksums:
        for name, expected in checksums.items():
            if name == "bm25":
                actual = _compute_sha256(bm25_path)
            elif name == "vectors":
                actual = _compute_sha256(vectors_path)
            elif name == "texts":
                actual = _compute_sha256(texts_path)
            else:
                continue
            
            if actual != expected:
                raise ValueError(
                    f"Checksum mismatch for {name}: "
                    f"expected {expected}, got {actual}"
                )
    
    # Load BM25
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    
    # Load vectors
    vectors_data = np.load(vectors_path)
    vectors = vectors_data["vectors"]
    
    # Load texts
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    
    return bm25, vectors, texts, meta
