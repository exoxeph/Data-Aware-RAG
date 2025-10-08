"""
Stable hashing utilities for content-addressable caching.

Provides deterministic hashing of dicts, lists, strings, and bytes.
Uses JSON canonicalization for dicts/lists and UTF-8 normalization for strings.
"""

import hashlib
import json
from typing import Any


def stable_hash(obj: dict | list | str | bytes) -> str:
    """
    Compute stable hash of an object.
    
    - Dicts: sorted by keys, then JSON-serialized
    - Lists: recursively hashed, then JSON-serialized
    - Strings: UTF-8 normalized
    - Bytes: used directly
    
    Returns:
        64-character hex string (blake2b)
    
    Examples:
        >>> stable_hash({"b": 2, "a": 1})
        >>> stable_hash({"a": 1, "b": 2})  # Same hash
        >>> stable_hash(["x", "y"])
        >>> stable_hash("hello world")
    """
    if isinstance(obj, dict):
        # Sort keys for determinism
        canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        data = canonical.encode("utf-8")
    elif isinstance(obj, list):
        # Recursively hash nested structures
        canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        data = canonical.encode("utf-8")
    elif isinstance(obj, str):
        # UTF-8 normalize (NFC form)
        import unicodedata
        normalized = unicodedata.normalize("NFC", obj)
        data = normalized.encode("utf-8")
    elif isinstance(obj, bytes):
        data = obj
    else:
        raise TypeError(f"Cannot hash type {type(obj)}: {obj}")
    
    # Use blake2b for fast, secure hashing
    h = hashlib.blake2b(data, digest_size=32)
    return h.hexdigest()


def corpus_hash(file_paths: list[str], include_mtime: bool = True) -> str:
    """
    Compute stable hash for a corpus based on file paths and optional metadata.
    
    Args:
        file_paths: Sorted list of file paths
        include_mtime: Whether to include file modification times
    
    Returns:
        Hex hash representing the corpus version
    """
    import os
    
    # Sort for determinism
    sorted_paths = sorted(file_paths)
    
    if include_mtime:
        # Include size and mtime for change detection
        metadata = []
        for path in sorted_paths:
            if os.path.exists(path):
                stat = os.stat(path)
                metadata.append({
                    "path": path,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
            else:
                metadata.append({"path": path, "size": 0, "mtime": 0})
        return stable_hash(metadata)
    else:
        # Just paths
        return stable_hash(sorted_paths)
