"""
Embedding cache - wrap SentenceTransformer with SQLite-backed cache.

Caches embeddings by content hash to avoid recomputing.
"""

import numpy as np
from typing import Any, Optional

from .hashing import stable_hash
from .sqlite_store import KVStore


class CachingEncoder:
    """
    Wrapper for SentenceTransformer that caches embeddings.
    
    Key = blake2b(text + model_name + normalize_params)
    Value = float32 vector bytes
    
    Usage:
        >>> from sentence_transformers import SentenceTransformer
        >>> model = SentenceTransformer("all-MiniLM-L6-v2")
        >>> kv = KVStore(Path("data/cache/cache.db"))
        >>> encoder = CachingEncoder(model, kv, "all-MiniLM-L6-v2")
        >>> vectors = encoder.encode(["hello", "world"])
        >>> # Second call hits cache
        >>> vectors2 = encoder.encode(["hello", "world"])
    """
    
    def __init__(
        self,
        model: Any,
        kv: KVStore,
        model_name: str,
        normalize: bool = True,
    ):
        """
        Initialize caching encoder.
        
        Args:
            model: SentenceTransformer or compatible encoder
            kv: KVStore instance
            model_name: Model identifier for cache key
            normalize: Whether embeddings are L2-normalized
        """
        self.model = model
        self.kv = kv
        self.model_name = model_name
        self.normalize = normalize
        
        # Track cache hits/misses
        self.hits = 0
        self.misses = 0
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Include model name and normalize flag in key
        key_data = {
            "text": text,
            "model": self.model_name,
            "normalize": self.normalize,
        }
        return stable_hash(key_data)
    
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode texts with caching.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for model inference
            show_progress_bar: Show progress during encoding
            **kwargs: Additional arguments for model.encode()
        
        Returns:
            NumPy array of shape (len(texts), dim)
        """
        if not texts:
            return np.array([])
        
        # Check cache for each text
        cached_vectors = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached_bytes = self.kv.get("embeddings", key)
            
            if cached_bytes is not None:
                # Cache hit
                self.hits += 1
                vector = np.frombuffer(cached_bytes, dtype=np.float32)
                cached_vectors.append((i, vector))
            else:
                # Cache miss
                self.misses += 1
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Encode uncached texts
        if uncached_texts:
            new_vectors = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=self.normalize,
                **kwargs
            )
            
            # Store in cache
            for text, vector in zip(uncached_texts, new_vectors):
                key = self._cache_key(text)
                vector_bytes = vector.astype(np.float32).tobytes()
                self.kv.set("embeddings", key, vector_bytes)
        else:
            new_vectors = np.array([])
        
        # Reconstruct in original order
        result = np.zeros((len(texts), cached_vectors[0][1].shape[0] if cached_vectors else new_vectors.shape[1]), dtype=np.float32)
        
        # Place cached vectors
        for idx, vector in cached_vectors:
            result[idx] = vector
        
        # Place new vectors
        for i, idx in enumerate(uncached_indices):
            result[idx] = new_vectors[i]
        
        return result
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.hits = 0
        self.misses = 0
