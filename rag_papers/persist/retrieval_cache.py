"""
Retrieval cache - cache BM25 + vector search results.

Caches candidate lists by query + config to avoid expensive retrieval.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional

from .hashing import stable_hash
from .sqlite_store import KVStore


@dataclass
class RetrievalKey:
    """Key for retrieval cache."""
    
    query: str
    corpus_id: str
    top_k_first: int
    rerank_top_k: int
    bm25_weight: float
    vector_weight: float
    
    def to_hash(self) -> str:
        """Convert to cache key hash."""
        # Round floats to avoid precision issues
        data = {
            "query": self.query,
            "corpus_id": self.corpus_id,
            "top_k_first": self.top_k_first,
            "rerank_top_k": self.rerank_top_k,
            "bm25_weight": round(self.bm25_weight, 4),
            "vector_weight": round(self.vector_weight, 4),
        }
        return stable_hash(data)


def cached_search(
    retriever: "EnsembleRetriever",
    key: RetrievalKey,
    kv: KVStore,
    use_cache: bool = True,
) -> tuple[list[tuple[str, float]], bool]:
    """
    Search with retrieval caching.
    
    Args:
        retriever: EnsembleRetriever instance
        key: RetrievalKey with query and config
        kv: KVStore instance
        use_cache: Whether to use cache (if False, always runs retrieval)
    
    Returns:
        Tuple of (candidates, cache_hit)
        - candidates: List of (text, score) tuples
        - cache_hit: True if result came from cache
    """
    cache_key = key.to_hash()
    
    # Try cache first
    if use_cache:
        cached_bytes = kv.get("retrieval", cache_key)
        if cached_bytes is not None:
            # Deserialize
            cached_json = cached_bytes.decode("utf-8")
            candidates = json.loads(cached_json)
            # Convert back to tuples
            candidates = [(c["text"], c["score"]) for c in candidates]
            return candidates, True
    
    # Cache miss or disabled - run retrieval
    candidates = retriever.search(
        query=key.query,
        top_k=key.top_k_first,
        bm25_weight=key.bm25_weight,
        vector_weight=key.vector_weight,
    )
    
    # Take top rerank_top_k
    candidates = candidates[:key.rerank_top_k]
    
    # Store in cache
    if use_cache:
        # Serialize to JSON
        candidates_json = json.dumps([
            {"text": text, "score": float(score)}
            for text, score in candidates
        ])
        kv.set("retrieval", cache_key, candidates_json.encode("utf-8"))
    
    return candidates, False


def get_retrieval_stats(kv: KVStore) -> dict:
    """
    Get retrieval cache statistics.
    
    Args:
        kv: KVStore instance
    
    Returns:
        Dict with count, total_bytes, oldest_ts, newest_ts
    """
    return kv.stats("retrieval")


def purge_retrieval_cache(kv: KVStore) -> int:
    """
    Clear all retrieval cache entries.
    
    Args:
        kv: KVStore instance
    
    Returns:
        Number of entries purged
    """
    return kv.purge_table("retrieval")
