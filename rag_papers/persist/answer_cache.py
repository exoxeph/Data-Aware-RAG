"""
Answer cache - cache full answer + verification results.

Caches complete DAG outputs by query + config + model to avoid regeneration.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional

from .hashing import stable_hash
from .sqlite_store import KVStore


@dataclass
class AnswerKey:
    """Key for answer cache."""
    
    query: str
    intent: str
    corpus_id: str
    model: str                  # Generator model name
    cfg: dict                   # Subset of Stage4Config
    
    def to_hash(self) -> str:
        """Convert to cache key hash."""
        # Round floats in config
        cfg_normalized = {}
        for k, v in self.cfg.items():
            if isinstance(v, float):
                cfg_normalized[k] = round(v, 4)
            else:
                cfg_normalized[k] = v
        
        data = {
            "query": self.query,
            "intent": self.intent,
            "corpus_id": self.corpus_id,
            "model": self.model,
            "cfg": cfg_normalized,
        }
        return stable_hash(data)


@dataclass
class AnswerValue:
    """Cached answer with metadata."""
    
    answer: str
    score: float
    accepted: bool
    context_chars: int
    path: list[str]
    timings: dict[str, float]
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "AnswerValue":
        """Create from dict."""
        return cls(**data)


def get_answer(key: AnswerKey, kv: KVStore) -> Optional[AnswerValue]:
    """
    Get cached answer if available.
    
    Args:
        key: AnswerKey with query and config
        kv: KVStore instance
    
    Returns:
        AnswerValue if cached, None otherwise
    """
    cache_key = key.to_hash()
    cached_bytes = kv.get("answers", cache_key)
    
    if cached_bytes is None:
        return None
    
    # Deserialize
    cached_json = cached_bytes.decode("utf-8")
    data = json.loads(cached_json)
    
    return AnswerValue.from_dict(data)


def set_answer(key: AnswerKey, val: AnswerValue, kv: KVStore) -> None:
    """
    Store answer in cache.
    
    Args:
        key: AnswerKey with query and config
        val: AnswerValue to cache
        kv: KVStore instance
    """
    cache_key = key.to_hash()
    
    # Serialize
    data_json = json.dumps(val.to_dict())
    kv.set("answers", cache_key, data_json.encode("utf-8"))


def get_answer_stats(kv: KVStore) -> dict:
    """
    Get answer cache statistics.
    
    Args:
        kv: KVStore instance
    
    Returns:
        Dict with count, total_bytes, oldest_ts, newest_ts
    """
    return kv.stats("answers")


def purge_answer_cache(kv: KVStore) -> int:
    """
    Clear all answer cache entries.
    
    Args:
        kv: KVStore instance
    
    Returns:
        Number of entries purged
    """
    return kv.purge_table("answers")
