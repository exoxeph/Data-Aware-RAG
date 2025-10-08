"""
Persistence and caching layer for Stage 7.

Provides:
- Stable hashing for content-addressable caching
- Corpus path management and versioning
- SQLite-backed KV store for caches
- Index serialization/deserialization
- Embedding, retrieval, and answer caches
"""

from .hashing import stable_hash
from .paths import CorpusPaths, corpus_id_from_dir, ensure_dirs
from .sqlite_store import KVStore
from .index_store import IndexMeta, SavedIndexes, save_indexes, load_indexes
from .embedding_cache import CachingEncoder
from .retrieval_cache import RetrievalKey, cached_search
from .answer_cache import AnswerKey, AnswerValue, get_answer, set_answer

__all__ = [
    "stable_hash",
    "CorpusPaths",
    "corpus_id_from_dir",
    "ensure_dirs",
    "KVStore",
    "IndexMeta",
    "SavedIndexes",
    "save_indexes",
    "load_indexes",
    "CachingEncoder",
    "RetrievalKey",
    "cached_search",
    "AnswerKey",
    "AnswerValue",
    "get_answer",
    "set_answer",
]
