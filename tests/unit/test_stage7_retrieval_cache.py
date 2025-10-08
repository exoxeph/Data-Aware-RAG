"""
Unit tests for rag_papers/persist/retrieval_cache.py

Tests retrieval cache determinism and key consistency.
"""
import pytest
import json
from rag_papers.persist.retrieval_cache import RetrievalKey, cached_search
from rag_papers.persist.hashing import stable_hash


def test_same_query_same_key():
    """Same query and corpus should produce identical cache key."""
    key1 = RetrievalKey(
        query="What is transfer learning?",
        corpus_id="abc123",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    key2 = RetrievalKey(
        query="What is transfer learning?",
        corpus_id="abc123",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    hash1 = stable_hash(key1.__dict__)
    hash2 = stable_hash(key2.__dict__)
    
    assert hash1 == hash2


def test_different_query_different_key():
    """Different queries should produce different cache keys."""
    key1 = RetrievalKey(
        query="What is transfer learning?",
        corpus_id="abc123",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    key2 = RetrievalKey(
        query="What is deep learning?",
        corpus_id="abc123",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    hash1 = stable_hash(key1.__dict__)
    hash2 = stable_hash(key2.__dict__)
    
    assert hash1 != hash2


def test_different_corpus_id_different_key():
    """Different corpus IDs should produce different cache keys."""
    key1 = RetrievalKey(
        query="test query",
        corpus_id="corpus_v1",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    key2 = RetrievalKey(
        query="test query",
        corpus_id="corpus_v2",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    hash1 = stable_hash(key1.__dict__)
    hash2 = stable_hash(key2.__dict__)
    
    assert hash1 != hash2


def test_different_weights_different_key():
    """Different weights should produce different cache keys."""
    key1 = RetrievalKey(
        query="test query",
        corpus_id="abc123",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    key2 = RetrievalKey(
        query="test query",
        corpus_id="abc123",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.7,
        vector_weight=0.3
    )
    
    hash1 = stable_hash(key1.__dict__)
    hash2 = stable_hash(key2.__dict__)
    
    assert hash1 != hash2


def test_cached_search_miss_then_hit(kv):
    """First search should miss, second should hit cache."""
    from unittest.mock import MagicMock
    
    # Mock retriever
    mock_retriever = MagicMock()
    mock_results = [
        ("Text about transfer learning", 0.95),
        ("Text about fine-tuning", 0.87),
        ("Text about neural networks", 0.82)
    ]
    mock_retriever.search.return_value = mock_results
    
    key = RetrievalKey(
        query="What is transfer learning?",
        corpus_id="test_corpus",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    # First call - should miss and call retriever
    result1, hit1 = cached_search(mock_retriever, key, kv)
    assert mock_retriever.search.call_count == 1
    assert result1 == mock_results
    assert hit1 is False
    
    # Second call - should hit cache
    result2, hit2 = cached_search(mock_retriever, key, kv)
    assert mock_retriever.search.call_count == 1  # Not called again
    assert result2 == mock_results
    assert hit2 is True


def test_candidate_list_serialization_roundtrip(kv):
    """Candidate list should serialize/deserialize correctly."""
    from unittest.mock import MagicMock
    
    mock_retriever = MagicMock()
    candidates = [
        ("First result with long text", 0.999),
        ("Second result", 0.850),
        ("Third result", 0.750),
        ("Fourth result", 0.650),
        ("Fifth result", 0.550)
    ]
    mock_retriever.search.return_value = candidates
    
    key = RetrievalKey(
        query="test query",
        corpus_id="test",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    # Store
    result1, _ = cached_search(mock_retriever, key, kv)
    
    # Retrieve
    result2, hit = cached_search(mock_retriever, key, kv)
    
    # Compare
    assert hit is True  # Should be cached
    assert len(result1) == len(result2)
    for (text1, score1), (text2, score2) in zip(result1, result2):
        assert text1 == text2
        assert abs(score1 - score2) < 1e-6  # Float precision


def test_float_precision_maintained(kv):
    """Float scores should maintain precision through cache."""
    from unittest.mock import MagicMock
    
    mock_retriever = MagicMock()
    precise_scores = [
        ("Text 1", 0.123456789),
        ("Text 2", 0.987654321),
        ("Text 3", 0.555555555)
    ]
    mock_retriever.search.return_value = precise_scores
    
    key = RetrievalKey(
        query="precision test",
        corpus_id="test",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    # Store and retrieve
    cached_search(mock_retriever, key, kv)
    result, _ = cached_search(mock_retriever, key, kv)
    
    # Check precision
    for (_, original_score), (_, cached_score) in zip(precise_scores, result):
        assert abs(original_score - cached_score) < 1e-6


def test_empty_results_cached(kv):
    """Empty result lists should be cached correctly."""
    from unittest.mock import MagicMock
    
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    
    key = RetrievalKey(
        query="no results query",
        corpus_id="test",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    # First call
    result1, hit1 = cached_search(mock_retriever, key, kv)
    assert result1 == []
    assert hit1 is False
    
    # Second call - should hit cache
    result2, hit2 = cached_search(mock_retriever, key, kv)
    assert result2 == []
    assert hit2 is True
    assert mock_retriever.search.call_count == 1


def test_unicode_text_in_results(kv):
    """Results with Unicode text should cache correctly."""
    from unittest.mock import MagicMock
    
    mock_retriever = MagicMock()
    unicode_results = [
        ("机器学习是人工智能的一个分支 🤖", 0.95),
        ("Deep learning uses neural networks 🧠", 0.87),
        ("Réseau de neurones artificiels 🇫🇷", 0.82)
    ]
    mock_retriever.search.return_value = unicode_results
    
    key = RetrievalKey(
        query="machine learning",
        corpus_id="test",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    # Store
    cached_search(mock_retriever, key, kv)
    
    # Retrieve
    result, hit = cached_search(mock_retriever, key, kv)
    
    assert hit is True
    assert result == unicode_results


def test_large_candidate_list(kv):
    """Large candidate lists should be cached efficiently."""
    from unittest.mock import MagicMock
    
    mock_retriever = MagicMock()
    large_results = [
        (f"Result text {i} with some content", 1.0 - i * 0.01)
        for i in range(100)
    ]
    mock_retriever.search.return_value = large_results
    
    key = RetrievalKey(
        query="test query",
        corpus_id="test",
        top_k_first=100,
        rerank_top_k=10,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    # Store and retrieve
    cached_search(mock_retriever, key, kv)
    result, hit = cached_search(mock_retriever, key, kv)
    
    assert hit is True
    assert len(result) == 100
    assert result == large_results


def test_cache_invalidation_on_config_change(kv):
    """Changing config parameters should create new cache entry."""
    from unittest.mock import MagicMock
    
    mock_retriever = MagicMock()
    mock_retriever.search.side_effect = [
        [("Result 1", 0.9)],
        [("Result 2", 0.8)]
    ]
    
    key1 = RetrievalKey(
        query="test query",
        corpus_id="test",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    key2 = RetrievalKey(
        query="test query",
        corpus_id="test",
        top_k_first=20,
        rerank_top_k=10,  # Different
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    # Both should miss (different keys)
    result1, hit1 = cached_search(mock_retriever, key1, kv)
    result2, hit2 = cached_search(mock_retriever, key2, kv)
    
    assert hit1 is False
    assert hit2 is False
    assert mock_retriever.search.call_count == 2
    assert result1 != result2


def test_key_determinism():
    """Same parameters should always produce same key hash."""
    keys = [
        RetrievalKey(
            query="test query",
            corpus_id="abc123",
            top_k_first=20,
            rerank_top_k=5,
            bm25_weight=0.5,
            vector_weight=0.5
        )
        for _ in range(10)
    ]
    
    hashes = [stable_hash(k.__dict__) for k in keys]
    
    # All hashes should be identical
    assert len(set(hashes)) == 1


def test_special_characters_in_query(kv):
    """Queries with special characters should work correctly."""
    from unittest.mock import MagicMock
    
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [("Result", 0.9)]
    
    key = RetrievalKey(
        query="What is 'transfer learning' & how does it work?",
        corpus_id="test",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    # Should not crash
    result, hit = cached_search(mock_retriever, key, kv)
    assert len(result) == 1
    assert hit is False
    
    # Second call should hit cache
    result2, hit2 = cached_search(mock_retriever, key, kv)
    assert len(result2) == 1
    assert hit2 is True
    assert mock_retriever.search.call_count == 1
