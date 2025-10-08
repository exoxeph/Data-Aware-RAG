"""
Unit tests for rag_papers/persist/answer_cache.py

Tests full answer caching with AnswerKey/AnswerValue serialization.
"""
import pytest
import json
from rag_papers.persist.answer_cache import (
    AnswerKey, AnswerValue, get_answer, set_answer
)
from rag_papers.persist.hashing import stable_hash


def test_answer_key_to_dict():
    """AnswerKey should convert to dict for hashing."""
    key = AnswerKey(
        query="What is transfer learning?",
        intent="definition",
        corpus_id="abc123",
        model="gpt-4",
        cfg={
            "top_k": 10,
            "temperature": 0.7,
            "max_tokens": 500
        }
    )
    
    key_dict = key.__dict__
    assert key_dict["query"] == "What is transfer learning?"
    assert key_dict["intent"] == "definition"
    assert key_dict["corpus_id"] == "abc123"
    assert key_dict["model"] == "gpt-4"
    assert "top_k" in key_dict["cfg"]


def test_answer_value_roundtrip_serialization(kv):
    """AnswerValue should serialize and deserialize correctly."""
    key = AnswerKey(
        query="test query",
        intent="definition",
        corpus_id="test_corpus",
        model="test-model",
        cfg={"top_k": 5}
    )
    
    original_value = AnswerValue(
        answer="Transfer learning is a technique...",
        score=0.92,
        accepted=True,
        context_chars=1500,
        path="text->generate",
        timings={"retrieval": 0.5, "generation": 1.2}
    )
    
    # Store
    set_answer(key, original_value, kv)
    
    # Retrieve
    retrieved_value = get_answer(key, kv)
    
    # Compare
    assert retrieved_value is not None
    assert retrieved_value.answer == original_value.answer
    assert abs(retrieved_value.score - original_value.score) < 1e-6
    assert retrieved_value.accepted == original_value.accepted
    assert retrieved_value.context_chars == original_value.context_chars
    assert retrieved_value.path == original_value.path
    assert retrieved_value.timings == original_value.timings


def test_get_nonexistent_answer(kv):
    """Getting nonexistent answer should return None."""
    key = AnswerKey(
        query="nonexistent query",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={}
    )
    
    result = get_answer(key, kv)
    assert result is None


def test_answer_hit_detection(kv):
    """Second get should indicate cache hit."""
    key = AnswerKey(
        query="test query",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={"top_k": 5}
    )
    
    value = AnswerValue(
        answer="Test answer",
        score=0.9,
        accepted=True,
        context_chars=1000,
        path="text->generate",
        timings={"total": 1.5}
    )
    
    # First get - miss
    assert get_answer(key, kv) is None
    
    # Store
    set_answer(key, value, kv)
    
    # Second get - hit
    cached = get_answer(key, kv)
    assert cached is not None
    assert cached.answer == value.answer


def test_different_query_different_cache_entry(kv):
    """Different queries should create separate cache entries."""
    value = AnswerValue(
        answer="Test answer",
        score=0.9,
        accepted=True,
        context_chars=1000,
        path="text->generate",
        timings={}
    )
    
    key1 = AnswerKey(
        query="query 1",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={}
    )
    
    key2 = AnswerKey(
        query="query 2",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={}
    )
    
    # Store for key1
    set_answer(key1, value, kv)
    
    # Should not exist for key2
    assert get_answer(key2, kv) is None


def test_different_model_different_cache_entry(kv):
    """Different models should create separate cache entries."""
    value = AnswerValue(
        answer="Test answer",
        score=0.9,
        accepted=True,
        context_chars=1000,
        path="text->generate",
        timings={}
    )
    
    key1 = AnswerKey(
        query="same query",
        intent="definition",
        corpus_id="test",
        model="gpt-3.5",
        cfg={}
    )
    
    key2 = AnswerKey(
        query="same query",
        intent="definition",
        corpus_id="test",
        model="gpt-4",
        cfg={}
    )
    
    set_answer(key1, value, kv)
    
    # Should not hit cache with different model
    assert get_answer(key2, kv) is None


def test_different_corpus_id_different_cache_entry(kv):
    """Different corpus IDs should create separate cache entries."""
    value = AnswerValue(
        answer="Test answer",
        score=0.9,
        accepted=True,
        context_chars=1000,
        path="text->generate",
        timings={}
    )
    
    key1 = AnswerKey(
        query="same query",
        intent="definition",
        corpus_id="corpus_v1",
        model="test-model",
        cfg={}
    )
    
    key2 = AnswerKey(
        query="same query",
        intent="definition",
        corpus_id="corpus_v2",
        model="test-model",
        cfg={}
    )
    
    set_answer(key1, value, kv)
    
    # Should not hit cache with different corpus
    assert get_answer(key2, kv) is None


def test_config_change_different_cache_entry(kv):
    """Different config should create separate cache entries."""
    value = AnswerValue(
        answer="Test answer",
        score=0.9,
        accepted=True,
        context_chars=1000,
        path="text->generate",
        timings={}
    )
    
    key1 = AnswerKey(
        query="same query",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={"top_k": 5}
    )
    
    key2 = AnswerKey(
        query="same query",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={"top_k": 10}
    )
    
    set_answer(key1, value, kv)
    
    # Should not hit cache with different config
    assert get_answer(key2, kv) is None


def test_float_precision_in_timings(kv):
    """Float precision in timings should be maintained."""
    key = AnswerKey(
        query="test",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={}
    )
    
    value = AnswerValue(
        answer="Test",
        score=0.123456789,
        accepted=True,
        context_chars=1000,
        path="test",
        timings={
            "retrieval": 0.987654321,
            "generation": 1.234567890,
            "total": 2.222222222
        }
    )
    
    set_answer(key, value, kv)
    retrieved = get_answer(key, kv)
    
    assert abs(retrieved.score - value.score) < 1e-6
    for key_name in value.timings:
        assert abs(retrieved.timings[key_name] - value.timings[key_name]) < 1e-6


def test_unicode_in_answer(kv):
    """Unicode text in answer should be preserved."""
    key = AnswerKey(
        query="What is machine learning?",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={}
    )
    
    value = AnswerValue(
        answer="机器学习 (Machine Learning) est l'apprentissage automatique 🤖",
        score=0.95,
        accepted=True,
        context_chars=1500,
        path="text->generate",
        timings={"total": 1.5}
    )
    
    set_answer(key, value, kv)
    retrieved = get_answer(key, kv)
    
    assert retrieved.answer == value.answer


def test_long_answer_text(kv):
    """Long answers should be stored and retrieved correctly."""
    key = AnswerKey(
        query="test",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={}
    )
    
    # Create a long answer (5000 characters)
    long_answer = "Transfer learning is a technique... " * 200
    
    value = AnswerValue(
        answer=long_answer,
        score=0.9,
        accepted=True,
        context_chars=10000,
        path="text->generate",
        timings={"total": 2.5}
    )
    
    set_answer(key, value, kv)
    retrieved = get_answer(key, kv)
    
    assert retrieved.answer == long_answer
    assert len(retrieved.answer) > 5000


def test_empty_timings_dict(kv):
    """Empty timings dict should be handled correctly."""
    key = AnswerKey(
        query="test",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={}
    )
    
    value = AnswerValue(
        answer="Test answer",
        score=0.9,
        accepted=True,
        context_chars=1000,
        path="text->generate",
        timings={}
    )
    
    set_answer(key, value, kv)
    retrieved = get_answer(key, kv)
    
    assert retrieved.timings == {}


def test_overwrite_cached_answer(kv):
    """Overwriting a cached answer should update the value."""
    key = AnswerKey(
        query="test",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={}
    )
    
    value1 = AnswerValue(
        answer="First answer",
        score=0.8,
        accepted=True,
        context_chars=1000,
        path="text->generate",
        timings={}
    )
    
    value2 = AnswerValue(
        answer="Updated answer",
        score=0.95,
        accepted=True,
        context_chars=1500,
        path="text->generate",
        timings={}
    )
    
    set_answer(key, value1, kv)
    set_answer(key, value2, kv)
    
    retrieved = get_answer(key, kv)
    assert retrieved.answer == "Updated answer"
    assert abs(retrieved.score - 0.95) < 1e-6


def test_cache_key_determinism():
    """Same AnswerKey parameters should produce same cache key."""
    keys = [
        AnswerKey(
            query="test query",
            intent="definition",
            corpus_id="abc123",
            model="gpt-4",
            cfg={"top_k": 5, "temp": 0.7}
        )
        for _ in range(10)
    ]
    
    hashes = [stable_hash(k.__dict__) for k in keys]
    
    # All should be identical
    assert len(set(hashes)) == 1


def test_none_values_in_answer_value(kv):
    """AnswerValue with zero/empty fields should work correctly."""
    key = AnswerKey(
        query="test",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={}
    )
    
    value = AnswerValue(
        answer="Test answer",
        score=0.0,  # Zero score instead of None
        accepted=False,
        context_chars=0,
        path="",
        timings={}
    )
    
    set_answer(key, value, kv)
    retrieved = get_answer(key, kv)
    
    assert retrieved.answer == "Test answer"
    assert retrieved.score == 0.0
    assert retrieved.accepted is False
