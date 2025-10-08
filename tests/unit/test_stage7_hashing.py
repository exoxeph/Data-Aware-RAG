"""
Unit tests for rag_papers/persist/hashing.py

Tests stable blake2b hashing with JSON canonicalization.
"""
import pytest
from rag_papers.persist.hashing import stable_hash


def test_dict_key_order_independence():
    """Same dict with different key order should produce same hash."""
    dict1 = {"a": 1, "b": 2, "c": 3}
    dict2 = {"c": 3, "a": 1, "b": 2}
    
    hash1 = stable_hash(dict1)
    hash2 = stable_hash(dict2)
    
    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # blake2b(32 bytes) = 64 hex chars


def test_nested_dict_stability():
    """Nested structures should hash consistently."""
    nested1 = {
        "outer": {
            "inner": [1, 2, 3],
            "meta": {"flag": True}
        }
    }
    nested2 = {
        "outer": {
            "meta": {"flag": True},
            "inner": [1, 2, 3]
        }
    }
    
    assert stable_hash(nested1) == stable_hash(nested2)


def test_list_order_matters():
    """Lists with different order should produce different hashes."""
    list1 = [1, 2, 3]
    list2 = [3, 2, 1]
    
    hash1 = stable_hash(list1)
    hash2 = stable_hash(list2)
    
    assert hash1 != hash2


def test_value_change_produces_different_hash():
    """Changing a value should change the hash."""
    dict1 = {"query": "test", "top_k": 10}
    dict2 = {"query": "test", "top_k": 20}
    
    hash1 = stable_hash(dict1)
    hash2 = stable_hash(dict2)
    
    assert hash1 != hash2


def test_empty_dict_and_none():
    """Empty dict should produce consistent hash."""
    empty_hash1 = stable_hash({})
    empty_hash2 = stable_hash({})
    empty_list1 = stable_hash([])
    empty_list2 = stable_hash([])
    
    assert empty_hash1 == empty_hash2
    assert empty_list1 == empty_list2
    assert empty_hash1 != empty_list1  # Empty dict != empty list


def test_string_normalization():
    """Strings should be UTF-8 normalized."""
    str1 = "hello world"
    str2 = "hello world"  # Same string
    
    hash1 = stable_hash(str1)
    hash2 = stable_hash(str2)
    
    assert hash1 == hash2


def test_bytes_passthrough():
    """Bytes should be hashed directly."""
    bytes1 = b"test data"
    bytes2 = b"test data"
    bytes3 = b"different"
    
    hash1 = stable_hash(bytes1)
    hash2 = stable_hash(bytes2)
    hash3 = stable_hash(bytes3)
    
    assert hash1 == hash2
    assert hash1 != hash3


def test_complex_nested_structure():
    """Complex nested structure with multiple types."""
    complex_obj = {
        "query": "What is transfer learning?",
        "intent": "definition",
        "config": {
            "top_k": 10,
            "rerank": True,
            "weights": [0.5, 0.5],
            "model": "all-MiniLM-L6-v2"
        },
        "metadata": {
            "timestamp": None,
            "user_id": 123
        }
    }
    
    # Hash multiple times to ensure consistency
    hashes = [stable_hash(complex_obj) for _ in range(5)]
    assert len(set(hashes)) == 1  # All hashes should be identical


def test_float_precision():
    """Floats should hash consistently."""
    dict1 = {"score": 0.12345678}
    dict2 = {"score": 0.12345678}
    dict3 = {"score": 0.12345679}
    
    hash1 = stable_hash(dict1)
    hash2 = stable_hash(dict2)
    hash3 = stable_hash(dict3)
    
    assert hash1 == hash2
    assert hash1 != hash3


def test_collision_resistance():
    """Different inputs should produce different hashes."""
    inputs = [
        {"a": 1},
        {"a": 2},
        {"b": 1},
        [1, 2, 3],
        [1, 2, 4],
        "test",
        "test2",
        {"num": 123},  # Wrap int in dict
        {"num": 124},
    ]
    
    hashes = [stable_hash(inp) for inp in inputs]
    # All hashes should be unique
    assert len(hashes) == len(set(hashes))


def test_determinism_across_sessions():
    """Hash should be deterministic across multiple calls."""
    test_obj = {
        "query": "test query",
        "config": {"k": 10, "weights": [0.3, 0.7]}
    }
    
    # Generate hash 100 times
    hashes = [stable_hash(test_obj) for _ in range(100)]
    
    # All should be identical
    assert len(set(hashes)) == 1
    assert all(h == hashes[0] for h in hashes)
