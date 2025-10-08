"""
Unit tests for rag_papers/persist/embedding_cache.py

Tests CachingEncoder wrapper with mock SentenceTransformer.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, Mock, patch
from rag_papers.persist.embedding_cache import CachingEncoder


def test_first_encode_miss_inserts_cache(kv):
    """First encode should be a cache miss and insert into cache."""
    # Mock the underlying model
    mock_model = MagicMock()
    mock_vectors = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    mock_model.encode.return_value = mock_vectors
    
    encoder = CachingEncoder(mock_model, kv, "test-model")
    
    # First encode - should be miss
    texts = ["test text"]
    result = encoder.encode(texts)
    
    # Model should have been called
    mock_model.encode.assert_called_once()
    assert encoder.miss_count == 1
    assert encoder.hit_count == 0
    
    # Result should match mock output
    np.testing.assert_array_equal(result, mock_vectors)


def test_second_encode_hit_no_model_call(kv):
    """Second encode of same text should hit cache and not call model."""
    mock_model = MagicMock()
    mock_vectors = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    mock_model.encode.return_value = mock_vectors
    
    encoder = CachingEncoder(mock_model, kv, "test-model")
    
    # First call - miss
    texts = ["test text"]
    result1 = encoder.encode(texts)
    assert mock_model.encode.call_count == 1
    
    # Second call - should hit cache
    result2 = encoder.encode(texts)
    assert mock_model.encode.call_count == 1  # Not called again!
    assert encoder.hit_count == 1
    assert encoder.miss_count == 1
    
    # Results should be identical
    np.testing.assert_array_equal(result1, result2)


def test_cache_hit_miss_counts_accurate(kv):
    """Hit and miss counts should be accurate across multiple calls."""
    mock_model = MagicMock()
    
    def mock_encode(texts, **kwargs):
        # Return unique vectors for each text
        return np.array([[i * 0.1] * 4 for i, _ in enumerate(texts)], dtype=np.float32)
    
    mock_model.encode.side_effect = mock_encode
    
    encoder = CachingEncoder(mock_model, kv, "test-model")
    
    # Encode 3 unique texts
    encoder.encode(["text1"])  # miss
    encoder.encode(["text2"])  # miss
    encoder.encode(["text3"])  # miss
    
    assert encoder.miss_count == 3
    assert encoder.hit_count == 0
    
    # Encode previously seen texts
    encoder.encode(["text1"])  # hit
    encoder.encode(["text2"])  # hit
    encoder.encode(["text1"])  # hit again
    
    assert encoder.miss_count == 3
    assert encoder.hit_count == 3


def test_different_model_name_separate_cache(kv):
    """Different model names should use separate cache keys."""
    mock_model = MagicMock()
    mock_vectors = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    mock_model.encode.return_value = mock_vectors
    
    encoder1 = CachingEncoder(mock_model, kv, "model-v1")
    encoder2 = CachingEncoder(mock_model, kv, "model-v2")
    
    text = ["same text"]
    
    # Both should miss (different model names)
    encoder1.encode(text)
    encoder2.encode(text)
    
    assert encoder1.miss_count == 1
    assert encoder2.miss_count == 1
    assert mock_model.encode.call_count == 2


def test_batch_encoding_with_mixed_cache_state(kv):
    """Batch with some cached and some new texts should work correctly."""
    mock_model = MagicMock()
    
    def mock_encode(texts, **kwargs):
        return np.array([[i * 0.1] * 4 for i, _ in enumerate(texts)], dtype=np.float32)
    
    mock_model.encode.side_effect = mock_encode
    
    encoder = CachingEncoder(mock_model, kv, "test-model")
    
    # First encode some texts
    texts1 = ["text1", "text2"]
    result1 = encoder.encode(texts1)
    assert encoder.miss_count == 2
    
    # Encode batch with one cached, one new
    texts2 = ["text1", "text3"]
    result2 = encoder.encode(texts2)
    
    # Should have 1 hit (text1) and 1 miss (text3)
    assert encoder.hit_count == 1
    assert encoder.miss_count == 3


def test_empty_text_list(kv):
    """Encoding empty list should return empty array."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([], dtype=np.float32).reshape(0, 4)
    
    encoder = CachingEncoder(mock_model, kv, "test-model")
    
    result = encoder.encode([])
    
    assert result.shape[0] == 0


def test_cache_persistence_across_instances(kv):
    """Cache should persist across different encoder instances."""
    mock_model1 = MagicMock()
    mock_vectors = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    mock_model1.encode.return_value = mock_vectors
    
    # First encoder instance
    encoder1 = CachingEncoder(mock_model1, kv, "test-model")
    encoder1.encode(["test text"])
    assert encoder1.miss_count == 1
    
    # Second encoder instance with same kv and model name
    mock_model2 = MagicMock()
    encoder2 = CachingEncoder(mock_model2, kv, "test-model")
    encoder2.encode(["test text"])
    
    # Should hit cache from previous instance
    assert encoder2.hit_count == 1
    mock_model2.encode.assert_not_called()


def test_whitespace_normalization(kv):
    """Texts with different whitespace should be treated as different."""
    mock_model = MagicMock()
    
    def mock_encode(texts, **kwargs):
        return np.array([[0.1] * 4 for _ in texts], dtype=np.float32)
    
    mock_model.encode.side_effect = mock_encode
    
    encoder = CachingEncoder(mock_model, kv, "test-model")
    
    # These should be treated as different texts
    encoder.encode(["test text"])
    encoder.encode(["test  text"])  # Double space
    
    # Both should miss (different strings)
    assert encoder.miss_count == 2


def test_large_batch_encoding(kv):
    """Should handle large batches efficiently."""
    mock_model = MagicMock()
    
    batch_size = 100
    mock_vectors = np.random.rand(batch_size, 4).astype(np.float32)
    mock_model.encode.return_value = mock_vectors
    
    encoder = CachingEncoder(mock_model, kv, "test-model")
    
    texts = [f"text_{i}" for i in range(batch_size)]
    result = encoder.encode(texts)
    
    assert result.shape == (batch_size, 4)
    assert encoder.miss_count == batch_size


def test_unicode_text_encoding(kv):
    """Should handle Unicode texts correctly."""
    mock_model = MagicMock()
    mock_vectors = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    mock_model.encode.return_value = mock_vectors
    
    encoder = CachingEncoder(mock_model, kv, "test-model")
    
    # Unicode text with emoji and Chinese characters
    texts = ["Hello 世界 🌍"]
    result = encoder.encode(texts)
    
    assert result.shape == (1, 4)
    assert encoder.miss_count == 1
    
    # Second call should hit cache
    result2 = encoder.encode(texts)
    assert encoder.hit_count == 1


def test_vector_dtype_preserved(kv):
    """Vector data type should be preserved in cache."""
    mock_model = MagicMock()
    mock_vectors = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    mock_model.encode.return_value = mock_vectors
    
    encoder = CachingEncoder(mock_model, kv, "test-model")
    
    # First encode
    result1 = encoder.encode(["test"])
    assert result1.dtype == np.float32
    
    # Second encode (from cache)
    result2 = encoder.encode(["test"])
    assert result2.dtype == np.float32


def test_cache_key_includes_model_name(kv):
    """Cache key should incorporate model name to avoid collisions."""
    mock_model = MagicMock()
    mock_vectors = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    mock_model.encode.return_value = mock_vectors
    
    text = ["test text"]
    
    # Encode with two different model names
    encoder1 = CachingEncoder(mock_model, kv, "model-a")
    encoder1.encode(text)
    
    encoder2 = CachingEncoder(mock_model, kv, "model-b")
    encoder2.encode(text)
    
    # Both should miss (different cache keys)
    assert encoder1.miss_count == 1
    assert encoder2.miss_count == 1
    assert mock_model.encode.call_count == 2
