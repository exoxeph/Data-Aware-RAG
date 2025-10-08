"""
Shared fixtures for Stage 7 unit tests.
"""
import pytest
import numpy as np
from pathlib import Path
from rag_papers.persist.sqlite_store import KVStore


@pytest.fixture
def kv(tmp_path):
    """Create a temporary KVStore instance."""
    db_path = tmp_path / "cache.db"
    store = KVStore(db_path)
    yield store
    store.close()


@pytest.fixture
def tiny_vectors():
    """Generate small synthetic vectors for testing."""
    np.random.seed(42)
    return np.random.rand(5, 4).astype("float32")


@pytest.fixture
def sample_texts():
    """Sample text documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple layers of neural networks.",
        "Transfer learning applies pre-trained models to new tasks.",
        "Fine-tuning adapts models to specific domains."
    ]


@pytest.fixture
def mock_bm25_index():
    """Mock BM25 index for testing."""
    from unittest.mock import MagicMock
    mock_index = MagicMock()
    mock_index.get_scores.return_value = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
    mock_index.corpus_size = 5
    return mock_index


@pytest.fixture
def temp_corpus_dir(tmp_path, sample_texts):
    """Create a temporary corpus directory with text files."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    
    for i, text in enumerate(sample_texts):
        file_path = corpus_dir / f"doc_{i}.txt"
        file_path.write_text(text, encoding="utf-8")
    
    return corpus_dir
