"""
Unit tests for rag_papers/persist/index_store.py

Tests save/load indexes with SHA256 checksum verification.
"""
import pytest
import numpy as np
import json
import pickle
from pathlib import Path
from rag_papers.persist.index_store import (
    IndexMeta, SavedIndexes, save_indexes, load_indexes
)
from rag_papers.persist.paths import CorpusPaths


def test_save_and_load_indexes_roundtrip(tmp_path, sample_texts, mock_bm25_index, tiny_vectors):
    """Save and load should preserve index metadata."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    # Create mock vector store
    class MockVectorStore:
        def __init__(self, vectors):
            self.vectors = vectors
    
    vector_store = MockVectorStore(tiny_vectors)
    
    # Create metadata
    meta = IndexMeta(
        corpus_id="test_corpus_123",
        doc_count=len(sample_texts),
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=tiny_vectors.shape[1]
    )
    
    # Save
    saved = save_indexes(corpus_paths, mock_bm25_index, vector_store, sample_texts, meta)
    
    # Load
    loaded_bm25, loaded_vectors, loaded_texts, loaded_meta = load_indexes(corpus_paths)
    
    # Verify metadata
    assert loaded_meta.corpus_id == meta.corpus_id
    assert loaded_meta.doc_count == meta.doc_count
    assert loaded_meta.embed_model_name == meta.embed_model_name
    assert loaded_meta.dim == meta.dim
    
    # Verify texts
    assert loaded_texts == sample_texts
    
    # Verify vectors shape
    assert loaded_vectors.shape == tiny_vectors.shape
    np.testing.assert_array_almost_equal(loaded_vectors, tiny_vectors)


def test_checksum_mismatch_returns_none(tmp_path, sample_texts, mock_bm25_index, tiny_vectors):
    """Tampered files should fail checksum and return None."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    class MockVectorStore:
        def __init__(self, vectors):
            self.vectors = vectors
    
    vector_store = MockVectorStore(tiny_vectors)
    
    meta = IndexMeta(
        corpus_id="test_corpus",
        doc_count=len(sample_texts),
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=tiny_vectors.shape[1]
    )
    
    # Save indexes
    saved = save_indexes(corpus_paths, mock_bm25_index, vector_store, sample_texts, meta)
    
    # Tamper with vectors file
    vectors_path = saved.vectors_path
    tampered_vectors = np.random.rand(*tiny_vectors.shape).astype(np.float32)
    np.savez_compressed(vectors_path, vectors=tampered_vectors)
    
    # Load should return None due to checksum mismatch
    result = load_indexes(corpus_paths)
    assert result is None


def test_corpus_id_roundtrip(tmp_path, sample_texts, mock_bm25_index, tiny_vectors):
    """Corpus ID should be preserved through save/load."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    class MockVectorStore:
        def __init__(self, vectors):
            self.vectors = vectors
    
    vector_store = MockVectorStore(tiny_vectors)
    
    original_corpus_id = "abc123def456"
    
    meta = IndexMeta(
        corpus_id=original_corpus_id,
        doc_count=len(sample_texts),
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=tiny_vectors.shape[1]
    )
    
    save_indexes(corpus_paths, mock_bm25_index, vector_store, sample_texts, meta)
    _, _, _, loaded_meta = load_indexes(corpus_paths)
    
    assert loaded_meta.corpus_id == original_corpus_id


def test_meta_json_has_checksums(tmp_path, sample_texts, mock_bm25_index, tiny_vectors):
    """Meta JSON should contain SHA256 checksums."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    class MockVectorStore:
        def __init__(self, vectors):
            self.vectors = vectors
    
    vector_store = MockVectorStore(tiny_vectors)
    
    meta = IndexMeta(
        corpus_id="test",
        doc_count=len(sample_texts),
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=tiny_vectors.shape[1]
    )
    
    saved = save_indexes(corpus_paths, mock_bm25_index, vector_store, sample_texts, meta)
    
    # Read meta.json
    meta_path = saved.meta_path
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    # Should have checksums
    assert "checksums" in meta_data
    assert "bm25" in meta_data["checksums"]
    assert "vectors" in meta_data["checksums"]
    assert "texts" in meta_data["checksums"]
    
    # Checksums should be 64 characters (SHA256 hex)
    assert len(meta_data["checksums"]["bm25"]) == 64
    assert len(meta_data["checksums"]["vectors"]) == 64
    assert len(meta_data["checksums"]["texts"]) == 64


def test_doc_count_matches(tmp_path, sample_texts, mock_bm25_index, tiny_vectors):
    """Doc count in metadata should match actual text count."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    class MockVectorStore:
        def __init__(self, vectors):
            self.vectors = vectors
    
    vector_store = MockVectorStore(tiny_vectors)
    
    meta = IndexMeta(
        corpus_id="test",
        doc_count=len(sample_texts),
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=tiny_vectors.shape[1]
    )
    
    save_indexes(corpus_paths, mock_bm25_index, vector_store, sample_texts, meta)
    _, _, loaded_texts, loaded_meta = load_indexes(corpus_paths)
    
    assert loaded_meta.doc_count == len(loaded_texts)
    assert loaded_meta.doc_count == len(sample_texts)


def test_empty_corpus(tmp_path, mock_bm25_index):
    """Should handle empty corpus correctly."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    class MockVectorStore:
        def __init__(self):
            self.vectors = np.array([]).reshape(0, 4)
    
    vector_store = MockVectorStore()
    empty_texts = []
    
    meta = IndexMeta(
        corpus_id="empty",
        doc_count=0,
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=4
    )
    
    saved = save_indexes(corpus_paths, mock_bm25_index, vector_store, empty_texts, meta)
    _, loaded_vectors, loaded_texts, loaded_meta = load_indexes(corpus_paths)
    
    assert loaded_meta.doc_count == 0
    assert len(loaded_texts) == 0
    assert loaded_vectors.shape[0] == 0


def test_large_corpus(tmp_path, mock_bm25_index):
    """Should handle large corpus efficiently."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    # Create large synthetic corpus
    np.random.seed(42)
    large_vectors = np.random.rand(1000, 384).astype(np.float32)
    large_texts = [f"Document {i} with some content" for i in range(1000)]
    
    class MockVectorStore:
        def __init__(self, vectors):
            self.vectors = vectors
    
    vector_store = MockVectorStore(large_vectors)
    
    meta = IndexMeta(
        corpus_id="large",
        doc_count=1000,
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=384
    )
    
    save_indexes(corpus_paths, mock_bm25_index, vector_store, large_texts, meta)
    _, loaded_vectors, loaded_texts, loaded_meta = load_indexes(corpus_paths)
    
    assert loaded_meta.doc_count == 1000
    assert len(loaded_texts) == 1000
    assert loaded_vectors.shape == (1000, 384)


def test_unicode_in_texts(tmp_path, mock_bm25_index, tiny_vectors):
    """Should handle Unicode text correctly."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    unicode_texts = [
        "机器学习是人工智能的一个分支",
        "L'apprentissage profond utilise des réseaux",
        "Машинное обучение использует алгоритмы",
        "التعلم العميق يستخدم الشبكات",
        "Deep learning uses neural networks 🧠"
    ]
    
    class MockVectorStore:
        def __init__(self, vectors):
            self.vectors = vectors
    
    vector_store = MockVectorStore(tiny_vectors)
    
    meta = IndexMeta(
        corpus_id="unicode",
        doc_count=len(unicode_texts),
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=tiny_vectors.shape[1]
    )
    
    save_indexes(corpus_paths, mock_bm25_index, vector_store, unicode_texts, meta)
    _, _, loaded_texts, _ = load_indexes(corpus_paths)
    
    assert loaded_texts == unicode_texts


def test_missing_files_returns_none(tmp_path):
    """Loading with missing files should return None."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    
    # No files created
    result = load_indexes(corpus_paths)
    assert result is None


def test_corrupted_json_returns_none(tmp_path, sample_texts, mock_bm25_index, tiny_vectors):
    """Corrupted meta.json should return None gracefully."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    class MockVectorStore:
        def __init__(self, vectors):
            self.vectors = vectors
    
    vector_store = MockVectorStore(tiny_vectors)
    
    meta = IndexMeta(
        corpus_id="test",
        doc_count=len(sample_texts),
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=tiny_vectors.shape[1]
    )
    
    saved = save_indexes(corpus_paths, mock_bm25_index, vector_store, sample_texts, meta)
    
    # Corrupt meta.json
    with open(saved.meta_path, 'w') as f:
        f.write("{invalid json")
    
    # Should return None gracefully
    result = load_indexes(corpus_paths)
    assert result is None


def test_saved_indexes_paths_exist(tmp_path, sample_texts, mock_bm25_index, tiny_vectors):
    """SavedIndexes should contain valid paths."""
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    class MockVectorStore:
        def __init__(self, vectors):
            self.vectors = vectors
    
    vector_store = MockVectorStore(tiny_vectors)
    
    meta = IndexMeta(
        corpus_id="test",
        doc_count=len(sample_texts),
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=tiny_vectors.shape[1]
    )
    
    saved = save_indexes(corpus_paths, mock_bm25_index, vector_store, sample_texts, meta)
    
    # All paths should exist
    assert saved.bm25_path.exists()
    assert saved.vectors_path.exists()
    assert saved.texts_path.exists()
    assert saved.meta_path.exists()
