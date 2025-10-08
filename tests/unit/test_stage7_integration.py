"""
Integration tests for Stage 7 complete cached pipeline.

Tests end-to-end flow with mocked models.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for testing."""
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = MagicMock()
        # Return deterministic vectors
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3, 0.4] for _ in range(5)
        ], dtype=np.float32)
        mock_st.return_value = mock_model
        yield mock_model


@pytest.fixture
def test_app(tmp_path, temp_corpus_dir):
    """Create test FastAPI app with temporary cache."""
    from rag_papers.api.main import app
    from rag_papers.persist.sqlite_store import KVStore
    from rag_papers.ops.jobs import JobManager
    from rag_papers.persist.paths import CorpusPaths
    
    # Setup test paths
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    
    # Initialize test dependencies
    kv = KVStore(cache_dir / "cache.db")
    job_manager = JobManager(state_file=jobs_dir / "jobs.jsonl", max_workers=2)
    corpus_paths = CorpusPaths(
        root=temp_corpus_dir,
        idx_dir=tmp_path / "indexes",
        cache_dir=cache_dir,
        runs_dir=tmp_path / "runs"
    )
    
    # Inject into app state
    app.state.kv_store = kv
    app.state.job_manager = job_manager
    app.state.corpus_paths = corpus_paths
    
    yield app
    
    # Cleanup
    kv.close()


def test_cache_stats_endpoint(test_app):
    """GET /cache/stats should return statistics."""
    client = TestClient(test_app)
    
    response = client.get("/cache/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "stats" in data
    assert isinstance(data["stats"], dict)


def test_cache_purge_endpoint(test_app):
    """POST /cache/purge should clear specified tables."""
    client = TestClient(test_app)
    
    # Add some data first
    kv = test_app.state.kv_store
    kv.set("embeddings", "test_key", b"test_value")
    
    # Purge
    response = client.post(
        "/cache/purge",
        json={"tables": ["embeddings"]}
    )
    assert response.status_code == 200
    
    data = response.json()
    assert data["ok"] is True
    assert "embeddings" in data["purged_tables"]


def test_ingest_start_endpoint(test_app, temp_corpus_dir):
    """POST /ingest/start should create a job."""
    client = TestClient(test_app)
    
    response = client.post(
        "/ingest/start",
        json={"corpus_dir": str(temp_corpus_dir)}
    )
    assert response.status_code == 200
    
    data = response.json()
    assert "job_id" in data
    assert isinstance(data["job_id"], str)


def test_ingest_status_endpoint(test_app, temp_corpus_dir):
    """GET /ingest/status/{job_id} should return job status."""
    client = TestClient(test_app)
    
    # Start a job first
    start_response = client.post(
        "/ingest/start",
        json={"corpus_dir": str(temp_corpus_dir)}
    )
    job_id = start_response.json()["job_id"]
    
    # Check status
    response = client.get(f"/ingest/status/{job_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["job_id"] == job_id
    assert "state" in data
    assert data["state"] in ["queued", "running", "succeeded", "failed"]


def test_ingest_list_endpoint(test_app):
    """GET /ingest/list should return all jobs."""
    client = TestClient(test_app)
    
    response = client.get("/ingest/list")
    assert response.status_code == 200
    
    data = response.json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


@pytest.mark.skip(reason="Requires full RAG stack initialization")
def test_ask_without_cache_then_with_cache(test_app, mock_sentence_transformer):
    """
    Full integration test: First ask (miss) → Second ask (hit).
    
    This test is skipped by default because it requires:
    - Building BM25 and vector indexes
    - Mocking LLM generator
    - Full DAG orchestration
    
    Enable when full stack is available.
    """
    client = TestClient(test_app)
    
    query_data = {
        "query": "What is transfer learning?",
        "intent": "definition",
        "use_cache": True
    }
    
    # First request - cache miss
    response1 = client.post("/ask", json=query_data)
    assert response1.status_code == 200
    
    data1 = response1.json()
    assert "cache" in data1
    assert data1["cache"]["answer_hit"] is False
    
    # Second request - cache hit
    response2 = client.post("/ask", json=query_data)
    assert response2.status_code == 200
    
    data2 = response2.json()
    assert data2["cache"]["answer_hit"] is True
    
    # Answers should be identical
    assert data1["answer"] == data2["answer"]


def test_embedding_cache_integration(kv, mock_sentence_transformer):
    """Test embedding cache with mocked model."""
    from rag_papers.persist.embedding_cache import CachingEncoder
    
    encoder = CachingEncoder(
        mock_sentence_transformer,
        kv,
        "test-model"
    )
    
    texts = ["test text 1", "test text 2"]
    
    # First encode - miss
    vectors1 = encoder.encode(texts)
    assert encoder.miss_count == 2
    assert encoder.hit_count == 0
    
    # Second encode - hit
    vectors2 = encoder.encode(texts)
    assert encoder.miss_count == 2
    assert encoder.hit_count == 2
    
    # Vectors should be identical
    np.testing.assert_array_equal(vectors1, vectors2)


def test_retrieval_cache_integration(kv):
    """Test retrieval cache with mocked retriever."""
    from rag_papers.persist.retrieval_cache import RetrievalKey, cached_search
    from unittest.mock import MagicMock
    
    # Mock retriever
    mock_retriever = MagicMock()
    mock_results = [
        ("Result 1", 0.95),
        ("Result 2", 0.87)
    ]
    mock_retriever.search.return_value = mock_results
    
    key = RetrievalKey(
        query="test query",
        corpus_id="test_corpus",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    # First search - miss
    result1 = cached_search(mock_retriever, key, kv)
    assert mock_retriever.search.call_count == 1
    assert result1 == mock_results
    
    # Second search - hit
    result2 = cached_search(mock_retriever, key, kv)
    assert mock_retriever.search.call_count == 1  # Not called again
    assert result2 == mock_results


def test_answer_cache_integration(kv):
    """Test answer cache roundtrip."""
    from rag_papers.persist.answer_cache import (
        AnswerKey, AnswerValue, get_answer, set_answer
    )
    
    key = AnswerKey(
        query="What is transfer learning?",
        intent="definition",
        corpus_id="test_corpus",
        model="gpt-4",
        cfg={"top_k": 5, "temperature": 0.7}
    )
    
    value = AnswerValue(
        answer="Transfer learning is a machine learning technique...",
        score=0.92,
        accepted=True,
        context_chars=1500,
        path="text->generate",
        timings={"retrieval": 0.5, "generation": 1.2, "total": 1.7}
    )
    
    # First get - miss
    assert get_answer(key, kv) is None
    
    # Set
    set_answer(key, value, kv)
    
    # Second get - hit
    cached_value = get_answer(key, kv)
    assert cached_value is not None
    assert cached_value.answer == value.answer
    assert abs(cached_value.score - value.score) < 1e-6


def test_index_store_integration(tmp_path, sample_texts, tiny_vectors):
    """Test index save/load with real files."""
    from rag_papers.persist.index_store import (
        IndexMeta, save_indexes, load_indexes
    )
    from rag_papers.persist.paths import CorpusPaths
    from unittest.mock import MagicMock
    
    corpus_paths = CorpusPaths(
        root=tmp_path,
        idx_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs"
    )
    corpus_paths.idx_dir.mkdir(parents=True)
    
    # Mock indexes
    mock_bm25 = MagicMock()
    
    class MockVectorStore:
        def __init__(self, vectors):
            self.vectors = vectors
    
    vector_store = MockVectorStore(tiny_vectors)
    
    meta = IndexMeta(
        corpus_id="integration_test",
        doc_count=len(sample_texts),
        built_at="2025-01-15T10:30:00",
        bm25_tokenizer="simple",
        embed_model_name="test-model",
        dim=tiny_vectors.shape[1]
    )
    
    # Save
    saved = save_indexes(corpus_paths, mock_bm25, vector_store, sample_texts, meta)
    assert saved.bm25_path.exists()
    assert saved.vectors_path.exists()
    assert saved.texts_path.exists()
    assert saved.meta_path.exists()
    
    # Load
    loaded_bm25, loaded_vectors, loaded_texts, loaded_meta = load_indexes(corpus_paths)
    
    # Verify
    assert loaded_meta.corpus_id == meta.corpus_id
    assert loaded_texts == sample_texts
    np.testing.assert_array_almost_equal(loaded_vectors, tiny_vectors)


def test_job_manager_integration(tmp_path):
    """Test JobManager with real file persistence."""
    from rag_papers.ops.jobs import JobManager
    import time
    
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    def task():
        time.sleep(0.1)
        return "success"
    
    # Submit job
    job_id = manager.submit(task, job_type="test")
    
    # Should be queued
    job = manager.get(job_id)
    assert job.state in ["queued", "running"]
    
    # Wait for completion
    time.sleep(0.3)
    
    job = manager.get(job_id)
    assert job.state == "succeeded"
    
    # Verify JSONL persistence
    assert state_file.exists()
    content = state_file.read_text()
    assert job_id in content


def test_full_caching_pipeline(tmp_path, sample_texts, tiny_vectors):
    """
    Test complete caching pipeline:
    1. Create indexes with embedding cache
    2. Query with retrieval cache
    3. Generate answer with answer cache
    4. Verify all cache hits on second run
    """
    from rag_papers.persist.sqlite_store import KVStore
    from rag_papers.persist.embedding_cache import CachingEncoder
    from rag_papers.persist.retrieval_cache import RetrievalKey, cached_search
    from rag_papers.persist.answer_cache import (
        AnswerKey, AnswerValue, get_answer, set_answer
    )
    from unittest.mock import MagicMock
    
    # Setup
    kv = KVStore(tmp_path / "cache.db")
    
    # 1. Embedding cache
    mock_model = MagicMock()
    mock_model.encode.return_value = tiny_vectors
    encoder = CachingEncoder(mock_model, kv, "test-model")
    
    # First encode - miss
    encoder.encode(sample_texts)
    assert encoder.miss_count == len(sample_texts)
    
    # Second encode - hit
    encoder.encode(sample_texts)
    assert encoder.hit_count == len(sample_texts)
    
    # 2. Retrieval cache
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [(sample_texts[0], 0.95)]
    
    ret_key = RetrievalKey(
        query="test query",
        corpus_id="test",
        top_k_first=20,
        rerank_top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    
    # First search - miss
    cached_search(mock_retriever, ret_key, kv)
    assert mock_retriever.search.call_count == 1
    
    # Second search - hit
    cached_search(mock_retriever, ret_key, kv)
    assert mock_retriever.search.call_count == 1  # Not called again
    
    # 3. Answer cache
    ans_key = AnswerKey(
        query="test query",
        intent="definition",
        corpus_id="test",
        model="test-model",
        cfg={}
    )
    
    ans_value = AnswerValue(
        answer="Test answer",
        score=0.9,
        accepted=True,
        context_chars=1000,
        path="text->generate",
        timings={"total": 1.5}
    )
    
    # First get - miss
    assert get_answer(ans_key, kv) is None
    
    # Set
    set_answer(ans_key, ans_value, kv)
    
    # Second get - hit
    cached_ans = get_answer(ans_key, kv)
    assert cached_ans is not None
    assert cached_ans.answer == ans_value.answer
    
    kv.close()


def test_corpus_versioning_invalidates_cache(tmp_path, sample_texts):
    """Changing corpus should invalidate caches via corpus_id."""
    from rag_papers.persist.sqlite_store import KVStore
    from rag_papers.persist.answer_cache import (
        AnswerKey, AnswerValue, get_answer, set_answer
    )
    
    kv = KVStore(tmp_path / "cache.db")
    
    # Cache answer for corpus v1
    key_v1 = AnswerKey(
        query="test query",
        intent="definition",
        corpus_id="corpus_v1",
        model="test-model",
        cfg={}
    )
    
    value = AnswerValue(
        answer="Answer from v1",
        score=0.9,
        accepted=True,
        context_chars=1000,
        path="text->generate",
        timings={}
    )
    
    set_answer(key_v1, value, kv)
    
    # Same query but different corpus - should miss cache
    key_v2 = AnswerKey(
        query="test query",
        intent="definition",
        corpus_id="corpus_v2",  # Different corpus
        model="test-model",
        cfg={}
    )
    
    assert get_answer(key_v2, kv) is None
    
    kv.close()
