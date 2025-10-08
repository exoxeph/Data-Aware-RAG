"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from unittest.mock import Mock, patch, MagicMock

from rag_papers.api.main import app, _docs, _retriever, _generator, _config, get_retriever, get_docs
from rag_papers.retrieval.router_dag import Stage4Config


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    # Override dependencies for testing
    def mock_retriever():
        return Mock()
    
    def mock_docs():
        return ["doc1", "doc2", "doc3"]
    
    app.dependency_overrides[get_retriever] = mock_retriever
    app.dependency_overrides[get_docs] = mock_docs
    
    client = TestClient(app)
    yield client
    
    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture
def mock_run_plan():
    """Mock run_plan function."""
    with patch("rag_papers.api.main.run_plan") as mock:
        yield mock


@pytest.fixture
def mock_plan_for_intent():
    """Mock plan_for_intent function."""
    with patch("rag_papers.api.main.plan_for_intent") as mock:
        yield mock


@pytest.fixture
def mock_materialize():
    """Mock _materialize function."""
    with patch("rag_papers.api.main._materialize") as mock:
        yield mock


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "stage" in data
    assert data["stage"] == "4-ready"


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["stage"] == "4-ready"
    assert "components" in data
    assert isinstance(data["components"], dict)


def test_plan_endpoint(client, mock_plan_for_intent, mock_materialize):
    """Test /plan endpoint."""
    # Setup mocks
    mock_plan_for_intent.return_value = [
        {"name": "retrieve"},
        {"name": "rerank"},
        {"name": "generate"},
        {"name": "verify"},
    ]
    mock_materialize.return_value = {
        "bm25_weight": 0.5,
        "vector_weight": 0.5,
        "top_k": 12,
    }
    
    response = client.post(
        "/plan",
        json={"intent": "definition"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "steps" in data
    assert "intent" in data
    assert "materialized_config" in data
    assert data["intent"] == "definition"
    assert data["steps"] == ["retrieve", "rerank", "generate", "verify"]


def test_plan_endpoint_comparison(client, mock_plan_for_intent, mock_materialize):
    """Test /plan endpoint with comparison intent."""
    mock_plan_for_intent.return_value = [
        {"name": "retrieve"},
        {"name": "rerank"},
        {"name": "prune"},
        {"name": "contextualize"},
        {"name": "generate"},
        {"name": "verify"},
    ]
    mock_materialize.return_value = {"top_k": 12}
    
    response = client.post(
        "/plan",
        json={"intent": "comparison"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "prune" in data["steps"]
    assert "contextualize" in data["steps"]


def test_ask_endpoint_missing_retriever(client):
    """Test /ask endpoint when retriever is not initialized."""
    # Override to return None
    def none_retriever():
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    app.dependency_overrides[get_retriever] = none_retriever
    
    response = client.post(
        "/ask",
        json={
            "query": "What is transfer learning?",
            "intent": "definition",
        },
    )
    
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "Retriever" in data["detail"]
    
    # Clean up override
    app.dependency_overrides.clear()
    app.dependency_overrides[get_retriever] = lambda: Mock()


@patch("rag_papers.api.main.run_plan")
def test_ask_endpoint_success(mock_run_plan, client):
    """Test successful /ask endpoint."""
    # Setup mock
    mock_run_plan.return_value = (
        "Transfer learning is a technique...",
        {
            "path": ["retrieve", "generate", "verify"],
            "verify_score": 0.85,
            "accepted": True,
            "timings": {"retrieve": 100, "generate": 200, "verify": 50},
            "retrieved_count": 10,
            "pruned_chars": 0,
            "context_chars": 4000,
            "sources": ["doc1.pdf", "doc2.pdf"],
        },
    )
    
    response = client.post(
        "/ask",
        json={
            "query": "What is transfer learning?",
            "intent": "definition",
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "score" in data
    assert "accepted" in data
    assert "path" in data
    assert "timings" in data
    
    assert data["answer"] == "Transfer learning is a technique..."
    assert data["score"] == 0.85
    assert data["accepted"] is True
    assert data["path"] == ["retrieve", "generate", "verify"]
    assert data["repair_used"] is False


@patch("rag_papers.api.main.run_plan")
def test_ask_endpoint_with_repair(mock_run_plan, client):
    """Test /ask endpoint with repair step."""
    mock_run_plan.return_value = (
        "Answer after repair",
        {
            "path": ["retrieve", "generate", "verify", "repair"],
            "verify_score": 0.75,
            "accepted": True,
            "timings": {
                "retrieve": 100,
                "generate": 200,
                "verify": 50,
                "repair": 150,
            },
            "retrieved_count": 12,
            "pruned_chars": 500,
            "context_chars": 3500,
            "sources": [],
        },
    )
    
    response = client.post(
        "/ask",
        json={
            "query": "Complex query",
            "intent": "explanation",
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["repair_used"] is True
    assert "repair" in data["path"]


@patch("rag_papers.api.main.run_plan")
def test_ask_endpoint_with_overrides(mock_run_plan, client):
    """Test /ask endpoint with config overrides."""
    mock_run_plan.return_value = (
        "Answer",
        {
            "path": ["retrieve", "generate", "verify"],
            "verify_score": 0.80,
            "accepted": True,
            "timings": {"retrieve": 100, "generate": 200, "verify": 50},
            "retrieved_count": 8,
            "pruned_chars": 0,
            "context_chars": 4000,
            "sources": [],
        },
    )
    
    response = client.post(
        "/ask",
        json={
            "query": "Test query",
            "intent": "definition",
            "bm25_weight": 0.7,
            "vector_weight": 0.3,
            "temperature": 0.5,
            "top_k": 15,
        },
    )
    
    assert response.status_code == 200
    
    # Verify run_plan was called with custom config
    assert mock_run_plan.called
    call_kwargs = mock_run_plan.call_args[1]
    cfg = call_kwargs["cfg"]
    assert cfg.bm25_weight == 0.7
    assert cfg.vector_weight == 0.3
    assert cfg.temperature == 0.5
    assert cfg.top_k == 15


@patch("rag_papers.api.main.run_plan")
def test_ask_endpoint_error_handling(mock_run_plan, client):
    """Test /ask endpoint error handling."""
    # Make run_plan raise an error
    mock_run_plan.side_effect = Exception("Pipeline error")
    
    response = client.post(
        "/ask",
        json={
            "query": "Test query",
            "intent": "definition",
        },
    )
    
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Pipeline error" in data["detail"]


def test_metrics_endpoint(client):
    """Test /metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "accept_at_1" in data
    assert "avg_score" in data
    assert "repair_rate" in data
    assert "recent_results" in data


def test_openapi_docs(client):
    """Test that OpenAPI docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema(client):
    """Test that OpenAPI schema is accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema
    
    # Check that our endpoints are in the schema
    assert "/health" in schema["paths"]
    assert "/plan" in schema["paths"]
    assert "/ask" in schema["paths"]
    assert "/metrics" in schema["paths"]
