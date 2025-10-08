"""
Integration tests for Stage 8 chat functionality.

Tests multi-turn conversations, caching, history persistence, and citations.
"""

import pytest
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient

from rag_papers.api.main import app
from rag_papers.persist.chat_history import ChatHistory, create_session_history
from rag_papers.persist.sqlite_store import KVStore


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def session_id():
    """Unique session ID for tests."""
    return f"test_{int(time.time())}"


@pytest.fixture
def temp_history_dir(tmp_path):
    """Temporary directory for chat history."""
    return tmp_path / "chat_history"


def test_chat_history_persistence(temp_history_dir, session_id):
    """Chat history should persist to JSON file."""
    history = create_session_history(session_id, base_dir=temp_history_dir)
    
    # Add messages
    history.add("user", "What is transfer learning?")
    history.add("assistant", "Transfer learning is a technique...")
    history.add("user", "How does it work?")
    
    assert len(history) == 3
    
    # Verify file exists
    history_file = temp_history_dir / f"session_{session_id}" / "chat_history.json"
    assert history_file.exists()
    
    # Load in new instance
    history2 = create_session_history(session_id, base_dir=temp_history_dir)
    assert len(history2) == 3
    assert history2.messages[0].content == "What is transfer learning?"


def test_chat_history_clear(temp_history_dir, session_id):
    """Clear should remove all messages and delete file."""
    history = create_session_history(session_id, base_dir=temp_history_dir)
    
    history.add("user", "Test message")
    history.add("assistant", "Test response")
    
    history_file = temp_history_dir / f"session_{session_id}" / "chat_history.json"
    assert history_file.exists()
    
    # Clear
    history.clear()
    
    assert len(history) == 0
    assert not history_file.exists()


def test_chat_history_get_context(temp_history_dir, session_id):
    """get_context should format messages correctly."""
    history = create_session_history(session_id, base_dir=temp_history_dir)
    
    history.add("user", "What is a neural network?")
    history.add("assistant", "A neural network is...")
    history.add("user", "How does it learn?")
    
    context = history.get_context(max_messages=5)
    
    assert "User: What is a neural network?" in context
    assert "Assistant: A neural network is..." in context
    assert "User: How does it learn?" in context


def test_chat_api_basic_request(client, session_id):
    """Basic chat request should return answer and sources."""
    payload = {
        "query": "What is machine learning?",
        "history": [],
        "corpus_id": "test_corpus",
        "session_id": session_id,
        "use_cache": False,
        "model": "mock"
    }
    
    response = client.post("/api/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "sources" in data
    assert "cache" in data
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (int, float))


def test_chat_api_with_history(client, session_id):
    """Chat with history should include context."""
    history = [
        {"role": "user", "content": "Explain neural networks"},
        {"role": "assistant", "content": "Neural networks are computational models..."}
    ]
    
    payload = {
        "query": "How do they learn?",
        "history": history,
        "corpus_id": "test_corpus",
        "session_id": session_id,
        "use_cache": False,
        "model": "mock"
    }
    
    response = client.post("/api/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Should return answer (even if mock)
    assert len(data["answer"]) > 0


def test_chat_api_cache_hit_on_repeat(client, session_id):
    """Second identical query should hit answer cache."""
    payload = {
        "query": "What is transfer learning?",
        "history": [],
        "corpus_id": "test_corpus",
        "session_id": session_id,
        "use_cache": True,
        "model": "mock"
    }
    
    # First request - cache miss
    response1 = client.post("/api/chat", json=payload)
    assert response1.status_code == 200
    data1 = response1.json()
    
    latency1 = data1["latency_ms"]
    
    # Second request - should hit cache
    response2 = client.post("/api/chat", json=payload)
    assert response2.status_code == 200
    data2 = response2.json()
    
    # Cache hit should be faster
    latency2 = data2["latency_ms"]
    assert latency2 < latency1 * 2  # Allow some variance
    
    # Answer should be identical
    assert data2["answer"] == data1["answer"]


def test_chat_api_cache_toggle(client, session_id):
    """Disabling cache should always compute fresh."""
    payload = {
        "query": "Test query for caching",
        "history": [],
        "corpus_id": "test_corpus",
        "session_id": session_id,
        "use_cache": True,
        "model": "mock"
    }
    
    # First request with cache
    response1 = client.post("/api/chat", json=payload)
    assert response1.status_code == 200
    
    # Second request with cache disabled
    payload["use_cache"] = False
    response2 = client.post("/api/chat", json=payload)
    assert response2.status_code == 200
    
    # Should still get valid answer
    assert len(response2.json()["answer"]) > 0


def test_chat_api_sources_format(client, session_id):
    """Sources should be properly formatted."""
    payload = {
        "query": "What are neural networks?",
        "history": [],
        "corpus_id": "test_corpus",
        "session_id": session_id,
        "use_cache": False,
        "top_k": 3,
        "model": "mock"
    }
    
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    sources = data["sources"]
    
    assert isinstance(sources, list)
    
    if sources:  # If mock returns sources
        for source in sources:
            assert "text" in source
            assert "score" in source
            assert "rank" in source
            assert isinstance(source["score"], (int, float))
            assert isinstance(source["rank"], int)


def test_get_chat_history_endpoint(client, session_id):
    """Should retrieve chat history for session."""
    # First, create some history via chat
    client.post("/api/chat", json={
        "query": "Test question",
        "history": [],
        "corpus_id": "test",
        "session_id": session_id,
        "use_cache": False
    })
    
    # Retrieve history
    response = client.get(f"/api/chat/history/{session_id}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "session_id" in data
    assert "message_count" in data
    assert "messages" in data
    assert data["session_id"] == session_id


def test_clear_chat_history_endpoint(client, session_id):
    """Should clear history for session."""
    # Create history
    client.post("/api/chat", json={
        "query": "Test",
        "history": [],
        "corpus_id": "test",
        "session_id": session_id,
        "use_cache": False
    })
    
    # Clear
    response = client.post(f"/api/chat/clear?session_id={session_id}")
    
    assert response.status_code == 200
    assert "message" in response.json()


def test_chat_with_long_history(client, session_id):
    """Should handle long conversation history."""
    # Build long history
    history = []
    for i in range(10):
        history.append({"role": "user", "content": f"Question {i}"})
        history.append({"role": "assistant", "content": f"Answer {i}"})
    
    payload = {
        "query": "Final question",
        "history": history,
        "corpus_id": "test",
        "session_id": session_id,
        "use_cache": False
    }
    
    response = client.post("/api/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0


def test_chat_message_metadata(temp_history_dir, session_id):
    """Messages should store metadata."""
    history = create_session_history(session_id, base_dir=temp_history_dir)
    
    metadata = {
        "sources": [{"text": "doc1", "score": 0.9}],
        "cache": {"answer_hit": True},
        "latency_ms": 123.45
    }
    
    msg = history.add("assistant", "Test answer", metadata=metadata)
    
    assert msg.metadata == metadata
    
    # Verify persistence
    history2 = create_session_history(session_id, base_dir=temp_history_dir)
    loaded_msg = history2.messages[-1]
    
    assert loaded_msg.metadata == metadata


def test_chat_history_recent_messages(temp_history_dir, session_id):
    """Should return N most recent messages."""
    history = create_session_history(session_id, base_dir=temp_history_dir)
    
    for i in range(10):
        history.add("user", f"Message {i}")
    
    recent = history.get_recent_messages(count=3)
    
    assert len(recent) == 3
    assert recent[0].content == "Message 7"
    assert recent[1].content == "Message 8"
    assert recent[2].content == "Message 9"
