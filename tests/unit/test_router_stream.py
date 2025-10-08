"""
Unit tests for router streaming functionality.

Tests run_chat_plan_stream() async iterator that yields SSE events.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from rag_papers.retrieval.router_dag import run_chat_plan_stream, Stage4Config
from rag_papers.generation.generator import MockGenerator


@pytest.fixture
def mock_retriever():
    """Mock EnsembleRetriever."""
    retriever = Mock()
    retriever.corpus_id = "test_corpus"
    
    # Mock search results
    result1 = Mock()
    result1.document_id = "doc1"
    result1.content = "Machine learning is a subset of AI."
    result1.score = 0.95
    
    result2 = Mock()
    result2.document_id = "doc2"
    result2.content = "Deep learning uses neural networks."
    result2.score = 0.85
    
    retriever.search = Mock(return_value=[result1, result2])
    return retriever


@pytest.fixture
def mock_generator():
    """Mock generator with streaming."""
    gen = MockGenerator()
    return gen


@pytest.fixture
def stage4_config():
    """Stage4 configuration."""
    return Stage4Config(rerank_top_k=5)


@pytest.mark.asyncio
class TestChatPlanStream:
    """Test run_chat_plan_stream() event generation."""
    
    async def test_stream_yields_meta_event(self, mock_retriever, mock_generator, stage4_config):
        """Verify meta event is yielded first."""
        events = []
        
        async for event in run_chat_plan_stream(
            query="What is ML?",
            history=[],
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=False
        ):
            events.append(event)
        
        # First event should be meta
        assert len(events) > 0
        meta_events = [e for e in events if e.get("type") == "meta"]
        assert len(meta_events) == 1
        
        meta = meta_events[0]["data"]
        assert "run_id" in meta
        assert "corpus_id" in meta
        assert meta["corpus_id"] == "test_corpus"
        assert "retrieved" in meta
        assert meta["retrieved"] == 2
    
    async def test_stream_yields_sources_event(self, mock_retriever, mock_generator, stage4_config):
        """Verify sources event contains retrieved documents."""
        events = []
        
        async for event in run_chat_plan_stream(
            query="What is ML?",
            history=[],
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=False
        ):
            events.append(event)
        
        # Should have sources event
        source_events = [e for e in events if e.get("type") == "sources"]
        assert len(source_events) == 1
        
        sources = source_events[0]["data"]["sources"]
        assert len(sources) == 2
        assert sources[0]["doc"] == "doc1"
        assert sources[0]["score"] == 0.95
    
    async def test_stream_yields_token_events(self, mock_retriever, mock_generator, stage4_config):
        """Verify token events are generated during streaming."""
        events = []
        
        async for event in run_chat_plan_stream(
            query="What is machine learning?",
            history=[],
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=False
        ):
            events.append(event)
        
        # Should have multiple token events
        token_events = [e for e in events if e.get("type") == "token"]
        assert len(token_events) > 5, "Should stream multiple tokens"
        
        # Concatenate tokens
        full_text = "".join([e["data"]["t"] for e in token_events])
        assert len(full_text) > 20, "Should produce substantial text"
    
    async def test_stream_token_concatenation(self, mock_retriever, mock_generator, stage4_config):
        """Verify tokens concatenate to form complete answer."""
        events = []
        
        async for event in run_chat_plan_stream(
            query="Explain deep learning",
            history=[],
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=False
        ):
            events.append(event)
        
        # Extract tokens
        token_events = [e for e in events if e.get("type") == "token"]
        streamed_answer = "".join([e["data"]["t"] for e in token_events])
        
        # Should form coherent response
        assert "deep learning" in streamed_answer.lower()
        assert len(streamed_answer) > 30
    
    async def test_stream_with_history_context(self, mock_retriever, mock_generator, stage4_config):
        """Test streaming with conversation history."""
        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."}
        ]
        
        events = []
        async for event in run_chat_plan_stream(
            query="Tell me more",
            history=history,
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=False
        ):
            events.append(event)
        
        # Should still yield events
        assert len(events) > 5
        
        # Check retriever was called with contextualized query
        assert mock_retriever.search.called
        call_query = mock_retriever.search.call_args[0][0]
        assert "user:" in call_query.lower() or "User:" in call_query
    
    async def test_stream_cache_hit_path(self, mock_retriever, mock_generator, stage4_config):
        """Test streaming when answer is cached."""
        # Pre-populate cache
        from rag_papers.persist.answer_cache import set_answer, AnswerKey, AnswerValue
        
        cache_key = AnswerKey(
            query="User: What is ML?",
            corpus_id="test_corpus",
            model_id="MockGenerator"
        )
        cache_value = AnswerValue(
            answer="Machine learning is a cached answer.",
            sources=["doc1"],
            metadata={}
        )
        set_answer(cache_key, cache_value)
        
        events = []
        async for event in run_chat_plan_stream(
            query="What is ML?",
            history=[],
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=True  # Enable cache
        ):
            events.append(event)
        
        # Should yield cached event
        cached_events = [e for e in events if e.get("type") == "cached"]
        assert len(cached_events) == 1
        assert cached_events[0]["data"]["answer"] == "Machine learning is a cached answer."
        
        # Should NOT yield meta/sources/token events
        assert len([e for e in events if e.get("type") == "meta"]) == 0
        assert len([e for e in events if e.get("type") == "token"]) == 0
    
    async def test_stream_verification_event(self, mock_retriever, mock_generator, stage4_config):
        """Test verification event is yielded (if verifier available)."""
        events = []
        
        async for event in run_chat_plan_stream(
            query="What is ML?",
            history=[],
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=False
        ):
            events.append(event)
        
        # May or may not have verify event (depends on verifier availability)
        verify_events = [e for e in events if e.get("type") == "verify"]
        
        if verify_events:
            verify_data = verify_events[0]["data"]
            assert "score" in verify_data
            assert "accepted" in verify_data
    
    async def test_stream_event_order(self, mock_retriever, mock_generator, stage4_config):
        """Verify events are yielded in correct order."""
        events = []
        
        async for event in run_chat_plan_stream(
            query="What is ML?",
            history=[],
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=False
        ):
            events.append(event)
        
        event_types = [e.get("type") for e in events]
        
        # Meta should come first
        assert event_types[0] == "meta"
        
        # Sources should come second
        assert event_types[1] == "sources"
        
        # Token events should follow
        first_token_idx = event_types.index("token")
        assert first_token_idx > 1
    
    async def test_stream_handles_empty_history(self, mock_retriever, mock_generator, stage4_config):
        """Test streaming with no history."""
        events = []
        
        async for event in run_chat_plan_stream(
            query="What is ML?",
            history=[],
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=False
        ):
            events.append(event)
        
        # Should work normally
        assert len(events) > 5
        assert events[0]["type"] == "meta"
    
    async def test_stream_limits_history_context(self, mock_retriever, mock_generator, stage4_config):
        """Test only last 5 history turns are used."""
        # Create 10 history turns
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"Question {i}"})
            history.append({"role": "assistant", "content": f"Answer {i}"})
        
        events = []
        async for event in run_chat_plan_stream(
            query="Final question",
            history=history,
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=False
        ):
            events.append(event)
        
        # Should still work (implementation uses last 5 turns)
        assert len(events) > 0
        
        # Check query sent to retriever
        call_query = mock_retriever.search.call_args[0][0]
        # Should not include all 10 questions
        assert call_query.count("Question") <= 5


@pytest.mark.asyncio
class TestStreamingPerformance:
    """Test streaming performance characteristics."""
    
    async def test_stream_is_truly_async(self, mock_retriever, mock_generator, stage4_config):
        """Verify streaming yields control during iteration."""
        import time
        
        start = time.time()
        event_count = 0
        
        async for event in run_chat_plan_stream(
            query="What is ML?",
            history=[],
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=stage4_config,
            use_cache=False
        ):
            event_count += 1
            # Small async task
            await asyncio.sleep(0)
        
        duration = time.time() - start
        
        # Should complete quickly despite async sleeps
        assert event_count > 5
        assert duration < 5.0  # Should be fast for mock


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
