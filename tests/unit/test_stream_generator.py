"""
Unit tests for generator streaming functionality.

Tests:
- BaseGenerator.stream() default implementation
- MockGenerator.stream() word-by-word streaming
- OllamaGenerator.stream() (mocked)
- Token concatenation equals generate() output
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from rag_papers.generation.generator import (
    BaseGenerator,
    MockGenerator,
    GeneratedResponse,
    GenerationConfig
)


class TestBaseGeneratorStream:
    """Test BaseGenerator streaming interface."""
    
    def test_base_generator_has_stream_method(self):
        """Verify stream() is part of base interface."""
        assert hasattr(BaseGenerator, 'stream')
    
    def test_mock_generator_stream_splits_words(self):
        """Test MockGenerator streams word-by-word."""
        gen = MockGenerator()
        prompt = "Tell me about machine learning"
        
        # Collect streamed tokens
        tokens = list(gen.stream(prompt))
        
        # Should have multiple tokens
        assert len(tokens) > 5, "Should stream multiple tokens"
        
        # Concatenate should form complete answer
        streamed_text = "".join(tokens).strip()
        assert len(streamed_text) > 20, "Should produce substantial text"
        
        # Should contain key phrase from mock response
        assert "machine learning" in streamed_text.lower()
    
    def test_mock_stream_matches_generate(self):
        """Verify streamed text equals generate() output."""
        gen = MockGenerator()
        prompt = "What is deep learning?"
        
        # Get non-streaming response
        generated = gen.generate(prompt)
        
        # Get streaming response
        streamed = "".join(gen.stream(prompt)).strip()
        
        # Should match
        assert streamed == generated.text, "Streamed text should equal generate() output"
    
    def test_mock_stream_timing(self):
        """Test streaming has delay between tokens."""
        gen = MockGenerator()
        prompt = "Explain computer vision"
        
        start = time.time()
        tokens = list(gen.stream(prompt))
        duration = time.time() - start
        
        # Should take at least num_tokens * 0.005s
        min_duration = len(tokens) * 0.003  # Allow some margin
        assert duration >= min_duration, f"Streaming should take time ({duration:.3f}s < {min_duration:.3f}s)"
    
    def test_mock_stream_handles_default_response(self):
        """Test streaming with no keyword match."""
        gen = MockGenerator()
        prompt = "Some random query about nothing specific"
        
        tokens = list(gen.stream(prompt))
        streamed = "".join(tokens).strip()
        
        # Should get default response
        assert "based on the provided context" in streamed.lower()
    
    def test_mock_stream_keyword_matching(self):
        """Test different keywords produce different streams."""
        gen = MockGenerator()
        
        prompts = [
            "What is machine learning?",
            "Explain deep learning",
            "Tell me about computer vision"
        ]
        
        responses = []
        for prompt in prompts:
            tokens = list(gen.stream(prompt))
            responses.append("".join(tokens).strip())
        
        # All should be different
        assert len(set(responses)) == len(responses), "Different keywords should produce different responses"


@patch('requests.post')
class TestOllamaGeneratorStream:
    """Test OllamaGenerator streaming with mocked HTTP."""
    
    def test_ollama_stream_parses_ndjson(self, mock_post):
        """Test Ollama NDJSON stream parsing."""
        from app.adapters.ollama_generator import OllamaGenerator
        
        # Mock successful availability check
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            
            # Create generator
            gen = OllamaGenerator(model="llama3")
        
        # Mock streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello", "done": false}',
            b'{"response": " world", "done": false}',
            b'{"response": "!", "done": true}'
        ]
        mock_post.return_value = mock_response
        
        # Stream
        tokens = list(gen.stream("Test prompt"))
        
        # Verify
        assert tokens == ["Hello", " world", "!"]
        assert mock_post.called
        
        # Check streaming was enabled
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['json']['stream'] is True
    
    def test_ollama_stream_handles_malformed_json(self, mock_post):
        """Test graceful handling of malformed NDJSON."""
        from app.adapters.ollama_generator import OllamaGenerator
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            gen = OllamaGenerator()
        
        # Mock response with malformed line
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "Good", "done": false}',
            b'{invalid json}',  # Should be skipped
            b'{"response": " token", "done": true}'
        ]
        mock_post.return_value = mock_response
        
        tokens = list(gen.stream("Test"))
        
        # Should skip bad line
        assert tokens == ["Good", " token"]
    
    def test_ollama_stream_stops_on_done(self, mock_post):
        """Test streaming stops when done flag is true."""
        from app.adapters.ollama_generator import OllamaGenerator
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            gen = OllamaGenerator()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "Token1", "done": false}',
            b'{"response": "Token2", "done": true}',
            b'{"response": "Token3", "done": false}'  # Should not be yielded
        ]
        mock_post.return_value = mock_response
        
        tokens = list(gen.stream("Test"))
        
        assert len(tokens) == 2
        assert "Token3" not in tokens
    
    def test_ollama_stream_uses_config(self, mock_post):
        """Test streaming respects GenerationConfig."""
        from app.adapters.ollama_generator import OllamaGenerator
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            gen = OllamaGenerator()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "Test", "done": true}'
        ]
        mock_post.return_value = mock_response
        
        config = GenerationConfig(
            temperature=0.9,
            max_new_tokens=1024
        )
        
        list(gen.stream("Test", config))
        
        # Verify config was passed
        call_kwargs = mock_post.call_args[1]
        options = call_kwargs['json']['options']
        assert options['temperature'] == 0.9
        assert options['num_predict'] == 1024
    
    def test_ollama_stream_error_handling(self, mock_post):
        """Test streaming handles API errors."""
        from app.adapters.ollama_generator import OllamaGenerator
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            gen = OllamaGenerator()
        
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="Ollama API returned status 500"):
            list(gen.stream("Test"))


class TestGeneratorStreamComparison:
    """Compare streaming vs non-streaming outputs."""
    
    def test_both_modes_produce_same_output(self):
        """Verify stream() and generate() produce identical text."""
        gen = MockGenerator()
        test_queries = [
            "What is machine learning?",
            "Explain deep learning concepts",
            "How does computer vision work?"
        ]
        
        for query in test_queries:
            # Generate
            generated = gen.generate(query)
            
            # Stream
            streamed = "".join(gen.stream(query)).strip()
            
            assert streamed == generated.text, f"Mismatch for query: {query}"
    
    def test_streaming_is_incremental(self):
        """Verify tokens come incrementally, not all at once."""
        gen = MockGenerator()
        
        tokens_received = []
        start_times = []
        
        start = time.time()
        for token in gen.stream("Tell me about machine learning"):
            tokens_received.append(token)
            start_times.append(time.time() - start)
        
        # Should receive tokens over time
        assert len(tokens_received) > 5
        
        # Timestamps should be increasing (not all zero)
        assert start_times[-1] > start_times[0] + 0.01, "Tokens should arrive over time"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
