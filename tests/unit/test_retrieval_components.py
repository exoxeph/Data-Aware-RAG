"""
Tests for the retrieval and contextualization components.
"""
import pytest
import numpy as np
from typing import List, Tuple
from unittest.mock import Mock, patch, MagicMock

# Import the modules we're testing
from rag_papers.index.build_bm25 import build_bm25, search_bm25, preprocess_text
from rag_papers.index.build_vector_store import (
    build_vector_store, search_vector_store, VectorStore
)
from rag_papers.retrieval.ensemble_retriever import (
    hybrid_search, simple_hybrid_search, EnsembleRetriever
)
from rag_papers.index.contextualize import (
    contextualize, contextualize_with_metadata, format_for_llm
)


@pytest.fixture
def sample_text_chunks() -> List[str]:
    """Sample text chunks for testing."""
    return [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing enables computers to understand and process human language.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Reinforcement learning is a type of machine learning where agents learn through trial and error."
    ]


@pytest.fixture
def sample_query() -> str:
    """Sample query for testing."""
    return "What is machine learning and how does it work?"


class TestBM25:
    """Tests for BM25 indexing and search."""
    
    def test_preprocess_text(self):
        """Test text preprocessing function."""
        text = "Hello, World! This is a TEST."
        tokens = preprocess_text(text)
        
        assert isinstance(tokens, list)
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "," not in tokens  # Punctuation should be removed
        assert all(len(token) > 2 for token in tokens)  # Short words removed
    
    def test_build_bm25_success(self, sample_text_chunks):
        """Test successful BM25 index creation."""
        bm25 = build_bm25(sample_text_chunks)
        
        assert bm25 is not None
        assert hasattr(bm25, 'get_scores')
        assert hasattr(bm25, 'corpus_size')
        assert bm25.corpus_size == len(sample_text_chunks)
    
    def test_build_bm25_empty_chunks(self):
        """Test BM25 index creation with empty chunks."""
        with pytest.raises(ValueError, match="text_chunks cannot be empty"):
            build_bm25([])
    
    def test_build_bm25_with_empty_strings(self):
        """Test BM25 index creation with some empty strings."""
        chunks = ["Valid text", "", "   ", "Another valid text"]
        bm25 = build_bm25(chunks)
        
        assert bm25 is not None
        assert bm25.corpus_size == len(chunks)
    
    def test_search_bm25(self, sample_text_chunks, sample_query):
        """Test BM25 search functionality."""
        bm25 = build_bm25(sample_text_chunks)
        results = search_bm25(bm25, sample_query, sample_text_chunks, top_k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        
        # Check that results are sorted by score (descending)
        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)
    
    def test_search_bm25_empty_query(self, sample_text_chunks):
        """Test BM25 search with empty query."""
        bm25 = build_bm25(sample_text_chunks)
        results = search_bm25(bm25, "", sample_text_chunks)
        
        assert results == []
    
    def test_search_bm25_no_matches(self, sample_text_chunks):
        """Test BM25 search with query that has no matches."""
        bm25 = build_bm25(sample_text_chunks)
        results = search_bm25(bm25, "xyz abc def", sample_text_chunks)
        
        # Should return empty or very low scored results
        assert isinstance(results, list)


class TestVectorStore:
    """Tests for vector store functionality."""
    
    @patch('rag_papers.index.build_vector_store.SentenceTransformer')
    def test_build_vector_store_success(self, mock_transformer, sample_text_chunks):
        """Test successful vector store creation."""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(len(sample_text_chunks), 384)
        mock_transformer.return_value = mock_model
        
        vector_store, model = build_vector_store(sample_text_chunks)
        
        assert isinstance(vector_store, VectorStore)
        assert vector_store.embeddings.shape[0] == len(sample_text_chunks)
        assert len(vector_store.text_chunks) == len(sample_text_chunks)
        assert model == mock_model
    
    def test_build_vector_store_empty_chunks(self):
        """Test vector store creation with empty chunks."""
        with pytest.raises(ValueError, match="text_chunks cannot be empty"):
            build_vector_store([])
    
    @patch('rag_papers.index.build_vector_store.SentenceTransformer')
    def test_build_vector_store_with_empty_strings(self, mock_transformer):
        """Test vector store creation with some empty strings."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(2, 384)
        mock_transformer.return_value = mock_model
        
        chunks = ["Valid text", "", "   ", "Another valid text"]
        vector_store, model = build_vector_store(chunks)
        
        # Should filter out empty chunks
        assert len(vector_store.text_chunks) == 2
        assert vector_store.embeddings.shape[0] == 2
    
    def test_vector_store_search(self):
        """Test vector store search functionality."""
        # Create a simple vector store with known embeddings
        embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        text_chunks = ["chunk1", "chunk2", "chunk3"]
        vector_store = VectorStore(embeddings, text_chunks)
        
        # Query that's most similar to first embedding
        query_embedding = np.array([0.9, 0.1, 0.1])
        results = vector_store.search(query_embedding, top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == "chunk1"  # Most similar
        assert results[0][1] > results[1][1]  # Higher similarity score
    
    @patch('rag_papers.index.build_vector_store.SentenceTransformer')
    def test_search_vector_store(self, mock_transformer):
        """Test vector store search with model encoding."""
        # Setup mocks
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([[1, 0, 0], [0, 1, 0]]),  # For chunks
            np.array([[0.9, 0.1, 0]])          # For query
        ]
        
        embeddings = np.array([[1, 0, 0], [0, 1, 0]])
        text_chunks = ["chunk1", "chunk2"]
        vector_store = VectorStore(embeddings, text_chunks)
        
        results = search_vector_store("test query", vector_store, mock_model, top_k=2)
        
        # Should return at least 1 result (the one with positive similarity)
        assert len(results) >= 1
        assert len(results) <= 2
        mock_model.encode.assert_called_with(["test query"], convert_to_numpy=True)


class TestHybridSearch:
    """Tests for hybrid search functionality."""
    
    @patch('rag_papers.retrieval.ensemble_retriever.search_bm25')
    @patch('rag_papers.retrieval.ensemble_retriever.search_vector_store')
    def test_hybrid_search_success(self, mock_vector_search, mock_bm25_search):
        """Test successful hybrid search."""
        # Mock the search results
        mock_bm25_search.return_value = [("chunk1", 0.8), ("chunk2", 0.6)]
        mock_vector_search.return_value = [("chunk1", 0.9), ("chunk3", 0.7)]
        
        # Create mock objects
        mock_bm25 = Mock()
        mock_vector_store = Mock()
        mock_model = Mock()
        text_chunks = ["chunk1", "chunk2", "chunk3"]
        
        results = hybrid_search(
            "test query", mock_bm25, mock_vector_store, mock_model, text_chunks, top_k=3
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(item, tuple) for item in results)
        
        # Verify that both search methods were called
        mock_bm25_search.assert_called_once()
        mock_vector_search.assert_called_once()
    
    def test_hybrid_search_empty_query(self):
        """Test hybrid search with empty query."""
        mock_bm25 = Mock()
        mock_vector_store = Mock()
        mock_model = Mock()
        text_chunks = ["chunk1", "chunk2"]
        
        results = hybrid_search("", mock_bm25, mock_vector_store, mock_model, text_chunks)
        assert results == []
    
    @patch('rag_papers.retrieval.ensemble_retriever.search_bm25')
    @patch('rag_papers.retrieval.ensemble_retriever.search_vector_store')
    def test_simple_hybrid_search(self, mock_vector_search, mock_bm25_search):
        """Test simple hybrid search that returns only text chunks."""
        mock_bm25_search.return_value = [("chunk1", 0.8)]
        mock_vector_search.return_value = [("chunk1", 0.9)]
        
        mock_bm25 = Mock()
        mock_vector_store = Mock()
        mock_model = Mock()
        text_chunks = ["chunk1", "chunk2"]
        
        results = simple_hybrid_search(
            "test query", mock_bm25, mock_vector_store, mock_model, text_chunks
        )
        
        assert isinstance(results, list)
        assert all(isinstance(item, str) for item in results)
    
    def test_ensemble_retriever(self):
        """Test EnsembleRetriever class."""
        mock_bm25 = Mock()
        mock_vector_store = Mock()
        mock_model = Mock()
        text_chunks = ["chunk1", "chunk2"]
        
        retriever = EnsembleRetriever(mock_bm25, mock_vector_store, mock_model, text_chunks)
        
        assert retriever.bm25_index == mock_bm25
        assert retriever.vector_store == mock_vector_store
        assert retriever.model == mock_model
        assert retriever.text_chunks == text_chunks


class TestContextualization:
    """Tests for contextualization functionality."""
    
    def test_contextualize_success(self, sample_text_chunks, sample_query):
        """Test successful contextualization."""
        context = contextualize(sample_text_chunks[:3], sample_query)
        
        assert isinstance(context, str)
        assert sample_query in context
        assert "Query:" in context
        assert "Relevant Context:" in context
        assert "[Context 1]" in context
        
        # Should include all chunks
        for chunk in sample_text_chunks[:3]:
            assert chunk in context
    
    def test_contextualize_empty_chunks(self, sample_query):
        """Test contextualization with empty chunks."""
        context = contextualize([], sample_query)
        
        assert "No relevant context found" in context
        assert sample_query in context
    
    def test_contextualize_without_query(self, sample_text_chunks):
        """Test contextualization without including query."""
        context = contextualize(sample_text_chunks[:2], "test query", include_query=False)
        
        assert "Query:" not in context
        assert "test query" not in context
        assert "Relevant Context:" in context
        assert "[Context 1]" in context
    
    def test_contextualize_max_length(self, sample_text_chunks, sample_query):
        """Test contextualization with max length constraint."""
        # Use a very small max length
        context = contextualize(sample_text_chunks, sample_query, max_length=200)
        
        assert len(context) <= 200
        assert sample_query in context
        assert "Relevant Context:" in context
    
    def test_contextualize_with_metadata(self, sample_text_chunks, sample_query):
        """Test contextualization with metadata."""
        metadata = [
            {"file": "doc1.pdf", "page": 1},
            {"file": "doc2.pdf", "page": 2},
            {"file": "doc1.pdf", "page": 3}
        ]
        
        context = contextualize_with_metadata(
            sample_text_chunks[:3], sample_query, metadata
        )
        
        assert "File: doc1.pdf" in context
        assert "Page: 1" in context
        assert "File: doc2.pdf" in context
        assert sample_query in context
    
    def test_format_for_llm(self, sample_text_chunks, sample_query):
        """Test LLM-specific formatting."""
        formatted = format_for_llm(sample_text_chunks[:2], sample_query)
        
        assert "Question:" in formatted
        assert sample_query in formatted
        assert "comprehensive answer" in formatted
        assert "Relevant Context:" in formatted
    
    def test_format_for_llm_with_system_prompt(self, sample_text_chunks, sample_query):
        """Test LLM formatting with system prompt."""
        system_prompt = "You are a helpful AI assistant."
        formatted = format_for_llm(
            sample_text_chunks[:2], sample_query, system_prompt
        )
        
        assert system_prompt in formatted
        assert formatted.startswith(system_prompt)


class TestIntegration:
    """Integration tests for the full retrieval pipeline."""
    
    @patch('rag_papers.index.build_vector_store.SentenceTransformer')
    def test_full_pipeline(self, mock_transformer, sample_text_chunks, sample_query):
        """Test the complete retrieval pipeline."""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.random.rand(len(sample_text_chunks), 384),  # For chunks
            np.random.rand(1, 384)                         # For query
        ]
        mock_transformer.return_value = mock_model
        
        # Build indices
        bm25 = build_bm25(sample_text_chunks)
        vector_store, model = build_vector_store(sample_text_chunks)
        
        # Perform hybrid search
        results = hybrid_search(
            sample_query, bm25, vector_store, model, sample_text_chunks, top_k=3
        )
        
        # Extract just the text chunks
        retrieved_chunks = [text for text, _ in results]
        
        # Contextualize
        context = contextualize(retrieved_chunks, sample_query)
        
        # Verify the pipeline worked
        assert isinstance(results, list)
        assert isinstance(context, str)
        assert sample_query in context
        assert "Relevant Context:" in context
    
    def test_ensemble_retriever_integration(self, sample_text_chunks, sample_query):
        """Test EnsembleRetriever integration."""
        with patch('rag_papers.index.build_vector_store.SentenceTransformer') as mock_transformer:
            # Mock the transformer
            mock_model = Mock()
            mock_model.encode.side_effect = [
                np.random.rand(len(sample_text_chunks), 384),
                np.random.rand(1, 384)
            ]
            mock_transformer.return_value = mock_model
            
            # Build components
            bm25 = build_bm25(sample_text_chunks)
            vector_store, model = build_vector_store(sample_text_chunks)
            
            # Create ensemble retriever
            retriever = EnsembleRetriever(bm25, vector_store, model, sample_text_chunks)
            
            # Search
            results = retriever.search(sample_query, top_k=3)
            
            assert isinstance(results, list)
            assert len(results) <= 3
            if results:
                assert all(isinstance(item, tuple) for item in results)
                assert all(len(item) == 2 for item in results)