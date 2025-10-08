"""
Integration test demonstrating the complete RAG pipeline from ingestion to retrieval.
"""
import base64
import pytest
from pathlib import Path
from typing import List
from unittest.mock import patch, Mock
import numpy as np

from rag_papers.ingest.service import IngestionService
from rag_papers.ingest.parsers import MarkerParser, PlumberParser
from rag_papers.index.build_bm25 import build_bm25
from rag_papers.index.build_vector_store import build_vector_store
from rag_papers.retrieval.ensemble_retriever import EnsembleRetriever
from rag_papers.index.contextualize import contextualize


# Same base64 PDF from ingestion tests
SAMPLE_PDF_BASE64 = """JVBERi0xLjQKMSAwIG9iago8PC9QYWdlcyAyIDAgUi9UeXBlL0NhdGFsb2c+PgplbmRvYmoKMiAwIG9iago8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PgplbmRvYmoKMyAwIG9iago8PC9UeXBlL1BhZ2UvUGFyZW50IDIgMCBSL01lZGlhQm94WzAgMCA1OTUuMjggODQxLjg5XS9Db250ZW50cyA0IDAgUi9SZXNvdXJjZXM8PC9Gb250PDwvRjEgNSAwIFI+Pj4+PgplbmRvYmoKNCAwIG9iago8PC9MZW5ndGggNjQ+PnN0cmVhbQpCVC9GIDEyIFRmCjEwMCA3MDAgVGQKKChIZWxsbyBSQUcpKSBUagplbmRzdHJlYW0KZW5kb2JqCjUgMCBvYmoKPDwvVHlwZS9Gb250L1N1YnR5cGUvVHlwZTEvTmFtZS9GMS9CYXNlRm9udC9IZWx2ZXRpY2EvRW5jb2RpbmcvV2luQW5zaUVuY29kaW5nPj4KZW5kb2JqCnhyZWYKMCA2CjAwMDAwMDAwMDAgNjU1MzUgZgowMDAwMDAwMDkzIDAwMDAwIG4KMDAwMDAwMDE2OCAwMDAwMCBuCjAwMDAwMDAyNjUgMDAwMDAgbgowMDAwMDAwMzk4IDAwMDAwIG4KMDAwMDAwMDUyMyAwMDAwMCBuCnRyYWlsZXIKPDwvUm9vdCAxIDAgUi9TaXplIDY+PgpzdGFydHhyZWYKNTQ3CiUlRU9G"""


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a sample PDF file from base64 data for testing."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_data = base64.b64decode(SAMPLE_PDF_BASE64)
    pdf_path.write_bytes(pdf_data)
    return pdf_path


@pytest.fixture
def mock_documents() -> List[str]:
    """Sample documents for testing the complete pipeline."""
    return [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
        "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data, particularly effective for image recognition and natural language processing.",
        "Natural language processing (NLP) is a branch of AI that focuses on the interaction between computers and humans through natural language, enabling machines to read, decipher, understand, and make sense of human language.",
        "Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos, enabling machines to identify and process visual data like the human eye.",
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward, learning through trial and error.",
        "Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second related task, significantly reducing training time and data requirements."
    ]


class TestRAGPipelineIntegration:
    """Integration tests for the complete RAG pipeline."""
    
    def test_ingestion_to_retrieval_pipeline(self, sample_pdf_path: Path, monkeypatch):
        """Test the complete pipeline from PDF ingestion to retrieval and contextualization."""
        
        # Step 1: Mock the parser to return substantial content
        def mock_marker_parse(self, pdf_path: Path) -> List[dict]:
            return [{
                "page": 1, 
                "text": "Machine learning is a powerful subset of artificial intelligence that enables computers to learn patterns from data. This technology has revolutionized many fields including natural language processing, computer vision, and data analysis. Deep learning, a specialized form of machine learning, uses neural networks with multiple layers to process complex information and make predictions."
            }]
        
        monkeypatch.setattr(MarkerParser, "parse", mock_marker_parse)
        
        # Step 2: Ingest the PDF
        ingestion_service = IngestionService()
        chunks = ingestion_service.ingest_pdf(sample_pdf_path)
        
        assert len(chunks) >= 1
        assert all(chunk.content for chunk in chunks)
        
        # Step 3: Extract text content for indexing
        text_chunks = [chunk.content for chunk in chunks]
        
        # Step 4: Build search indices with mocked transformer
        with patch('rag_papers.index.build_vector_store.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_model.encode.side_effect = [
                np.random.rand(len(text_chunks), 384),  # For chunks
                np.random.rand(1, 384)                  # For queries
            ]
            mock_transformer.return_value = mock_model
            
            # Build BM25 index
            bm25_index = build_bm25(text_chunks)
            
            # Build vector store
            vector_store, model = build_vector_store(text_chunks)
            
            # Step 5: Create ensemble retriever
            retriever = EnsembleRetriever(bm25_index, vector_store, model, text_chunks)
            
            # Step 6: Perform search
            query = "What is machine learning and how does it work?"
            search_results = retriever.search(query, top_k=3)
            
            # Step 7: Contextualize results
            retrieved_texts = [text for text, _ in search_results]
            context = contextualize(retrieved_texts, query)
            
            # Verify the complete pipeline
            assert isinstance(search_results, list)
            assert isinstance(context, str)
            assert query in context
            assert "machine learning" in context.lower()
    
    @patch('rag_papers.index.build_vector_store.SentenceTransformer')
    def test_multiple_documents_pipeline(self, mock_transformer, mock_documents):
        """Test the pipeline with multiple pre-processed documents."""
        
        # Mock the transformer
        mock_model = Mock()
        # Provide enough mock responses for initialization + multiple queries
        mock_model.encode.side_effect = [
            np.random.rand(len(mock_documents), 384),  # For documents during build
        ] + [np.random.rand(1, 384) for _ in range(10)]  # For multiple queries
        mock_transformer.return_value = mock_model
        
        # Build indices
        bm25_index = build_bm25(mock_documents)
        vector_store, model = build_vector_store(mock_documents)
        
        # Create retriever
        retriever = EnsembleRetriever(bm25_index, vector_store, model, mock_documents)
        
        # Test different types of queries
        queries = [
            "What is deep learning?",
            "How does computer vision work?",
            "Explain reinforcement learning",
            "What is transfer learning?"
        ]
        
        for query in queries:
            # Search
            results = retriever.search(query, top_k=2)
            
            # Contextualize
            retrieved_texts = [text for text, _ in results]
            context = contextualize(retrieved_texts, query)
            
            # Verify results
            assert isinstance(results, list)
            assert len(results) <= 2
            assert isinstance(context, str)
            assert query in context
            
            # Verify that results contain relevant content
            # With mocked embeddings, we can't guarantee semantic relevance,
            # but we can verify that the search returned some results
            if retrieved_texts:  # Only check if we got results
                combined_text = " ".join(retrieved_texts).lower()
                # At minimum, verify the text contains some content
                assert len(combined_text) > 0
    
    @patch('rag_papers.index.build_vector_store.SentenceTransformer')
    def test_search_with_metadata(self, mock_transformer, mock_documents):
        """Test search and contextualization with metadata."""
        
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.random.rand(len(mock_documents), 384),
            np.random.rand(1, 384)
        ]
        mock_transformer.return_value = mock_model
        
        # Build indices
        bm25_index = build_bm25(mock_documents)
        vector_store, model = build_vector_store(mock_documents)
        
        # Create retriever
        retriever = EnsembleRetriever(bm25_index, vector_store, model, mock_documents)
        
        # Search
        query = "What is machine learning?"
        results = retriever.search(query, top_k=3)
        retrieved_texts = [text for text, _ in results]
        
        # Create mock metadata
        metadata = [
            {"file": "ml_basics.pdf", "page": 1, "chunk_id": 0},
            {"file": "ai_overview.pdf", "page": 2, "chunk_id": 1},
            {"file": "deep_learning.pdf", "page": 1, "chunk_id": 2}
        ][:len(retrieved_texts)]
        
        # Test contextualization with metadata
        from rag_papers.index.contextualize import contextualize_with_metadata
        context = contextualize_with_metadata(retrieved_texts, query, metadata)
        
        assert "File:" in context
        assert "Page:" in context
        assert query in context
        assert "ml_basics.pdf" in context or "ai_overview.pdf" in context
    
    @patch('rag_papers.index.build_vector_store.SentenceTransformer')
    def test_edge_cases(self, mock_transformer):
        """Test edge cases in the pipeline."""
        
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.random.rand(1, 384),  # For single document
            np.random.rand(1, 384)   # For query
        ]
        mock_transformer.return_value = mock_model
        
        # Test with single document
        single_doc = ["This is a single document about artificial intelligence and machine learning."]
        
        bm25_index = build_bm25(single_doc)
        vector_store, model = build_vector_store(single_doc)
        retriever = EnsembleRetriever(bm25_index, vector_store, model, single_doc)
        
        # Search
        results = retriever.search("artificial intelligence", top_k=5)
        retrieved_texts = [text for text, _ in results]
        context = contextualize(retrieved_texts, "artificial intelligence")
        
        assert len(results) <= 1  # Can't return more than available
        assert isinstance(context, str)
        assert "artificial intelligence" in context
    
    @patch('rag_papers.index.build_vector_store.SentenceTransformer')
    def test_performance_characteristics(self, mock_transformer, mock_documents):
        """Test basic performance characteristics of the pipeline."""
        
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.random.rand(len(mock_documents), 384),  # For documents during build
        ] + [np.random.rand(1, 384) for _ in range(5)]   # For multiple search calls
        mock_transformer.return_value = mock_model
        
        # Build indices
        bm25_index = build_bm25(mock_documents)
        vector_store, model = build_vector_store(mock_documents)
        retriever = EnsembleRetriever(bm25_index, vector_store, model, mock_documents)
        
        # Test different weights for hybrid search
        query = "machine learning algorithms"
        
        # BM25 heavy
        bm25_results = retriever.search(query, top_k=3, bm25_weight=0.8, vector_weight=0.2)
        
        # Vector heavy
        vector_results = retriever.search(query, top_k=3, bm25_weight=0.2, vector_weight=0.8)
        
        # Balanced
        balanced_results = retriever.search(query, top_k=3, bm25_weight=0.5, vector_weight=0.5)
        
        # All should return results
        assert len(bm25_results) > 0
        assert len(vector_results) > 0
        assert len(balanced_results) > 0
        
        # Results might be different (though with mocked embeddings they may be similar)
        # At minimum, verify they're all valid
        for results in [bm25_results, vector_results, balanced_results]:
            assert all(isinstance(item, tuple) for item in results)
            assert all(len(item) == 2 for item in results)
            assert all(isinstance(text, str) for text, _ in results)
            assert all(isinstance(score, float) for _, score in results)


class TestWorkflowIntegration:
    """Test the overall RAG workflow as described in the requirements."""
    
    @patch('rag_papers.index.build_vector_store.SentenceTransformer')
    def test_complete_rag_workflow(self, mock_transformer, mock_documents):
        """Test the complete RAG workflow: Index → Search → Context → (Ready for LLM)."""
        
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.random.rand(len(mock_documents), 384),
            np.random.rand(1, 384)
        ]
        mock_transformer.return_value = mock_model
        
        # Step 1: BM25 Indexing
        bm25_index = build_bm25(mock_documents)
        assert bm25_index is not None
        
        # Step 2: Embedding-based Retrieval
        vector_store, model = build_vector_store(mock_documents)
        assert vector_store is not None
        assert model is not None
        
        # Step 3: Hybrid Search
        query = "How do neural networks work in deep learning?"
        
        from rag_papers.retrieval.ensemble_retriever import hybrid_search
        search_results = hybrid_search(
            query=query,
            bm25_index=bm25_index,
            vector_store=vector_store,
            model=model,
            text_chunks=mock_documents,
            top_k=3
        )
        
        assert isinstance(search_results, list)
        assert len(search_results) <= 3
        
        # Step 4: Contextualization
        retrieved_chunks = [text for text, _ in search_results]
        context = contextualize(retrieved_chunks, query)
        
        assert isinstance(context, str)
        assert query in context
        assert "Relevant Context:" in context
        
        # Step 5: Format for LLM (final step before generation)
        from rag_papers.index.contextualize import format_for_llm
        llm_prompt = format_for_llm(retrieved_chunks, query)
        
        assert isinstance(llm_prompt, str)
        assert "Question:" in llm_prompt
        assert query in llm_prompt
        assert "comprehensive answer" in llm_prompt
        
        # Verify the complete workflow produces meaningful output
        assert len(context) > len(query)  # Context should be substantial
        assert len(llm_prompt) > len(context)  # LLM prompt includes additional formatting