"""
End-to-end integration test for the complete RAG pipeline.
"""
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from rag_papers.generation.prompts import QueryProcessor
from rag_papers.generation.generator import RAGGenerator, MockGenerator
from rag_papers.generation.verifier import ResponseVerifier


class TestEndToEndRAGPipeline:
    """Integration tests for the complete RAG pipeline from query to response."""
    
    def test_complete_rag_workflow(self):
        """Test the complete RAG workflow: query processing -> generation -> verification."""
        # Setup components
        query_processor = QueryProcessor()
        mock_generator = MockGenerator()
        rag_generator = RAGGenerator(generator=mock_generator)
        response_verifier = ResponseVerifier()
        
        # Simulate retrieved context
        context = (
            "Machine learning is a subset of artificial intelligence that enables "
            "computers to learn and improve from experience without being explicitly "
            "programmed. It focuses on developing algorithms that can access data "
            "and use it to learn for themselves. The process involves training a model "
            "on a dataset to make predictions or decisions."
        )
        
        # User query
        user_query = "What is machine learning and how does it work?"
        
        # Step 1: Process the query
        processed_query = query_processor.preprocess_query(user_query)
        
        # Step 2: Generate response using RAG
        generated_response = rag_generator.generate_answer(
            context=context,
            query=user_query,
            processed_query=processed_query
        )
        
        # Step 3: Verify response quality
        quality_score = response_verifier.verify_response(
            response=generated_response.text,
            query=user_query,
            context=context
        )
        
        # Assertions
        assert processed_query.intent.value in ["definition", "explanation"]
        assert len(processed_query.keywords) > 0
        assert "machine" in processed_query.keywords or "learning" in processed_query.keywords
        
        assert generated_response.text is not None
        assert len(generated_response.text) > 50  # Substantial response
        assert generated_response.model_used == "mock_generator"
        assert generated_response.confidence > 0
        
        assert quality_score.overall_score > 0.7  # High quality response
        assert quality_score.relevance_score > 0.3  # Adjusted relevance threshold
        assert len(quality_score.issues) == 0  # No quality issues
        
        # Verify response is acceptable
        assert response_verifier.is_acceptable(quality_score, threshold=0.7)
    
    def test_different_query_types_workflow(self):
        """Test the RAG workflow with different types of queries."""
        # Setup components
        query_processor = QueryProcessor()
        mock_generator = MockGenerator()
        rag_generator = RAGGenerator(generator=mock_generator)
        response_verifier = ResponseVerifier()
        
        # Context about neural networks
        context = (
            "Neural networks are computing systems inspired by biological neural networks. "
            "They consist of interconnected nodes (neurons) that process information. "
            "Deep learning uses multiple layers of neural networks to learn complex patterns. "
            "Convolutional Neural Networks (CNNs) are specialized for image processing, "
            "while Recurrent Neural Networks (RNNs) are designed for sequential data."
        )
        
        test_queries = [
            ("What are neural networks?", "definition"),
            ("Explain how deep learning works", "explanation"),
            ("Compare CNNs and RNNs", "comparison"),
            ("How to implement a neural network?", "how_to"),
            ("Summarize neural network concepts", "summarization")
        ]
        
        for query, expected_category in test_queries:
            # Process query
            processed = query_processor.preprocess_query(query)
            
            # Generate response
            response = rag_generator.generate_answer(context, query, processed)
            
            # Verify response
            quality = response_verifier.verify_response(response.text, query, context)
            
            # Basic assertions for all query types
            assert processed.intent.value in [
                "definition", "explanation", "comparison", "how_to", "summarization", "unknown"
            ]
            assert len(response.text) > 30  # Minimum response length
            assert quality.overall_score > 0.6  # Acceptable quality
            assert response.confidence > 0
    
    def test_workflow_with_empty_context(self):
        """Test RAG workflow when no relevant context is found."""
        # Setup components
        query_processor = QueryProcessor()
        mock_generator = MockGenerator()
        rag_generator = RAGGenerator(generator=mock_generator)
        response_verifier = ResponseVerifier()
        
        query = "What is quantum computing?"
        empty_context = ""
        
        # Process query
        processed = query_processor.preprocess_query(query)
        
        # Generate response with empty context
        response = rag_generator.generate_answer(empty_context, query, processed)
        
        # Verify response
        quality = response_verifier.verify_response(response.text, query, empty_context)
        
        # Should still generate a response, but quality may be lower
        assert response.text is not None
        assert len(response.text) > 0
        assert response.confidence > 0
        assert quality.overall_score >= 0.0  # May be low but not error
    
    def test_workflow_error_handling(self):
        """Test RAG workflow error handling with edge cases."""
        # Setup components
        query_processor = QueryProcessor()
        mock_generator = MockGenerator()
        rag_generator = RAGGenerator(generator=mock_generator)
        response_verifier = ResponseVerifier()
        
        # Test cases with potential issues
        test_cases = [
            ("", "Some context about AI"),  # Empty query
            ("What is AI?", ""),  # Empty context
            ("a", "AI is important"),  # Very short query
            ("What is AI?", "x"),  # Very short context
        ]
        
        for query, context in test_cases:
            # Should not raise exceptions
            processed = query_processor.preprocess_query(query)
            response = rag_generator.generate_answer(context, query, processed)
            quality = response_verifier.verify_response(response.text, query, context)
            
            # Basic validation - should not crash
            assert isinstance(response.text, str)
            assert isinstance(quality.overall_score, (int, float))
            assert 0 <= quality.overall_score <= 1
    
    def test_workflow_with_quality_feedback_loop(self):
        """Test RAG workflow with quality feedback for response improvement."""
        # Setup components
        query_processor = QueryProcessor()
        mock_generator = MockGenerator()
        rag_generator = RAGGenerator(generator=mock_generator)
        response_verifier = ResponseVerifier()
        
        context = "Artificial intelligence involves machine learning and neural networks."
        query = "Explain AI in detail"
        
        # First attempt
        processed = query_processor.preprocess_query(query)
        response1 = rag_generator.generate_answer(context, query, processed)
        quality1 = response_verifier.verify_response(response1.text, query, context)
        
        # Simulate response improvement if quality is low
        if not response_verifier.is_acceptable(quality1, threshold=0.8):
            # Get improvement suggestions
            suggestions = response_verifier.get_improvement_suggestions(quality1)
            assert len(suggestions) > 0
            
            # Simulate regeneration with improved prompt (mock behavior)
            # In a real system, this would modify the prompt based on suggestions
            response2 = rag_generator.generate_answer(context, query, processed)
            quality2 = response_verifier.verify_response(response2.text, query, context)
            
            # Second attempt should maintain or improve quality
            assert quality2.overall_score >= 0.0
    
    def test_workflow_performance_metrics(self):
        """Test RAG workflow and collect performance metrics."""
        import time
        
        # Setup components
        query_processor = QueryProcessor()
        mock_generator = MockGenerator()
        rag_generator = RAGGenerator(generator=mock_generator)
        response_verifier = ResponseVerifier()
        
        context = (
            "Natural language processing (NLP) is a branch of artificial intelligence "
            "that helps computers understand, interpret and manipulate human language. "
            "It involves computational linguistics and machine learning techniques."
        )
        
        query = "What is natural language processing?"
        
        # Measure processing time
        start_time = time.time()
        
        # Process query
        query_start = time.time()
        processed = query_processor.preprocess_query(query)
        query_time = time.time() - query_start
        
        # Generate response
        gen_start = time.time()
        response = rag_generator.generate_answer(context, query, processed)
        gen_time = time.time() - gen_start
        
        # Verify response
        verify_start = time.time()
        quality = response_verifier.verify_response(response.text, query, context)
        verify_time = time.time() - verify_start
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert query_time < 1.0  # Query processing should be fast
        assert gen_time < 5.0  # Generation should be reasonable
        assert verify_time < 1.0  # Verification should be fast
        assert total_time < 10.0  # Total workflow should complete quickly
        
        # Quality assertions
        assert quality.overall_score > 0.6
        assert len(response.text) > 50
        assert response.confidence > 0
        
        # Log metrics for reference
        print(f"\\nPerformance metrics:")
        print(f"Query processing: {query_time:.3f}s")
        print(f"Response generation: {gen_time:.3f}s")
        print(f"Quality verification: {verify_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        print(f"Response quality: {quality.overall_score:.3f}")
        print(f"Response length: {len(response.text)} chars")


class TestRAGPipelineComponents:
    """Test integration between different RAG pipeline components."""
    
    def test_query_processor_generator_integration(self):
        """Test integration between query processor and generator."""
        processor = QueryProcessor()
        mock_gen = MockGenerator()
        rag_gen = RAGGenerator(generator=mock_gen)
        
        query = "How does machine learning work?"
        context = "Machine learning uses algorithms to learn from data."
        
        # Process query
        processed = processor.preprocess_query(query)
        
        # Generate with processed query
        response = rag_gen.generate_answer(context, query, processed)
        
        # Response should reflect the processed query intent
        assert response.text is not None
        assert len(response.text) > 0
        assert processed.intent.value in ["explanation", "how_to", "definition"]
    
    def test_generator_verifier_integration(self):
        """Test integration between generator and response verifier."""
        mock_gen = MockGenerator()
        rag_gen = RAGGenerator(generator=mock_gen)
        verifier = ResponseVerifier()
        
        query = "What is deep learning?"
        context = "Deep learning uses neural networks with multiple layers."
        
        # Generate response
        response = rag_gen.generate_answer(context, query)
        
        # Verify the generated response
        quality = verifier.verify_response(response.text, query, context)
        
        # Quality assessment should be comprehensive
        assert 0 <= quality.overall_score <= 1
        assert 0 <= quality.relevance_score <= 1
        assert 0 <= quality.coherence_score <= 1
        assert 0 <= quality.completeness_score <= 1
        assert isinstance(quality.issues, list)
        assert isinstance(quality.suggestions, list)
    
    def test_full_pipeline_with_mock_retrieval(self):
        """Test full pipeline with mocked retrieval component."""
        # Mock retrieval results
        mock_chunks = [
            {
                "text": "Machine learning is a subset of AI.",
                "score": 0.95,
                "source": "doc1.pdf"
            },
            {
                "text": "ML algorithms learn from data patterns.",
                "score": 0.87,
                "source": "doc2.pdf"
            }
        ]
        
        # Simulate retrieved context
        context = " ".join([chunk["text"] for chunk in mock_chunks])
        
        # Initialize pipeline components
        processor = QueryProcessor()
        mock_gen = MockGenerator()
        rag_gen = RAGGenerator(generator=mock_gen)
        verifier = ResponseVerifier()
        
        query = "What is machine learning?"
        
        # Execute pipeline
        processed = processor.preprocess_query(query)
        response = rag_gen.generate_answer(context, query, processed)
        quality = verifier.verify_response(response.text, query, context)
        
        # Validate end-to-end results
        assert processed.intent.value in ["definition", "explanation"]
        assert response.text is not None
        assert quality.overall_score > 0.5
        assert quality.relevance_score > 0.5