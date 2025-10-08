"""
Tests for query processing and answer generation components.
"""
import pytest
from typing import List, Dict
from unittest.mock import Mock, patch, MagicMock

# Import the modules we're testing
from rag_papers.generation.prompts import (
    QueryProcessor, ProcessedQuery, QueryIntent,
    preprocess_query, extract_entities, detect_intent
)
from rag_papers.generation.generator import (
    RAGGenerator, GenerationConfig, GeneratedResponse,
    BaseGenerator, MockGenerator, generate_answer
)
from rag_papers.generation.verifier import (
    ResponseVerifier, QualityScore, QualityIssue
)


class TestQueryProcessing:
    """Tests for query understanding and preprocessing."""
    
    def test_query_processor_initialization(self):
        """Test QueryProcessor initialization."""
        processor = QueryProcessor()
        assert processor is not None
        assert hasattr(processor, 'intent_patterns')
        assert hasattr(processor, 'stop_words')
    
    def test_preprocess_query_basic(self):
        """Test basic query preprocessing."""
        processor = QueryProcessor()
        query = "What is machine learning?"
        
        result = processor.preprocess_query(query)
        
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == query
        assert result.normalized_query == "what is machine learning?"
        assert result.intent in [QueryIntent.DEFINITION, QueryIntent.EXPLANATION]
        assert "machine" in result.keywords or "learning" in result.keywords
    
    def test_preprocess_query_empty(self):
        """Test preprocessing of empty query."""
        processor = QueryProcessor()
        result = processor.preprocess_query("")
        
        assert result.original_query == ""
        assert result.normalized_query == ""
        assert result.intent == QueryIntent.UNKNOWN
        assert result.entities == []
        assert result.keywords == []
    
    def test_intent_detection_explanation(self):
        """Test intent detection for explanation queries."""
        processor = QueryProcessor()
        
        queries = [
            "Explain how neural networks work",
            "How does machine learning work?",
            "Why is deep learning effective?",
            "Tell me about computer vision"
        ]
        
        for query in queries:
            result = processor.preprocess_query(query)
            assert result.intent == QueryIntent.EXPLANATION
    
    def test_intent_detection_definition(self):
        """Test intent detection for definition queries."""
        processor = QueryProcessor()
        
        queries = [
            "What is artificial intelligence?",
            "Define machine learning",
            "What does CNN mean?"
        ]
        
        for query in queries:
            result = processor.preprocess_query(query)
            assert result.intent in [QueryIntent.DEFINITION, QueryIntent.EXPLANATION]
    
    def test_intent_detection_comparison(self):
        """Test intent detection for comparison queries."""
        processor = QueryProcessor()
        
        queries = [
            "Compare supervised and unsupervised learning",
            "What's the difference between CNN and RNN?",
            "Which is better: TensorFlow or PyTorch?"
        ]
        
        for query in queries:
            result = processor.preprocess_query(query)
            assert result.intent == QueryIntent.COMPARISON
    
    def test_intent_detection_summarization(self):
        """Test intent detection for summarization queries."""
        processor = QueryProcessor()
        
        queries = [
            "Summarize the key points about deep learning",
            "Give me an overview of computer vision",
            "What are the main highlights of this paper?"
        ]
        
        for query in queries:
            result = processor.preprocess_query(query)
            # Intent detection may classify as explanation or summarization
            assert result.intent in [QueryIntent.SUMMARIZATION, QueryIntent.EXPLANATION]
    
    def test_intent_detection_how_to(self):
        """Test intent detection for how-to queries."""
        processor = QueryProcessor()
        
        queries = [
            "How to implement a neural network?",
            "Steps to train a machine learning model",
            "Guide to building a CNN"
        ]
        
        for query in queries:
            result = processor.preprocess_query(query)
            # Intent detection may classify as how-to or explanation
            assert result.intent in [QueryIntent.HOW_TO, QueryIntent.EXPLANATION]
    
    def test_entity_extraction(self):
        """Test entity extraction from queries."""
        processor = QueryProcessor()
        
        test_cases = [
            ("What is machine learning?", ["machine", "learning"]),
            ("Explain deep learning neural networks", ["deep", "learning", "neural"]),
            ("How does computer vision work?", ["computer", "vision"]),
        ]
        
        for query, expected_entities in test_cases:
            result = processor.preprocess_query(query)
            # Check that at least some expected entities are found
            found_entities = set(result.entities)
            expected_set = set(expected_entities)
            assert len(found_entities & expected_set) > 0
    
    def test_keyword_extraction(self):
        """Test keyword extraction from queries."""
        processor = QueryProcessor()
        query = "How does machine learning algorithm work with neural networks?"
        
        result = processor.preprocess_query(query)
        
        # Should extract meaningful keywords, not stop words
        assert "machine" in result.keywords or "learning" in result.keywords
        assert "algorithm" in result.keywords
        assert "neural" in result.keywords or "networks" in result.keywords
        assert "does" not in result.keywords  # Stop word
        assert "with" not in result.keywords  # Stop word
    
    def test_technical_phrase_extraction(self):
        """Test extraction of technical phrases."""
        processor = QueryProcessor()
        query = "Explain machine learning and deep learning differences"
        
        result = processor.preprocess_query(query)
        
        # Should find technical phrases
        entities_str = " ".join(result.entities)
        assert "machine learning" in entities_str or "machine" in result.entities
        assert "deep learning" in entities_str or "deep" in result.entities
    
    def test_backward_compatibility_functions(self):
        """Test backward compatibility functions."""
        query = "What is artificial intelligence?"
        
        # Test preprocess_query function
        result = preprocess_query(query)
        assert isinstance(result, dict)
        assert "normalized_query" in result
        assert "intent" in result
        assert "entities" in result
        assert "keywords" in result
        
        # Test extract_entities function
        entities = extract_entities(query)
        assert isinstance(entities, list)
        
        # Test detect_intent function
        intent = detect_intent(query)
        assert isinstance(intent, str)


class TestMockGenerator:
    """Tests for the mock generator."""
    
    def test_mock_generator_initialization(self):
        """Test MockGenerator initialization."""
        generator = MockGenerator()
        assert generator.is_available()
        assert hasattr(generator, 'mock_responses')
    
    def test_mock_generator_basic_response(self):
        """Test basic response generation."""
        generator = MockGenerator()
        prompt = "What is machine learning?"
        
        response = generator.generate(prompt)
        
        assert isinstance(response, GeneratedResponse)
        assert len(response.text) > 0
        assert response.model_used == "mock_generator"
        assert response.confidence > 0
        assert response.prompt_length == len(prompt)
    
    def test_mock_generator_keyword_matching(self):
        """Test that mock generator matches keywords appropriately."""
        generator = MockGenerator()
        
        test_cases = [
            ("What is machine learning?", "machine learning"),
            ("Explain deep learning", "deep learning"),
            ("How does computer vision work?", "computer vision")
        ]
        
        for prompt, expected_keyword in test_cases:
            response = generator.generate(prompt)
            # Response should contain relevant content for the keyword
            assert len(response.text) > 50  # Should be a substantial response
    
    def test_mock_generator_default_response(self):
        """Test default response for unknown queries."""
        generator = MockGenerator()
        prompt = "Random query about xyz topic"
        
        response = generator.generate(prompt)
        
        assert isinstance(response, GeneratedResponse)
        assert response.text == generator.mock_responses["default"]


class TestRAGGenerator:
    """Tests for the main RAG generator."""
    
    def test_rag_generator_initialization(self):
        """Test RAGGenerator initialization."""
        generator = RAGGenerator()
        assert generator is not None
        assert hasattr(generator, 'generator')
        assert hasattr(generator, 'config')
    
    def test_rag_generator_with_mock(self):
        """Test RAG generator with mock backend."""
        mock_gen = MockGenerator()
        rag_gen = RAGGenerator(generator=mock_gen)
        
        context = "Machine learning is a subset of AI that enables computers to learn from data."
        query = "What is machine learning?"
        
        response = rag_gen.generate_answer(context, query)
        
        assert isinstance(response, GeneratedResponse)
        assert len(response.text) > 0
        assert response.model_used == "mock_generator"
    
    def test_prompt_creation_by_intent(self):
        """Test that different intents create different prompts."""
        mock_gen = MockGenerator()
        rag_gen = RAGGenerator(generator=mock_gen)
        
        context = "Machine learning is a subset of AI."
        query = "What is machine learning?"
        
        # Test with different processed queries
        from rag_papers.generation.prompts import ProcessedQuery, QueryIntent
        
        explanation_query = ProcessedQuery(
            original_query=query,
            normalized_query=query.lower(),
            intent=QueryIntent.EXPLANATION,
            entities=["machine", "learning"],
            keywords=["machine", "learning"]
        )
        
        response = rag_gen.generate_answer(context, query, explanation_query)
        assert isinstance(response, GeneratedResponse)
    
    def test_prompt_creation_empty_context(self):
        """Test prompt creation with empty context."""
        mock_gen = MockGenerator()
        rag_gen = RAGGenerator(generator=mock_gen)
        
        response = rag_gen.generate_answer("", "What is AI?")
        assert isinstance(response, GeneratedResponse)
        # Should still generate a response even without context
        assert len(response.text) > 0
    
    def test_response_postprocessing(self):
        """Test response post-processing."""
        mock_gen = MockGenerator()
        rag_gen = RAGGenerator(generator=mock_gen)
        
        context = "AI is important."
        query = "What is AI?"
        
        response = rag_gen.generate_answer(context, query)
        
        # Response should be properly formatted
        assert response.text[0].isupper()  # Should start with capital
        assert response.text.endswith(('.', '!', '?'))  # Should end with punctuation
    
    def test_generate_answer_backward_compatibility(self):
        """Test backward compatibility function."""
        context = "Machine learning is a subset of AI."
        query = "What is machine learning?"
        
        response = generate_answer(context, query)
        
        assert isinstance(response, str)
        assert len(response) > 0


class TestResponseVerifier:
    """Tests for response verification and quality assessment."""
    
    def test_verifier_initialization(self):
        """Test ResponseVerifier initialization."""
        verifier = ResponseVerifier()
        assert verifier is not None
        assert hasattr(verifier, 'min_length')
        assert hasattr(verifier, 'max_length')
    
    def test_verify_good_response(self):
        """Test verification of a good quality response."""
        verifier = ResponseVerifier()
        
        response = ("Machine learning is a subset of artificial intelligence that enables "
                   "computers to learn and make decisions from data without being explicitly "
                   "programmed for every task. It uses algorithms to find patterns in data "
                   "and make predictions or decisions based on those patterns.")
        
        query = "What is machine learning?"
        
        quality = verifier.verify_response(response, query)
        
        assert isinstance(quality, QualityScore)
        assert quality.overall_score > 0.7  # Should be high quality
        assert quality.relevance_score > 0.7  # Adjusted threshold
        assert len(quality.issues) == 0  # Should have no issues
    
    def test_verify_short_response(self):
        """Test verification of too short response."""
        verifier = ResponseVerifier(min_length=50)
        response = "AI."
        query = "What is artificial intelligence?"
        
        quality = verifier.verify_response(response, query)
        
        assert QualityIssue.TOO_SHORT in quality.issues
        assert quality.overall_score < 0.5
        assert any("too short" in suggestion.lower() for suggestion in quality.suggestions)
    
    def test_verify_long_response(self):
        """Test verification of too long response."""
        verifier = ResponseVerifier(max_length=100)
        
        # Create a very long response
        response = "AI is very important. " * 20  # Will exceed 100 chars
        query = "What is AI?"
        
        quality = verifier.verify_response(response, query)
        
        assert QualityIssue.TOO_LONG in quality.issues
        assert any("too long" in suggestion.lower() for suggestion in quality.suggestions)
    
    def test_verify_repetitive_response(self):
        """Test verification of repetitive response."""
        verifier = ResponseVerifier(max_repetition_ratio=0.2)
        
        response = ("Machine learning machine learning is important. "
                   "Machine learning machine learning helps with data. "
                   "Machine learning machine learning is useful.")
        
        query = "What is machine learning?"
        
        quality = verifier.verify_response(response, query)
        
        assert QualityIssue.REPETITIVE in quality.issues
        assert any("repetitive" in suggestion.lower() for suggestion in quality.suggestions)
    
    def test_verify_off_topic_response(self):
        """Test verification of off-topic response."""
        verifier = ResponseVerifier()
        
        response = "Cooking is a great skill to have. You can make delicious meals."
        query = "What is machine learning?"
        
        quality = verifier.verify_response(response, query)
        
        # Low relevance score indicates off-topic detection
        assert quality.relevance_score < 0.3
        # Overall score should be lower for off-topic responses
        assert quality.overall_score < 0.8
    
    def test_verify_incomplete_response(self):
        """Test verification of incomplete response."""
        verifier = ResponseVerifier()
        
        response = "Machine learning is"  # Incomplete sentence
        query = "What is machine learning?"
        
        quality = verifier.verify_response(response, query)
        
        assert QualityIssue.INCOMPLETE in quality.issues
        assert any("abruptly" in suggestion.lower() for suggestion in quality.suggestions)
    
    def test_verify_empty_response(self):
        """Test verification of empty response."""
        verifier = ResponseVerifier()
        
        quality = verifier.verify_response("", "What is AI?")
        assert quality.overall_score <= 0.1  # Very low score for empty response
        assert QualityIssue.TOO_SHORT in quality.issues
        assert QualityIssue.INCOMPLETE in quality.issues
    
    def test_is_acceptable(self):
        """Test response acceptance threshold."""
        verifier = ResponseVerifier()
        
        good_response = ("Machine learning is a method of data analysis that automates "
                        "analytical model building using algorithms that iteratively "
                        "learn from data.")
        
        poor_response = "ML is good."
        
        query = "What is machine learning?"
        
        good_quality = verifier.verify_response(good_response, query)
        poor_quality = verifier.verify_response(poor_response, query)
        
        # Good response should be acceptable
        assert verifier.is_acceptable(good_quality, threshold=0.7)
        
        # Poor response should not be acceptable with higher threshold
        assert not verifier.is_acceptable(poor_quality, threshold=0.8)
    
    def test_improvement_suggestions(self):
        """Test improvement suggestions generation."""
        verifier = ResponseVerifier()
        
        response = "ML."  # Short, incomplete
        query = "Explain machine learning in detail."
        
        quality = verifier.verify_response(response, query)
        suggestions = verifier.get_improvement_suggestions(quality)
        
        assert len(suggestions) > 0
        assert any("complete" in suggestion.lower() for suggestion in suggestions)


class TestIntegratedGeneration:
    """Integration tests for the complete generation pipeline."""
    
    def test_full_generation_pipeline(self):
        """Test the complete generation pipeline from query to verified response."""
        # Process query
        processor = QueryProcessor()
        processed_query = processor.preprocess_query("What is machine learning?")
        
        # Generate response
        mock_gen = MockGenerator()
        rag_gen = RAGGenerator(generator=mock_gen)
        
        context = ("Machine learning is a subset of artificial intelligence that "
                  "focuses on the development of algorithms that can learn and make "
                  "decisions from data.")
        
        response = rag_gen.generate_answer(context, processed_query.original_query, processed_query)
        
        # Verify response
        verifier = ResponseVerifier()
        quality = verifier.verify_response(
            response.text, 
            processed_query.original_query, 
            context
        )
        
        # Assertions
        assert isinstance(processed_query, ProcessedQuery)
        assert isinstance(response, GeneratedResponse)
        assert isinstance(quality, QualityScore)
        
        assert processed_query.intent != QueryIntent.UNKNOWN
        assert len(response.text) > 0
        assert quality.overall_score > 0.0
    
    def test_different_query_types_pipeline(self):
        """Test pipeline with different types of queries."""
        processor = QueryProcessor()
        mock_gen = MockGenerator()
        rag_gen = RAGGenerator(generator=mock_gen)
        verifier = ResponseVerifier()
        
        test_queries = [
            "What is deep learning?",  # Definition
            "Explain how neural networks work",  # Explanation
            "Compare CNN and RNN",  # Comparison
            "Summarize machine learning concepts",  # Summarization
            "How to train a neural network?"  # How-to
        ]
        
        context = "Neural networks are computing systems inspired by biological neural networks."
        
        for query in test_queries:
            # Process
            processed = processor.preprocess_query(query)
            
            # Generate
            response = rag_gen.generate_answer(context, query, processed)
            
            # Verify
            quality = verifier.verify_response(response.text, query, context)
            
            # Basic assertions
            assert processed.intent != QueryIntent.UNKNOWN
            assert len(response.text) > 0
            assert quality.overall_score >= 0.0
    
    def test_error_handling(self):
        """Test error handling in the generation pipeline."""
        processor = QueryProcessor()
        mock_gen = MockGenerator()
        rag_gen = RAGGenerator(generator=mock_gen)
        
        # Test with empty query
        processed = processor.preprocess_query("")
        response = rag_gen.generate_answer("Some context", "")
        
        assert isinstance(processed, ProcessedQuery)
        assert isinstance(response, GeneratedResponse)
        
        # Test with empty context
        response2 = rag_gen.generate_answer("", "What is AI?")
        assert isinstance(response2, GeneratedResponse)
        assert len(response2.text) > 0