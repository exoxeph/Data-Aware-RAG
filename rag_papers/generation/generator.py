"""LLM integration and answer generation for RAG pipeline."""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Iterator
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import logging
import time
from pathlib import Path

try:
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer,
        AutoTokenizer, AutoModelForCausalLM,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. LLM generation will be mocked.")

from .prompts import ProcessedQuery, QueryIntent


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model_name: str = "gpt2"
    max_length: int = 200
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class GeneratedResponse:
    """Container for generated response with metadata."""
    text: str
    confidence: float
    model_used: str
    prompt_length: int
    response_length: int
    processing_time: float = 0.0


class BaseGenerator(ABC):
    """Abstract base class for text generators."""
    
    @abstractmethod
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GeneratedResponse:
        """Generate text based on the given prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the generator is available and ready to use."""
        pass
    
    def stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[str]:
        """
        Stream tokens/chunks as they are generated.
        
        Default implementation: generate full response and split into words.
        Override this for true streaming with LLM APIs.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
        
        Yields:
            Token strings
        """
        # Default: generate full response and simulate streaming
        response = self.generate(prompt, config)
        words = response.text.split()
        
        for word in words:
            yield word + " "
            time.sleep(0.01)  # Simulate streaming delay


class MockGenerator(BaseGenerator):
    """Mock generator for testing and when transformers is not available."""
    
    def __init__(self):
        self.mock_responses = {
            "machine learning": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            "deep learning": "Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
            "computer vision": "Computer vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world.",
            "default": "Based on the provided context, this is a generated response that addresses your question using the available information."
        }
    
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GeneratedResponse:
        """Generate a mock response based on keywords in the prompt."""
        prompt_lower = prompt.lower()
        
        # Find the best matching response
        response_text = self.mock_responses["default"]
        for keyword, response in self.mock_responses.items():
            if keyword != "default" and keyword in prompt_lower:
                response_text = response
                break
        
        return GeneratedResponse(
            text=response_text,
            confidence=0.8,
            model_used="mock_generator",
            prompt_length=len(prompt),
            response_length=len(response_text)
        )
    
    def stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[str]:
        """
        Stream mock response word-by-word to simulate real LLM streaming.
        
        Args:
            prompt: Input prompt
            config: Generation configuration (unused for mock)
        
        Yields:
            Words with trailing spaces
        """
        prompt_lower = prompt.lower()
        
        # Find best matching response (same logic as generate)
        response_text = self.mock_responses["default"]
        for keyword, response in self.mock_responses.items():
            if keyword != "default" and keyword in prompt_lower:
                response_text = response
                break
        
        # Split into words and yield with delay
        words = response_text.split()
        for i, word in enumerate(words):
            # Add space after each word except the last
            token = word if i == len(words) - 1 else word + " "
            yield token
            time.sleep(0.005)  # Fast streaming for testing
    
    def is_available(self) -> bool:
        """Mock generator is always available."""
        return True


class TransformersGenerator(BaseGenerator):
    """Generator using Hugging Face Transformers library."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for TransformersGenerator")
        
        self.config = config or GenerationConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logging.error(f"Failed to load model {self.config.model_name}: {e}")
            raise
    
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GeneratedResponse:
        """Generate text using the loaded model."""
        if not self.is_available():
            raise RuntimeError("Model is not available")
        
        gen_config = config or self.config
        
        # Tokenize the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for TransformersGenerator")
            
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=min(gen_config.max_length, len(inputs[0]) + gen_config.max_new_tokens),
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                num_return_sequences=gen_config.num_return_sequences,
                do_sample=gen_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part (remove the prompt)
        response_text = generated_text[len(prompt):].strip()
        
        return GeneratedResponse(
            text=response_text,
            confidence=0.7,  # Could be improved with actual confidence scoring
            model_used=self.config.model_name,
            prompt_length=len(prompt),
            response_length=len(response_text)
        )
    
    def is_available(self) -> bool:
        """Check if model and tokenizer are loaded."""
        return self.model is not None and self.tokenizer is not None


class RAGGenerator:
    """Main RAG generator that combines retrieval context with query processing."""
    
    def __init__(self, generator: Optional[BaseGenerator] = None, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        
        if generator:
            self.generator = generator
        else:
            # Try to use transformers, fall back to mock
            try:
                if TRANSFORMERS_AVAILABLE:
                    self.generator = TransformersGenerator(self.config)
                else:
                    self.generator = MockGenerator()
            except Exception:
                logging.warning("Failed to initialize TransformersGenerator, using MockGenerator")
                self.generator = MockGenerator()
    
    def generate_answer(
        self, 
        context: str, 
        query: str, 
        processed_query: Optional[ProcessedQuery] = None
    ) -> GeneratedResponse:
        """
        Generate an answer based on retrieved context and user query.
        
        Args:
            context: Retrieved relevant content to be used as context
            query: The original user query
            processed_query: Optional processed query information
            
        Returns:
            Generated response object
        """
        # Create prompt based on query intent
        prompt = self._create_prompt(context, query, processed_query)
        
        # Generate response
        response = self.generator.generate(prompt, self.config)
        
        # Post-process the response
        processed_text = self._postprocess_response(response.text, query)
        
        return GeneratedResponse(
            text=processed_text,
            confidence=response.confidence,
            model_used=response.model_used,
            prompt_length=response.prompt_length,
            response_length=len(processed_text),
            processing_time=response.processing_time
        )
    
    def _create_prompt(
        self, 
        context: str, 
        query: str, 
        processed_query: Optional[ProcessedQuery] = None
    ) -> str:
        """Create an appropriate prompt based on query intent."""
        if not context.strip():
            return f"Question: {query}\nAnswer:"
        
        # Determine prompt template based on intent
        if processed_query and processed_query.intent:
            intent = processed_query.intent
        else:
            intent = QueryIntent.INFORMATION_RETRIEVAL
        
        if intent == QueryIntent.EXPLANATION:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nProvide a clear and detailed explanation based on the context above:\n"
        elif intent == QueryIntent.SUMMARIZATION:
            prompt = f"Context: {context}\n\nPlease provide a concise summary addressing: {query}\n\nSummary:"
        elif intent == QueryIntent.COMPARISON:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nBased on the context, compare and contrast the relevant aspects:\n"
        elif intent == QueryIntent.DEFINITION:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nProvide a clear definition based on the context:\n"
        elif intent == QueryIntent.HOW_TO:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nBased on the context, provide step-by-step guidance:\n"
        else:  # INFORMATION_RETRIEVAL or UNKNOWN
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        return prompt
    
    def _postprocess_response(self, response: str, query: str) -> str:
        """Post-process the generated response for quality and format."""
        if not response.strip():
            return "I don't have enough information to answer your question based on the provided context."
        
        # Clean up the response
        cleaned = response.strip()
        
        # Remove any repetition of the prompt or query
        query_words = set(query.lower().split())
        lines = cleaned.split('\n')
        
        # Filter out lines that just repeat the question
        filtered_lines = []
        for line in lines:
            line_words = set(line.lower().split())
            # Skip lines that are mostly question words
            if len(line_words & query_words) < len(line_words) * 0.7:
                filtered_lines.append(line)
        
        if filtered_lines:
            cleaned = '\n'.join(filtered_lines)
        
        # Ensure proper capitalization
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        # Ensure it ends with proper punctuation
        if cleaned and cleaned[-1] not in '.!?':
            cleaned += '.'
        
        return cleaned


def generate_answer(context: str, query: str, model_name: str = "gpt2") -> str:
    """
    Convenience function for generating answers (backward compatibility).
    
    Args:
        context: Retrieved relevant content
        query: The original user query
        model_name: Name of the model to use
        
    Returns:
        Generated answer string
    """
    config = GenerationConfig(model_name=model_name)
    generator = RAGGenerator(config=config)
    response = generator.generate_answer(context, query)
    return response.text
