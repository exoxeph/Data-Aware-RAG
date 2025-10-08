"""Query processing and prompt engineering for RAG pipeline."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import re
import string
from dataclasses import dataclass
from enum import Enum


class QueryIntent(Enum):
    """Enum for different types of query intents."""
    EXPLANATION = "explanation"
    INFORMATION_RETRIEVAL = "information_retrieval"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    HOW_TO = "how_to"
    UNKNOWN = "unknown"


@dataclass
class ProcessedQuery:
    """Container for processed query information."""
    original_query: str
    normalized_query: str
    intent: QueryIntent
    entities: List[str]
    keywords: List[str]
    confidence: float = 0.0


class QueryProcessor:
    """Handles query understanding and preprocessing."""
    
    def __init__(self):
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.EXPLANATION: [
                r"\b(explain|how|why|what is|what are|describe)\b",
                r"\b(tell me about|help me understand)\b"
            ],
            QueryIntent.DEFINITION: [
                r"\b(define|definition of|meaning of|what does .* mean)\b",
                r"\b(what is|what are) .* \?"
            ],
            QueryIntent.COMPARISON: [
                r"\b(compare|difference|versus|vs|better|worse)\b",
                r"\b(which is|what's the difference)\b"
            ],
            QueryIntent.SUMMARIZATION: [
                r"\b(summarize|summary|overview|brief)\b",
                r"\b(key points|main ideas|highlights)\b"
            ],
            QueryIntent.HOW_TO: [
                r"\b(how to|how do|how can|steps to)\b",
                r"\b(guide|tutorial|instructions)\b"
            ]
        }
        
        # Stop words for entity extraction
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'how', 'why', 'when', 'where',
            'who', 'which', 'can', 'could', 'should', 'would', 'do', 'does',
            'did', 'have', 'had', 'this', 'that', 'these', 'those'
        }
    
    def preprocess_query(self, query: str) -> ProcessedQuery:
        """
        Preprocesses the input query by normalizing text, detecting intent, and extracting entities.
        
        Args:
            query: The raw user query
            
        Returns:
            ProcessedQuery object with all processed information
        """
        if not query or not query.strip():
            return ProcessedQuery(
                original_query=query,
                normalized_query="",
                intent=QueryIntent.UNKNOWN,
                entities=[],
                keywords=[]
            )
        
        # Normalize the query
        normalized_query = self._normalize_text(query)
        
        # Detect intent
        intent, confidence = self._detect_intent(normalized_query)
        
        # Extract entities and keywords
        entities = self._extract_entities(normalized_query)
        keywords = self._extract_keywords(normalized_query)
        
        return ProcessedQuery(
            original_query=query,
            normalized_query=normalized_query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            confidence=confidence
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by cleaning and standardizing format."""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up punctuation (keep some for intent detection)
        text = re.sub(r'[^\w\s\?\!\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Detect the user's intent from the query."""
        query_lower = query.lower()
        
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return QueryIntent.INFORMATION_RETRIEVAL, 0.5  # Default intent
        
        # Get the intent with highest score
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        confidence = min(0.9, 0.5 + (best_intent[1] * 0.1))  # Simple confidence scoring
        
        return best_intent[0], confidence
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from the query."""
        # Simple entity extraction based on capitalization and technical terms
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            # Skip common words and single characters
            if (len(word) > 2 and 
                word.lower() not in self.stop_words and
                (word[0].isupper() or self._is_technical_term(word))):
                entities.append(word.lower())
        
        # Look for multi-word technical terms
        technical_phrases = self._extract_technical_phrases(query)
        entities.extend(technical_phrases)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Remove punctuation and split
        clean_query = re.sub(r'[^\w\s]', '', query)
        words = clean_query.split()
        
        keywords = []
        for word in words:
            if (len(word) > 3 and 
                word.lower() not in self.stop_words):
                keywords.append(word.lower())
        
        return keywords
    
    def _is_technical_term(self, word: str) -> bool:
        """Check if a word is likely a technical term."""
        technical_indicators = [
            'learning', 'neural', 'algorithm', 'model', 'data', 'network',
            'artificial', 'intelligence', 'machine', 'deep', 'computer',
            'vision', 'language', 'processing', 'classification', 'regression',
            'clustering', 'optimization', 'training', 'prediction'
        ]
        
        return word.lower() in technical_indicators
    
    def _extract_technical_phrases(self, query: str) -> List[str]:
        """Extract common technical phrases."""
        technical_phrases = [
            'machine learning', 'deep learning', 'artificial intelligence',
            'computer vision', 'natural language processing', 'neural network',
            'data science', 'reinforcement learning', 'supervised learning',
            'unsupervised learning', 'transfer learning', 'feature extraction'
        ]
        
        found_phrases = []
        query_lower = query.lower()
        
        for phrase in technical_phrases:
            if phrase in query_lower:
                found_phrases.append(phrase)
        
        return found_phrases


def preprocess_query(query: str) -> Dict[str, any]:
    """
    Convenience function for backward compatibility.
    
    Args:
        query: The raw user query
        
    Returns:
        Dictionary containing processed query information
    """
    processor = QueryProcessor()
    processed = processor.preprocess_query(query)
    
    return {
        "normalized_query": processed.normalized_query,
        "intent": processed.intent.value,
        "entities": processed.entities,
        "keywords": processed.keywords,
        "confidence": processed.confidence
    }


def extract_entities(query: str) -> List[str]:
    """Extract key entities from the query."""
    processor = QueryProcessor()
    processed = processor.preprocess_query(query)
    return processed.entities


def detect_intent(query: str) -> str:
    """Detect the user's intent from the query."""
    processor = QueryProcessor()
    processed = processor.preprocess_query(query)
    return processed.intent.value
