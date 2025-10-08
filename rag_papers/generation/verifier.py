"""Response verification and quality assessment for generated answers."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class QualityIssue(Enum):
    """Types of quality issues that can be detected."""
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    REPETITIVE = "repetitive"
    OFF_TOPIC = "off_topic"
    INCOHERENT = "incoherent"
    INCOMPLETE = "incomplete"
    FACTUAL_INCONSISTENCY = "factual_inconsistency"


@dataclass
class QualityScore:
    """Quality assessment score with details."""
    overall_score: float  # 0.0 to 1.0
    relevance_score: float
    coherence_score: float
    completeness_score: float
    issues: List[QualityIssue]
    suggestions: List[str]


class ResponseVerifier:
    """Verifies and assesses the quality of generated responses."""
    
    def __init__(
        self, 
        min_length: int = 10,
        max_length: int = 1000,
        max_repetition_ratio: float = 0.3
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_repetition_ratio = max_repetition_ratio
    
    def verify_response(
        self, 
        response: str, 
        query: str, 
        context: str = ""
    ) -> QualityScore:
        """
        Verify the quality of a generated response.
        
        Args:
            response: The generated response text
            query: The original query
            context: The context used for generation
            
        Returns:
            QualityScore object with assessment details
        """
        issues = []
        suggestions = []
        
        # Check length
        length_score, length_issues, length_suggestions = self._check_length(response)
        issues.extend(length_issues)
        suggestions.extend(length_suggestions)
        
        # Check relevance
        relevance_score, relevance_issues, relevance_suggestions = self._check_relevance(
            response, query
        )
        issues.extend(relevance_issues)
        suggestions.extend(relevance_suggestions)
        
        # Check coherence
        coherence_score, coherence_issues, coherence_suggestions = self._check_coherence(response)
        issues.extend(coherence_issues)
        suggestions.extend(coherence_suggestions)
        
        # Check completeness
        completeness_score, completeness_issues, completeness_suggestions = self._check_completeness(
            response, query
        )
        issues.extend(completeness_issues)
        suggestions.extend(completeness_suggestions)
        
        # Check for repetition
        repetition_score, repetition_issues, repetition_suggestions = self._check_repetition(response)
        issues.extend(repetition_issues)
        suggestions.extend(repetition_suggestions)
        
        # Calculate overall score
        overall_score = (
            length_score * 0.2 +
            relevance_score * 0.3 +
            coherence_score * 0.25 +
            completeness_score * 0.15 +
            repetition_score * 0.1
        )
        
        return QualityScore(
            overall_score=overall_score,
            relevance_score=relevance_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _check_length(self, response: str) -> Tuple[float, List[QualityIssue], List[str]]:
        """Check if response length is appropriate."""
        length = len(response.strip())
        issues = []
        suggestions = []
        
        if length < self.min_length:
            issues.append(QualityIssue.TOO_SHORT)
            suggestions.append(f"Response is too short ({length} chars). Consider expanding the answer.")
            score = length / self.min_length
        elif length > self.max_length:
            issues.append(QualityIssue.TOO_LONG)
            suggestions.append(f"Response is too long ({length} chars). Consider being more concise.")
            score = max(0.5, 1.0 - (length - self.max_length) / self.max_length)
        else:
            score = 1.0
        
        return score, issues, suggestions
    
    def _check_relevance(self, response: str, query: str) -> Tuple[float, List[QualityIssue], List[str]]:
        """Check if response is relevant to the query."""
        if not response.strip() or not query.strip():
            return 0.0, [QualityIssue.OFF_TOPIC], ["Response or query is empty"]
        
        # Extract keywords from query
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words -= stop_words
        
        if not query_words:
            return 0.7, [], []  # Neutral score for queries with only stop words
        
        # Calculate overlap
        overlap = len(query_words & response_words) / len(query_words)
        
        issues = []
        suggestions = []
        
        if overlap < 0.2:
            issues.append(QualityIssue.OFF_TOPIC)
            suggestions.append("Response seems off-topic. Ensure it addresses the query directly.")
        
        return overlap, issues, suggestions
    
    def _check_coherence(self, response: str) -> Tuple[float, List[QualityIssue], List[str]]:
        """Check if response is coherent and well-structured."""
        if not response.strip():
            return 0.0, [QualityIssue.INCOHERENT], ["Response is empty"]
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        issues = []
        suggestions = []
        score = 1.0
        
        # Check for very short sentences (might indicate incoherence)
        short_sentences = [s for s in sentences if len(s.split()) < 3]
        if len(short_sentences) > len(sentences) * 0.5:
            issues.append(QualityIssue.INCOHERENT)
            suggestions.append("Many sentences are very short. Consider combining related ideas.")
            score -= 0.3
        
        # Check for proper sentence structure
        incomplete_sentences = 0
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 0:
                # Very basic check: sentence should have some structure
                if len(words) == 1 or not any(word.isalpha() for word in words):
                    incomplete_sentences += 1
        
        if incomplete_sentences > len(sentences) * 0.3:
            issues.append(QualityIssue.INCOHERENT)
            suggestions.append("Some sentences appear incomplete or malformed.")
            score -= 0.2
        
        return max(0.0, score), issues, suggestions
    
    def _check_completeness(self, response: str, query: str) -> Tuple[float, List[QualityIssue], List[str]]:
        """Check if response adequately addresses the query."""
        if not response.strip():
            return 0.0, [QualityIssue.INCOMPLETE], ["Response is empty"]
        
        # Simple heuristics for completeness
        issues = []
        suggestions = []
        score = 0.8  # Base score
        
        # Check if response ends abruptly
        if not response.rstrip().endswith(('.', '!', '?')):
            issues.append(QualityIssue.INCOMPLETE)
            suggestions.append("Response appears to end abruptly.")
            score -= 0.2
        
        # For explanation queries, check if response provides reasoning
        if any(word in query.lower() for word in ['why', 'how', 'explain']):
            reasoning_indicators = ['because', 'due to', 'since', 'as a result', 'therefore', 'thus']
            if not any(indicator in response.lower() for indicator in reasoning_indicators):
                suggestions.append("For explanation queries, consider providing reasoning or causation.")
                score -= 0.1
        
        return max(0.0, score), issues, suggestions
    
    def _check_repetition(self, response: str) -> Tuple[float, List[QualityIssue], List[str]]:
        """Check for excessive repetition in the response."""
        if not response.strip():
            return 1.0, [], []
        
        words = response.lower().split()
        if len(words) < 5:
            return 1.0, [], []  # Too short to meaningfully check repetition
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only check longer words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate repetition ratio
        total_significant_words = sum(word_counts.values())
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        
        if total_significant_words == 0:
            return 1.0, [], []
        
        repetition_ratio = repeated_words / total_significant_words
        
        issues = []
        suggestions = []
        
        if repetition_ratio > self.max_repetition_ratio:
            issues.append(QualityIssue.REPETITIVE)
            suggestions.append(f"Response is repetitive ({repetition_ratio:.1%} repetition). Use more varied vocabulary.")
            score = max(0.2, 1.0 - repetition_ratio)
        else:
            score = 1.0
        
        return score, issues, suggestions
    
    def is_acceptable(self, quality_score: QualityScore, threshold: float = 0.6) -> bool:
        """Check if the response quality meets the acceptance threshold."""
        return quality_score.overall_score >= threshold
    
    def get_improvement_suggestions(self, quality_score: QualityScore) -> List[str]:
        """Get prioritized suggestions for improving the response."""
        suggestions = quality_score.suggestions.copy()
        
        # Add general suggestions based on scores
        if quality_score.relevance_score < 0.5:
            suggestions.insert(0, "Focus more on addressing the specific question asked.")
        
        if quality_score.coherence_score < 0.5:
            suggestions.append("Improve sentence structure and logical flow.")
        
        if quality_score.completeness_score < 0.5:
            suggestions.append("Provide a more complete answer to the question.")
        
        return suggestions
