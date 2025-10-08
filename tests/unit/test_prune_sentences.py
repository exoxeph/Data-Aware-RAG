"""Unit test for sentence pruning and scoring."""

import pytest
from rag_papers.retrieval.prune_sentences import score_sentence, sentence_prune, batch_prune, _tokens


def test_score_sentence_high_relevance():
    """Test that sentences with relevant keywords score higher."""
    query = "What is the AUC score on CIFAR-10?"
    
    relevant_sentence = "The model achieved an AUC of 0.94 on CIFAR-10 dataset with 95% confidence."
    generic_sentence = "This paper presents a new approach to image classification."
    
    relevant_score = score_sentence(query, relevant_sentence)
    generic_score = score_sentence(query, generic_sentence)
    
    assert relevant_score > generic_score
    assert relevant_score > 0.5  # Should be reasonably high


def test_score_sentence_keyword_detection():
    """Test that keyword detection works correctly."""
    query = "accuracy results"
    sentence_with_keywords = "The accuracy on ImageNet was 85% with F1 score of 0.78."
    sentence_without_keywords = "The authors thank their colleagues for helpful discussions."
    
    score_with = score_sentence(query, sentence_with_keywords)
    score_without = score_sentence(query, sentence_without_keywords)
    
    assert score_with > score_without


def test_score_sentence_token_overlap():
    """Test that token overlap contributes to scoring."""
    query = "neural network performance"
    
    high_overlap = "Neural network performance was evaluated on multiple datasets."
    low_overlap = "The statistical analysis revealed significant differences."
    
    high_score = score_sentence(query, high_overlap)
    low_score = score_sentence(query, low_overlap)
    
    assert high_score > low_score


def test_score_sentence_case_insensitive():
    """Test that scoring is case insensitive."""
    query = "AUC CIFAR"
    
    lower_sentence = "auc results on cifar dataset"
    upper_sentence = "AUC RESULTS ON CIFAR DATASET"
    mixed_sentence = "Auc Results On Cifar Dataset"
    
    lower_score = score_sentence(query, lower_sentence)
    upper_score = score_sentence(query, upper_sentence)
    mixed_score = score_sentence(query, mixed_sentence)
    
    # All should have similar scores
    assert abs(lower_score - upper_score) < 0.1
    assert abs(lower_score - mixed_score) < 0.1


def test_score_sentence_empty_query():
    """Test handling of empty query."""
    query = ""
    sentence = "Some sentence content here."
    
    score = score_sentence(query, sentence)
    assert score >= 0.0
    assert score <= 1.0


def test_score_sentence_empty_sentence():
    """Test handling of empty sentence."""
    query = "test query"
    sentence = ""
    
    score = score_sentence(query, sentence)
    assert score >= 0.0
    assert score <= 1.0


# ============================================================================
# Tests for sentence_prune and batch_prune
# ============================================================================

class TestTokenExtraction:
    """Test token extraction helper."""
    
    def test_basic_tokens(self):
        """Test basic tokenization."""
        text = "Machine learning is awesome!"
        tokens = _tokens(text)
        assert "machine" in tokens
        assert "learning" in tokens
        assert "awesome" in tokens
    
    def test_lowercasing(self):
        """Test that tokens are lowercased."""
        tokens = _tokens("NEURAL NETWORKS")
        assert "neural" in tokens
        assert "networks" in tokens
        assert "NEURAL" not in tokens
    
    def test_stopwords_and_short_tokens(self):
        """Test that tokens are lowercased but not filtered in _tokens."""
        tokens = _tokens("The cat in the hat")
        # _tokens just splits and lowercases - filtering happens elsewhere
        assert "cat" in tokens
        assert "hat" in tokens
        assert "the" in tokens  # _tokens doesn't filter stopwords
        assert "in" in tokens


class TestSentencePrune:
    """Test sentence pruning logic."""
    
    def test_no_overlap_filters_all(self):
        """Test that sentences without overlap are filtered."""
        query = "deep learning neural networks"
        text = "This is about gardening and cooking. No relevant content."
        result = sentence_prune(query, text, min_overlap=1)
        # All sentences should be filtered
        assert result == ""
    
    def test_high_overlap_keeps_sentences(self):
        """Test that sentences with sufficient overlap are kept."""
        query = "neural networks machine learning"
        text = "Neural networks are powerful. Machine learning models learn patterns."
        result = sentence_prune(query, text, min_overlap=1)
        # Both sentences have overlap
        assert "neural" in result.lower() or "machine" in result.lower()
    
    def test_min_overlap_threshold(self):
        """Test that min_overlap parameter works correctly."""
        query = "deep learning neural networks"
        # First sentence has 1 overlap ("neural")
        # Second sentence has 2 overlaps ("deep", "learning")
        text = "Neural computation is interesting. Deep learning uses neural networks."
        
        result = sentence_prune(query, text, min_overlap=2)
        assert "deep" in result.lower() or "learning" in result.lower()
    
    def test_empty_query(self):
        """Test handling of empty query."""
        result = sentence_prune("", "Some text here.", min_overlap=1)
        assert result == ""
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = sentence_prune("query", "", min_overlap=1)
        assert result == ""


class TestBatchPrune:
    """Test batch pruning across multiple documents."""
    
    def test_empty_docs(self):
        """Test handling of empty document list."""
        result = batch_prune("query", [], min_overlap=1)
        assert result == []
    
    def test_single_document(self):
        """Test pruning single document."""
        query = "machine learning"
        docs = ["Machine learning is a field of AI. It involves algorithms."]
        result = batch_prune(query, docs, min_overlap=1)
        assert len(result) == 1
        assert "machine" in result[0].lower() or "learning" in result[0].lower()
    
    def test_multiple_documents(self):
        """Test pruning multiple documents."""
        query = "neural networks"
        docs = [
            "Neural networks are computational models.",
            "This is about gardening.",
            "Networks of neurons process information."
        ]
        result = batch_prune(query, docs, min_overlap=1)
        # Should keep first and third documents
        assert len(result) >= 1
        assert not any("gardening" in doc.lower() for doc in result)
