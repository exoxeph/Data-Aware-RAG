"""Unit test for sentence pruning and scoring."""

from rag_papers.retrieval.prune_sentences import score_sentence


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