"""Unit test for context string generation."""

from rag_papers.index.contextualize import make_context_string


def test_make_context_string():
    """Test that context string contains all required components."""
    title = "Deep Learning for Computer Vision"
    section_path = "Methods > Architecture > CNN Layers"
    neighbor_terms = ["convolution", "pooling", "relu", "dropout"]
    
    context = make_context_string(title, section_path, neighbor_terms)
    
    assert title in context
    assert section_path in context
    assert "convolution" in context
    assert "pooling" in context
    assert "relu" in context
    assert "dropout" in context
    
    # Check basic format
    assert "|" in context  # Should contain pipe separators


def test_make_context_string_with_duplicates():
    """Test that duplicate neighbor terms are handled correctly."""
    title = "Machine Learning Survey"
    section_path = "Introduction"
    neighbor_terms = ["neural", "network", "neural", "deep", "network"]
    
    context = make_context_string(title, section_path, neighbor_terms)
    
    # Should contain unique terms in order
    assert "neural" in context
    assert "network" in context
    assert "deep" in context
    
    # Count occurrences of 'neural' should be 1 (deduplicated)
    parts = context.split("|")
    terms_part = parts[2] if len(parts) >= 3 else ""
    assert terms_part.count("neural") <= 1


def test_make_context_string_empty_terms():
    """Test context string generation with empty neighbor terms."""
    title = "Test Paper"
    section_path = "Results"
    neighbor_terms = []
    
    context = make_context_string(title, section_path, neighbor_terms)
    
    assert "Test Paper" in context
    assert "Results" in context
    assert "|" in context