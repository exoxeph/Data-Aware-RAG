from __future__ import annotations


def score_text(text: str) -> float:
    """
    Score text quality using heuristics.
    
    Returns a score in [0, 1] where higher is better quality.
    """
    if not text or len(text) == 0:
        return 0.0
    
    # Estimate coverage based on typical page length
    total_chars_estimate_per_page = 1500
    coverage = min(1.0, len(text) / max(1, total_chars_estimate_per_page))
    
    # Calculate junk ratio (suspicious characters)
    junk_chars = {"�", "\x00", "\ufffd"}  # Common encoding error characters
    junk_count = sum(1 for char in text if char in junk_chars)
    junk_ratio = junk_count / len(text) if len(text) > 0 else 1.0
    
    # Short text penalty
    word_count = len(text.split())
    short_penalty = 1.0 if word_count >= 30 else word_count / 30.0
    
    # Combine factors
    base_score = coverage * (1.0 - junk_ratio) * short_penalty
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, base_score))


def is_low_quality(text: str, threshold: float = 0.25) -> bool:
    """
    Determine if text is low quality based on threshold.
    
    Args:
        text: Text to evaluate
        threshold: Quality threshold (0-1), below which text is considered low quality
    
    Returns:
        True if text quality is below threshold
    """
    return score_text(text) < threshold
