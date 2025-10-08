"""Sentence-level relevance scoring and pruning."""

import re

# Keywords that indicate research paper relevance
KEYWORDS = ("auc", "f1", "accuracy", "cifar", "imagenet", "95%", "n=", "p<", "dataset")


def score_sentence(q: str, s: str) -> float:
    """Score sentence relevance to query.
    
    Args:
        q: Query string
        s: Sentence to score
        
    Returns:
        Relevance score between 0 and 1
    """
    ql, sl = q.lower(), s.lower()
    
    # Count keyword matches
    kw = sum(1 for k in KEYWORDS if k in sl)
    
    # Calculate token overlap
    q_tokens = set(re.findall(r"\w+", ql))
    s_tokens = set(re.findall(r"\w+", sl))
    overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens)) if q_tokens else 0
    
    # Weighted combination
    return 0.55 * overlap + 0.45 * (kw > 0)