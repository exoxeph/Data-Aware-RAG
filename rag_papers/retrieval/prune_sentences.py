"""Sentence-level relevance scoring and pruning."""

import re
from typing import List

# Keywords that indicate research paper relevance
KEYWORDS = ("auc", "f1", "accuracy", "cifar", "imagenet", "95%", "n=", "p<", "dataset")

# Token extraction pattern
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(s: str) -> List[str]:
    """Extract tokens from string, converted to lowercase."""
    return [w.lower() for w in _WORD_RE.findall(s)]


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


def sentence_prune(query: str, text: str, min_overlap: int = 1) -> str:
    """
    Keep only sentences that share at least `min_overlap` tokens with the query.
    
    Tokens are filtered to length >= 3 to avoid stopword-like matches.
    Returns a single string of kept sentences joined by spaces.
    
    Args:
        query: The user's query string
        text: The text to prune
        min_overlap: Minimum number of shared tokens required to keep a sentence
    
    Returns:
        Pruned text containing only relevant sentences
    
    Examples:
        >>> sentence_prune("neural networks", "Neural networks learn patterns. The sky is blue.")
        'Neural networks learn patterns.'
        
        >>> sentence_prune("deep learning", "AI is cool.", min_overlap=1)
        ''
    """
    if not text or not query:
        return ""
    
    # Extract query tokens (length >= 3 to avoid stopwords)
    q_tokens = [t for t in _tokens(query) if len(t) >= 3]
    qset = set(q_tokens)
    
    if not qset:
        # No meaningful query tokens, return original
        return text
    
    # Simple sentence split on sentence-ending punctuation
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    
    for sent in sents:
        # Extract tokens from sentence
        sent_tokens = [t for t in _tokens(sent) if len(t) >= 3]
        
        # Count overlapping tokens
        overlap_count = sum(1 for t in sent_tokens if t in qset)
        
        # Keep sentence if it meets minimum overlap
        if overlap_count >= min_overlap:
            kept.append(sent)
    
    return " ".join(kept).strip()


def batch_prune(query: str, texts: List[str], min_overlap: int = 1) -> List[str]:
    """
    Prune multiple texts in batch.
    
    Args:
        query: The user's query string
        texts: List of texts to prune
        min_overlap: Minimum number of shared tokens required
    
    Returns:
        List of pruned texts
    """
    return [sentence_prune(query, text, min_overlap) for text in texts]