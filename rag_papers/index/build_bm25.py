"""BM25 index construction for keyword-based retrieval."""
from __future__ import annotations
from typing import List, Optional
import string
import re
from rank_bm25 import BM25Okapi


def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text for BM25 indexing.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        List of preprocessed tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace and split into tokens
    tokens = text.split()
    
    # Remove empty tokens and very short words
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens


def build_bm25(text_chunks: List[str]) -> BM25Okapi:
    """
    Build a BM25 index for the provided text chunks.

    Args:
        text_chunks (List[str]): List of text chunks extracted from documents.

    Returns:
        BM25Okapi: BM25 index for text retrieval.
    """
    if not text_chunks:
        raise ValueError("text_chunks cannot be empty")
    
    # Preprocess all text chunks
    tokenized_chunks = []
    for chunk in text_chunks:
        if not chunk or not chunk.strip():
            # Handle empty chunks by adding a placeholder
            tokenized_chunks.append(['empty'])
        else:
            tokens = preprocess_text(chunk)
            if not tokens:
                # If preprocessing results in no tokens, add placeholder
                tokens = ['empty']
            tokenized_chunks.append(tokens)
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_chunks)
    
    return bm25


def search_bm25(bm25: BM25Okapi, query: str, text_chunks: List[str], top_k: int = 10) -> List[tuple[str, float]]:
    """
    Search the BM25 index for relevant documents.
    
    Args:
        bm25: The BM25 index
        query: Search query
        text_chunks: Original text chunks (for returning results)
        top_k: Number of top results to return
        
    Returns:
        List of tuples (text_chunk, score) sorted by relevance
    """
    if not query.strip():
        return []
    
    # Preprocess query
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []
    
    # Get scores for all documents
    scores = bm25.get_scores(query_tokens)
    
    # Get top k results
    top_indices = scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if idx < len(text_chunks) and scores[idx] > 0:
            results.append((text_chunks[idx], float(scores[idx])))
    
    return results
