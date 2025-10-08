from __future__ import annotations
from typing import List
import re


def normalize(text: str) -> str:
    """
    Normalize text by collapsing spaces/newlines and stripping.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Collapse multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()


def chunk_text(text: str, max_words: int = 512) -> List[str]:
    """
    Split text into chunks based on word count.
    
    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Normalize the text first
    normalized_text = normalize(text)
    
    # Split into words
    words = normalized_text.split()
    
    if len(words) <= max_words:
        return [normalized_text]
    
    # Create chunks
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        
        # If we've reached the max words, create a chunk
        if len(current_chunk) >= max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    # Add any remaining words as a final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
