"""Vector store construction using ChromaDB and sentence transformers."""
from __future__ import annotations
from typing import List, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """
    Simple in-memory vector store for text embeddings.
    """
    
    def __init__(self, embeddings: np.ndarray, text_chunks: List[str]):
        self.embeddings = embeddings
        self.text_chunks = text_chunks
        
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar text chunks using cosine similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (text_chunk, similarity_score)
        """
        if len(self.embeddings) == 0:
            return []
            
        # Calculate cosine similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            self.embeddings
        )[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.text_chunks[idx], float(similarities[idx])))
                
        return results


def build_vector_store(
    text_chunks: List[str], 
    model: Optional[SentenceTransformer] = None
) -> Tuple[VectorStore, SentenceTransformer]:
    """
    Generate embeddings for the text chunks and store them in a vector store.

    Args:
        text_chunks (List[str]): List of text chunks.
        model (SentenceTransformer, optional): Pre-trained transformer model to generate embeddings.
                                             If None, will use 'all-MiniLM-L6-v2'.

    Returns:
        Tuple[VectorStore, SentenceTransformer]: Vector store and the model used.
    """
    if not text_chunks:
        raise ValueError("text_chunks cannot be empty")
    
    # Initialize model if not provided
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in text_chunks if chunk and chunk.strip()]
    if not valid_chunks:
        raise ValueError("No valid text chunks provided")
    
    # Generate embeddings
    embeddings = model.encode(valid_chunks, convert_to_numpy=True)
    
    # Create vector store
    vector_store = VectorStore(embeddings, valid_chunks)
    
    return vector_store, model


def search_vector_store(
    query: str, 
    vector_store: VectorStore, 
    model: SentenceTransformer, 
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Search the vector store for semantically similar documents.
    
    Args:
        query: Search query
        vector_store: The vector store to search
        model: SentenceTransformer model for encoding the query
        top_k: Number of top results to return
        
    Returns:
        List of tuples (text_chunk, similarity_score)
    """
    if not query.strip():
        return []
    
    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    
    # Search the vector store
    return vector_store.search(query_embedding, top_k)
