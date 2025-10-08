"""Ensemble retriever combining BM25 and vector-based search."""
from __future__ import annotations
from typing import List, Tuple, Dict, Set
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

from ..index.build_bm25 import search_bm25
from ..index.build_vector_store import VectorStore, search_vector_store


class EnsembleRetriever:
    """
    Hybrid retriever that combines BM25 and vector-based search.
    """
    
    def __init__(
        self, 
        bm25_index: BM25Okapi, 
        vector_store: VectorStore, 
        model: SentenceTransformer,
        text_chunks: List[str]
    ):
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.model = model
        self.text_chunks = text_chunks
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        bm25_weight: float = 0.5, 
        vector_weight: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid search combining BM25 and vector-based retrieval.
        
        Args:
            query: The query string
            top_k: Number of top results to return
            bm25_weight: Weight for BM25 scores (should sum to 1.0 with vector_weight)
            vector_weight: Weight for vector similarity scores
            
        Returns:
            List of tuples (text_chunk, combined_score) sorted by relevance
        """
        return hybrid_search(
            query=query,
            bm25_index=self.bm25_index,
            vector_store=self.vector_store,
            model=self.model,
            text_chunks=self.text_chunks,
            top_k=top_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight
        )


def hybrid_search(
    query: str, 
    bm25_index: BM25Okapi, 
    vector_store: VectorStore, 
    model: SentenceTransformer,
    text_chunks: List[str],
    top_k: int = 10,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Perform a hybrid search combining BM25 and vector-based retrieval.

    Args:
        query (str): The query string provided by the user.
        bm25_index (BM25Okapi): The BM25 index for keyword-based retrieval.
        vector_store (VectorStore): The vector store containing embeddings for text chunks.
        model (SentenceTransformer): The sentence transformer model.
        text_chunks (List[str]): Original text chunks.
        top_k (int): The number of top results to retrieve.
        bm25_weight (float): Weight for BM25 scores.
        vector_weight (float): Weight for vector similarity scores.

    Returns:
        List[Tuple[str, float]]: List of tuples (text_chunk, combined_score) sorted by relevance.
    """
    if not query.strip():
        return []
    
    # Get results from both retrieval methods
    bm25_results = search_bm25(bm25_index, query, text_chunks, top_k * 2)
    vector_results = search_vector_store(query, vector_store, model, top_k * 2)
    
    # Normalize scores to [0, 1] range
    def normalize_scores(results: List[Tuple[str, float]]) -> Dict[str, float]:
        if not results:
            return {}
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return {text: 1.0 for text, _ in results}
        
        normalized = {}
        for text, score in results:
            normalized[text] = (score - min_score) / (max_score - min_score)
        
        return normalized
    
    # Normalize scores
    bm25_scores = normalize_scores(bm25_results)
    vector_scores = normalize_scores(vector_results)
    
    # Combine scores
    all_texts: Set[str] = set(bm25_scores.keys()) | set(vector_scores.keys())
    combined_scores = {}
    
    for text in all_texts:
        bm25_score = bm25_scores.get(text, 0.0)
        vector_score = vector_scores.get(text, 0.0)
        combined_score = (bm25_weight * bm25_score) + (vector_weight * vector_score)
        combined_scores[text] = combined_score
    
    # Sort by combined score and return top k
    sorted_results = sorted(
        combined_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    return [(text, score) for text, score in sorted_results if score > 0]


def simple_hybrid_search(
    query: str, 
    bm25_index: BM25Okapi, 
    vector_store: VectorStore,
    model: SentenceTransformer,
    text_chunks: List[str], 
    top_k: int = 10
) -> List[str]:
    """
    Simplified hybrid search that returns only the text chunks (for backward compatibility).
    
    Args:
        query (str): The query string provided by the user.
        bm25_index (BM25Okapi): The BM25 index for keyword-based retrieval.
        vector_store (VectorStore): The vector store containing embeddings for text chunks.
        model (SentenceTransformer): The sentence transformer model.
        text_chunks (List[str]): Original text chunks.
        top_k (int): The number of top results to retrieve.

    Returns:
        List[str]: List of top-k relevant text chunks based on hybrid search.
    """
    results = hybrid_search(query, bm25_index, vector_store, model, text_chunks, top_k)
    return [text for text, _ in results]