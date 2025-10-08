"""
Memory recall system with semantic scoring.

Retrieves relevant memories using token overlap and cosine similarity.
"""

from typing import List, Optional, Dict, Any, Set
import math
import time
import hashlib
import json
from pathlib import Path

from .schemas import MemoryNote, MemoryQuery, MemoryScope
from .store import MemoryStore


class MemoryRecall:
    """
    Retrieves relevant memories for a query.
    
    Scoring:
    - Token overlap (60%): Word intersection between query and memory
    - Cosine similarity (40%): Semantic similarity using embeddings
    - Usage boost: log(uses + 1)
    - Recency decay: exp(-age_days * 0.1)
    """
    
    def __init__(
        self,
        store: MemoryStore,
        embedding_cache: Optional[Any] = None
    ):
        """
        Initialize recall system.
        
        Args:
            store: MemoryStore instance
            embedding_cache: Optional embedding cache for semantic scoring
        """
        self.store = store
        self.embedding_cache = embedding_cache
        
        # Load or create embedding cache
        if embedding_cache is None:
            try:
                from rag_papers.persist.embedding_cache import get_cached_embedding, cache_embedding
                self.get_embedding = get_cached_embedding
                self.cache_embedding = cache_embedding
                self.has_embeddings = True
            except ImportError:
                self.has_embeddings = False
        else:
            self.has_embeddings = True
    
    def query_memories(
        self,
        query: str,
        scope: Optional[MemoryScope] = None,
        scope_key: Optional[str] = None,
        recent_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        max_chars: int = 1200
    ) -> List[MemoryNote]:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: Query text
            scope: Filter by scope
            scope_key: Filter by scope key
            recent_history: Recent turns for context
            top_k: Maximum results
            min_score: Minimum score threshold
            max_chars: Maximum total characters
        
        Returns:
            List of scored MemoryNote objects
        """
        # Get candidate memories
        if scope and scope_key:
            candidates = self.store.list_by_scope(scope, scope_key, limit=200)
        else:
            candidates = self.store.list_all(limit=500)
        
        if not candidates:
            return []
        
        # Build query context (include recent history)
        query_text = query.lower()
        if recent_history:
            history_text = " ".join([
                turn.get("content", "")
                for turn in recent_history[-3:]  # Last 3 turns
            ]).lower()
            query_text = f"{query_text} {history_text}"
        
        # Score all candidates
        scored = []
        for note in candidates:
            score = self._score_memory(note, query_text)
            
            if score >= min_score:
                note.score = score
                scored.append(note)
        
        # Sort by score
        scored.sort(key=lambda n: n.score, reverse=True)
        
        # Take top-k respecting character limit
        results = []
        total_chars = 0
        
        for note in scored[:top_k * 2]:  # Over-fetch to respect char limit
            note_chars = len(note.text)
            
            if total_chars + note_chars <= max_chars:
                results.append(note)
                total_chars += note_chars
                
                # Update usage stats (in-memory, caller should persist)
                note.mark_used()
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def _score_memory(self, note: MemoryNote, query: str) -> float:
        """
        Score memory relevance to query.
        
        Args:
            note: MemoryNote to score
            query: Query text (lowercase)
        
        Returns:
            Relevance score [0.0, 1.0]
        """
        memory_text = note.text.lower()
        
        # 1. Token overlap score (60% weight)
        overlap_score = self._token_overlap(query, memory_text)
        
        # 2. Semantic similarity (40% weight) - if available
        semantic_score = 0.0
        if self.has_embeddings:
            try:
                semantic_score = self._cosine_similarity(query, memory_text, note.id)
            except Exception:
                pass  # Fall back to overlap only
        
        # Combine scores
        base_score = 0.6 * overlap_score + 0.4 * semantic_score
        
        # 3. Usage boost: log(uses + 1) scaled to [0, 0.15]
        usage_boost = min(0.15, math.log(note.uses + 1) * 0.03)
        
        # 4. Recency decay: exp(-age_days * 0.1) scaled to [0, 0.1]
        age_days = (time.time() - note.last_used_at) / 86400
        recency_boost = 0.1 * math.exp(-age_days * 0.1)
        
        final_score = min(1.0, base_score + usage_boost + recency_boost)
        
        return final_score
    
    def _token_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate token overlap score.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Overlap score [0.0, 1.0]
        """
        # Tokenize (simple word split)
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Remove punctuation and split
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        
        # Filter stop words and short tokens
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but'}
        tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]
        
        return tokens
    
    def _cosine_similarity(self, text1: str, text2: str, note_id: str) -> float:
        """
        Calculate cosine similarity using embeddings.
        
        Args:
            text1: First text (query)
            text2: Second text (memory)
            note_id: Memory note ID for caching
        
        Returns:
            Cosine similarity [0.0, 1.0]
        """
        if not self.has_embeddings:
            return 0.0
        
        try:
            # Get or compute embeddings
            emb1 = self._get_or_compute_embedding(text1, f"query_{hashlib.md5(text1.encode()).hexdigest()[:8]}")
            emb2 = self._get_or_compute_embedding(text2, f"mem:{note_id}")
            
            if emb1 is None or emb2 is None:
                return 0.0
            
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = math.sqrt(sum(a * a for a in emb1))
            norm2 = math.sqrt(sum(b * b for b in emb2))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Normalize to [0, 1]
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception:
            return 0.0
    
    def _get_or_compute_embedding(self, text: str, cache_key: str) -> Optional[List[float]]:
        """
        Get cached embedding or compute new one.
        
        Args:
            text: Text to embed
            cache_key: Cache key
        
        Returns:
            Embedding vector or None
        """
        if not self.has_embeddings:
            return None
        
        try:
            # Try cache first
            cached = self.get_embedding(cache_key)
            if cached is not None:
                return cached
            
            # Compute embedding (mock for now)
            # In production, use your sentence transformer from build_vector_store
            embedding = self._mock_embedding(text)
            
            # Cache it
            self.cache_embedding(cache_key, embedding)
            
            return embedding
            
        except Exception:
            return None
    
    def _mock_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        Generate mock embedding for testing.
        
        In production, replace with:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text).tolist()
        
        Args:
            text: Input text
            dim: Embedding dimension
        
        Returns:
            Mock embedding vector
        """
        # Deterministic mock based on text hash
        import hashlib
        import struct
        
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Generate dim floats from hash
        embedding = []
        for i in range(0, min(len(hash_bytes), dim * 4), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                val = struct.unpack('f', chunk)[0]
                embedding.append(val)
        
        # Pad if needed
        while len(embedding) < dim:
            embedding.append(0.0)
        
        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding[:dim]
    
    def format_memory_context(self, memories: List[MemoryNote]) -> str:
        """
        Format memories for injection into RAG context.
        
        Args:
            memories: List of retrieved memories
        
        Returns:
            Formatted context string
        """
        if not memories:
            return ""
        
        lines = ["[MEMORY NOTES]"]
        total_chars = 0
        
        for note in memories:
            line = f"• {note.text}"
            lines.append(line)
            total_chars += len(note.text)
        
        lines.append(f"(used {len(memories)} notes, {total_chars} chars)")
        lines.append("")  # Blank line separator
        
        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def query_memories(
    query: str,
    memory_store: MemoryStore,
    session_id: str,
    top_k: int = 5,
    encoder: Optional[Any] = None
) -> List[MemoryNote]:
    """
    Standalone function for querying memories.
    
    Args:
        query: Search query
        memory_store: MemoryStore instance
        session_id: Session identifier
        top_k: Number of results to return
        encoder: Optional encoder for semantic scoring
    
    Returns:
        List of MemoryNote sorted by relevance
    """
    recall = MemoryRecall(store=memory_store)
    return recall.query_memories(
        query=query,
        session_id=session_id,
        top_k=top_k
    )
