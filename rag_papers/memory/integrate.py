"""
Memory integration hooks for RAG pipeline.

Provides functions to inject memories into retrieval context and
write memories after generation.
"""

from typing import Dict, Any, List, Optional, Tuple
import hashlib
import json

from .store import MemoryStore
from .recall import MemoryRecall
from .summarizer import MemorySummarizer
from .policy import WritePolicy
from .schemas import MemoryNote, MemoryScope, MemoryMetadata


class MemoryIntegration:
    """
    Integration layer for memory system with RAG pipeline.
    
    Provides:
    - Pre-retrieval memory injection
    - Post-generation memory writing
    - Memory version tracking for cache keys
    """
    
    def __init__(
        self,
        store: MemoryStore,
        recall: MemoryRecall,
        summarizer: MemorySummarizer,
        policy: WritePolicy
    ):
        """
        Initialize memory integration.
        
        Args:
            store: Memory store
            recall: Memory recall system
            summarizer: Memory summarizer
            policy: Write policy
        """
        self.store = store
        self.recall = recall
        self.summarizer = summarizer
        self.policy = policy
    
    def inject_memories(
        self,
        query: str,
        recent_history: List[Dict[str, str]],
        scope: MemoryScope,
        scope_key: str,
        cfg: Any
    ) -> Tuple[str, MemoryMetadata]:
        """
        Inject relevant memories into context.
        
        Called before retrieval to augment query with memory context.
        
        Args:
            query: User query
            recent_history: Recent conversation turns
            scope: Memory scope
            scope_key: Scope identifier
            cfg: Configuration with memory settings
        
        Returns:
            (memory_context_string, metadata)
        """
        # Check if memory enabled
        if not getattr(cfg, "enable_memory", True):
            return "", MemoryMetadata()
        
        # Retrieve relevant memories
        top_k = getattr(cfg, "memory_top_k", 5)
        max_chars = getattr(cfg, "memory_inject_max_chars", 1200)
        
        memories = self.recall.query_memories(
            query=query,
            scope=scope,
            scope_key=scope_key,
            recent_history=recent_history,
            top_k=top_k,
            max_chars=max_chars
        )
        
        if not memories:
            return "", MemoryMetadata()
        
        # Format context
        context = self.recall.format_memory_context(memories)
        
        # Build metadata
        metadata = MemoryMetadata(
            used_ids=[m.id for m in memories],
            used_count=len(memories),
            used_chars=sum(len(m.text) for m in memories),
            snippets=[m.text for m in memories]
        )
        
        # Persist usage updates
        for memory in memories:
            try:
                self.store.update_usage(memory)
            except Exception:
                pass  # Non-critical
        
        return context, metadata
    
    def write_memories(
        self,
        turn_data: Dict[str, Any],
        history: List[Dict[str, str]],
        verify_score: float,
        scope: MemoryScope,
        scope_key: str,
        cfg: Any
    ) -> List[str]:
        """
        Write memories after successful generation.
        
        Called after verification if answer is accepted.
        
        Args:
            turn_data: Turn information (query, answer, sources, etc.)
            history: Full conversation history
            verify_score: Verification score
            scope: Memory scope
            scope_key: Scope identifier
            cfg: Configuration
        
        Returns:
            List of memory IDs written
        """
        # Check if memory enabled
        if not getattr(cfg, "enable_memory", True):
            return []
        
        # Check write policy
        write_policy = getattr(cfg, "memory_write_policy", "conservative")
        self.policy.policy = write_policy
        
        if not self.policy.should_write(turn_data, verify_score, cfg):
            return []
        
        written_ids = []
        
        # Check if summarization needed
        summarize_every = getattr(cfg, "memory_summarize_every", 4)
        
        if self.policy.should_summarize(len(history), summarize_every):
            # Summarize conversation into multiple notes
            target_chars = getattr(cfg, "memory_summary_target_chars", 800)
            
            notes = self.summarizer.summarize_with_context(
                history=history,
                scope=scope,
                scope_key=scope_key,
                target_chars=target_chars
            )
            
            for note in notes:
                try:
                    note_id = self.store.add(note)
                    written_ids.append(note_id)
                except Exception:
                    pass  # Continue with other notes
        
        else:
            # Write single memory from this turn
            note = self.summarizer.to_memory(
                turn_data=turn_data,
                history=history,
                scope=scope,
                scope_key=scope_key
            )
            
            if note:
                try:
                    note_id = self.store.add(note)
                    written_ids.append(note_id)
                except Exception:
                    pass
        
        return written_ids
    
    def compute_memory_version(
        self,
        memory_ids: List[str],
        scope: MemoryScope,
        scope_key: str
    ) -> str:
        """
        Compute memory version hash for cache keys.
        
        Args:
            memory_ids: List of memory IDs used
            scope: Memory scope
            scope_key: Scope key
        
        Returns:
            Short hash string (8 chars)
        """
        if not memory_ids:
            return "none"
        
        # Create stable hash from memory IDs
        combined = f"{scope}:{scope_key}:{'|'.join(sorted(memory_ids))}"
        hash_obj = hashlib.md5(combined.encode())
        
        return hash_obj.hexdigest()[:8]
    
    def should_enable_memory(self, cfg: Any) -> bool:
        """
        Check if memory system is enabled.
        
        Args:
            cfg: Configuration object
        
        Returns:
            True if memory enabled
        """
        return getattr(cfg, "enable_memory", True)
    
    def get_max_notes(self, cfg: Any) -> int:
        """
        Get maximum notes to maintain per scope.
        
        Args:
            cfg: Configuration object
        
        Returns:
            Maximum note count
        """
        return getattr(cfg, "memory_max_notes", 32)
    
    def prune_old_memories(
        self,
        scope: MemoryScope,
        scope_key: str,
        max_notes: int
    ) -> int:
        """
        Prune least-used old memories to stay under limit.
        
        Args:
            scope: Memory scope
            scope_key: Scope key
            max_notes: Maximum notes to keep
        
        Returns:
            Number of notes pruned
        """
        # Get all notes for scope
        notes = self.store.list_by_scope(scope, scope_key, limit=1000)
        
        if len(notes) <= max_notes:
            return 0
        
        # Sort by usage and recency (least valuable first)
        notes.sort(key=lambda n: (n.uses, n.last_used_at))
        
        # Delete excess notes
        to_delete = notes[:len(notes) - max_notes]
        deleted = 0
        
        for note in to_delete:
            try:
                if self.store.delete(note.id, scope, scope_key):
                    deleted += 1
            except Exception:
                continue
        
        return deleted


def create_memory_integration(
    db_path: Optional[str] = None,
    generator: Optional[Any] = None,
    policy_mode: str = "conservative"
) -> Optional[MemoryIntegration]:
    """
    Factory function to create memory integration.
    
    Args:
        db_path: Path to SQLite database
        generator: LLM generator for summarization
        policy_mode: Write policy mode
    
    Returns:
        MemoryIntegration instance or None if dependencies unavailable
    """
    try:
        from rag_papers.generation.generator import MockGenerator
        
        # Create store
        store = MemoryStore(db_path)
        
        # Create recall system
        recall = MemoryRecall(store)
        
        # Create policy
        policy = WritePolicy(policy_mode)
        
        # Create summarizer
        if generator is None:
            generator = MockGenerator()
        
        summarizer = MemorySummarizer(generator, policy)
        
        # Create integration
        integration = MemoryIntegration(store, recall, summarizer, policy)
        
        return integration
        
    except Exception as e:
        # Memory system optional - return None on failure
        print(f"Warning: Could not initialize memory system: {e}")
        return None
