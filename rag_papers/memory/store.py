"""
Memory persistence layer using SQLite KVStore.

Stores MemoryNote objects with efficient retrieval by scope/tags.
"""

import json
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid

from rag_papers.persist.sqlite_store import KVStore
from .schemas import MemoryNote, MemoryScope


class MemoryStore:
    """
    Persistent storage for memory notes.
    
    Uses SQLite KVStore with namespace 'mem' and keys like:
    - mem:<scope>:<scope_key>:<uuid>
    
    Features:
    - CRUD operations for memory notes
    - Filtering by scope, scope_key, tags
    - Usage tracking (uses count, last_used_at)
    - Efficient batching and caching
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize memory store.
        
        Args:
            db_path: Path to SQLite database (default: data/cache/cache.db)
        """
        if db_path is None:
            db_path = Path("data/cache/cache.db")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use KVStore with 'mem' table
        self.kv = KVStore(str(self.db_path), table_name="mem")
    
    def _make_key(self, note: MemoryNote) -> str:
        """Generate storage key for a memory note."""
        return f"mem:{note.scope}:{note.scope_key}:{note.id}"
    
    def _parse_key(self, key: str) -> Dict[str, str]:
        """Parse storage key into components."""
        parts = key.split(":", 3)
        if len(parts) != 4 or parts[0] != "mem":
            return {}
        
        return {
            "scope": parts[1],
            "scope_key": parts[2],
            "note_id": parts[3]
        }
    
    def add(self, note: MemoryNote) -> str:
        """
        Store a memory note.
        
        Args:
            note: MemoryNote to store
        
        Returns:
            Memory ID
        """
        key = self._make_key(note)
        value = json.dumps(note.to_storage_dict())
        
        self.kv.put(key, value)
        return note.id
    
    def get(self, note_id: str, scope: MemoryScope, scope_key: str) -> Optional[MemoryNote]:
        """
        Retrieve a memory note by ID.
        
        Args:
            note_id: Memory identifier
            scope: Memory scope
            scope_key: Scope key (session ID, corpus ID, etc.)
        
        Returns:
            MemoryNote if found, else None
        """
        key = f"mem:{scope}:{scope_key}:{note_id}"
        value = self.kv.get(key)
        
        if value is None:
            return None
        
        try:
            data = json.loads(value)
            return MemoryNote.from_storage_dict(data)
        except (json.JSONDecodeError, ValueError):
            return None
    
    def delete(self, note_id: str, scope: MemoryScope, scope_key: str) -> bool:
        """
        Delete a memory note.
        
        Args:
            note_id: Memory identifier
            scope: Memory scope
            scope_key: Scope key
        
        Returns:
            True if deleted, False if not found
        """
        key = f"mem:{scope}:{scope_key}:{note_id}"
        return self.kv.delete(key)
    
    def update_usage(self, note: MemoryNote) -> None:
        """
        Update usage statistics for a memory note.
        
        Args:
            note: MemoryNote with updated uses/last_used_at
        """
        note.mark_used()
        key = self._make_key(note)
        value = json.dumps(note.to_storage_dict())
        self.kv.put(key, value)
    
    def list_by_scope(
        self,
        scope: MemoryScope,
        scope_key: str,
        limit: int = 100,
        tags: Optional[List[str]] = None
    ) -> List[MemoryNote]:
        """
        List all memory notes for a scope.
        
        Args:
            scope: Memory scope to filter
            scope_key: Specific scope key
            limit: Maximum results
            tags: Optional tag filter (OR logic)
        
        Returns:
            List of MemoryNote objects
        """
        prefix = f"mem:{scope}:{scope_key}:"
        all_keys = self.kv.list_keys(prefix=prefix, limit=limit * 2)  # Over-fetch for filtering
        
        notes = []
        for key in all_keys:
            value = self.kv.get(key)
            if value is None:
                continue
            
            try:
                data = json.loads(value)
                note = MemoryNote.from_storage_dict(data)
                
                # Apply tag filter if specified
                if tags:
                    if not any(tag in note.tags for tag in tags):
                        continue
                
                notes.append(note)
                
                if len(notes) >= limit:
                    break
                    
            except (json.JSONDecodeError, ValueError):
                continue
        
        # Sort by recency and usage
        notes.sort(key=lambda n: (n.last_used_at, n.uses), reverse=True)
        
        return notes[:limit]
    
    def list_all(self, limit: int = 1000) -> List[MemoryNote]:
        """
        List all memory notes across all scopes.
        
        Args:
            limit: Maximum results
        
        Returns:
            List of MemoryNote objects
        """
        all_keys = self.kv.list_keys(prefix="mem:", limit=limit)
        
        notes = []
        for key in all_keys:
            value = self.kv.get(key)
            if value is None:
                continue
            
            try:
                data = json.loads(value)
                note = MemoryNote.from_storage_dict(data)
                notes.append(note)
            except (json.JSONDecodeError, ValueError):
                continue
        
        return notes
    
    def clear_scope(self, scope: MemoryScope, scope_key: str) -> int:
        """
        Delete all memories for a specific scope.
        
        Args:
            scope: Memory scope
            scope_key: Scope key
        
        Returns:
            Number of memories deleted
        """
        prefix = f"mem:{scope}:{scope_key}:"
        keys = self.kv.list_keys(prefix=prefix, limit=10000)
        
        count = 0
        for key in keys:
            if self.kv.delete(key):
                count += 1
        
        return count
    
    def count(self, scope: Optional[MemoryScope] = None, scope_key: Optional[str] = None) -> int:
        """
        Count memory notes.
        
        Args:
            scope: Optional scope filter
            scope_key: Optional scope key filter
        
        Returns:
            Count of matching memories
        """
        if scope and scope_key:
            prefix = f"mem:{scope}:{scope_key}:"
        elif scope:
            prefix = f"mem:{scope}:"
        else:
            prefix = "mem:"
        
        keys = self.kv.list_keys(prefix=prefix, limit=100000)
        return len(keys)
    
    def create_note(
        self,
        text: str,
        scope: MemoryScope,
        scope_key: str,
        source: str = "chat",
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> MemoryNote:
        """
        Create and store a new memory note.
        
        Args:
            text: Memory text content
            scope: Memory scope level
            scope_key: Session/corpus identifier
            source: Origin of memory
            tags: Optional tags
            meta: Optional metadata
        
        Returns:
            Created MemoryNote
        """
        note = MemoryNote(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            scope=scope,
            scope_key=scope_key,
            text=text,
            tags=tags or [],
            source=source,
            created_at=time.time(),
            last_used_at=time.time(),
            uses=0,
            meta=meta or {}
        )
        
        self.add(note)
        return note
