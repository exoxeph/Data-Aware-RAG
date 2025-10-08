"""
Memory system data models.

Defines MemoryNote structure and related types.
"""

from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import time


# Type aliases
MemoryScope = Literal["session", "corpus", "global"]
MemoryWritePolicy = Literal["never", "conservative", "aggressive"]


class MemoryNote(BaseModel):
    """
    A single memory note stored for long-term recall.
    
    Memories can be scoped to:
    - session: Specific conversation session
    - corpus: Shared across corpus (e.g., learned document facts)
    - global: System-wide (rare)
    """
    
    id: str = Field(..., description="Unique identifier (UUID4)")
    scope: MemoryScope = Field("session", description="Memory scope level")
    scope_key: str = Field(..., description="Session ID, corpus ID, or 'global'")
    
    # Content
    text: str = Field(..., description="Concise memory text (1-3 sentences)")
    tags: List[str] = Field(default_factory=list, description="Tags like 'entity:optimizer', 'task:setup'")
    
    # Provenance
    source: str = Field(..., description="Origin: 'chat', 'doc:<path>', 'system'")
    created_at: float = Field(default_factory=time.time, description="Unix timestamp")
    
    # Usage tracking
    last_used_at: float = Field(default_factory=time.time, description="Last retrieval time")
    uses: int = Field(0, description="Number of times retrieved")
    
    # Retrieval scoring (transient)
    score: float = Field(0.0, description="Last retrieval score (not persisted)")
    
    # Metadata
    meta: Dict[str, Any] = Field(default_factory=dict, description="Turn IDs, doc IDs, etc.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "mem_abc123",
                "scope": "session",
                "scope_key": "session_xyz",
                "text": "User prefers Llama 3 model with temperature 0.2 for factual queries.",
                "tags": ["preference:model", "setting:temperature"],
                "source": "chat",
                "created_at": 1696723200.0,
                "last_used_at": 1696723200.0,
                "uses": 5,
                "meta": {"turn_ids": ["turn_1", "turn_2"]}
            }
        }
    
    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage (exclude transient score)."""
        data = self.model_dump()
        data.pop("score", None)  # Don't persist retrieval score
        return data
    
    @classmethod
    def from_storage_dict(cls, data: Dict[str, Any]) -> "MemoryNote":
        """Load from storage dict."""
        return cls(**data)
    
    def mark_used(self) -> None:
        """Update usage statistics."""
        self.uses += 1
        self.last_used_at = time.time()
    
    def snippet(self, max_chars: int = 100) -> str:
        """Get truncated text for display."""
        if len(self.text) <= max_chars:
            return self.text
        return self.text[:max_chars-3] + "..."


class MemoryQuery(BaseModel):
    """Query parameters for memory retrieval."""
    
    query: str = Field(..., description="Query text to match against")
    scope: Optional[MemoryScope] = Field(None, description="Filter by scope")
    scope_key: Optional[str] = Field(None, description="Filter by specific session/corpus")
    tags: List[str] = Field(default_factory=list, description="Filter by tags (OR logic)")
    top_k: int = Field(5, description="Maximum results to return")
    min_score: float = Field(0.0, description="Minimum retrieval score threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What model settings did we use?",
                "scope": "session",
                "scope_key": "session_xyz",
                "top_k": 5,
                "min_score": 0.3
            }
        }


class MemorySummaryRequest(BaseModel):
    """Request to summarize conversation turns into memory."""
    
    turns: List[Dict[str, str]] = Field(..., description="List of {role, content} dicts")
    target_chars: int = Field(800, description="Target character count for summary")
    scope: MemoryScope = Field("session", description="Memory scope")
    scope_key: str = Field(..., description="Session/corpus identifier")
    tags: List[str] = Field(default_factory=list, description="Additional tags")
    
    class Config:
        json_schema_extra = {
            "example": {
                "turns": [
                    {"role": "user", "content": "What is dropout?"},
                    {"role": "assistant", "content": "Dropout is a regularization technique..."}
                ],
                "target_chars": 800,
                "scope": "session",
                "scope_key": "session_xyz",
                "tags": ["concept:dropout"]
            }
        }


class MemoryMetadata(BaseModel):
    """Metadata about memory usage in a response."""
    
    used_ids: List[str] = Field(default_factory=list, description="Memory IDs retrieved")
    used_count: int = Field(0, description="Number of memories used")
    used_chars: int = Field(0, description="Total characters from memories")
    written: List[str] = Field(default_factory=list, description="Memory IDs written this turn")
    snippets: List[str] = Field(default_factory=list, description="Actual memory text used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "used_ids": ["mem_abc", "mem_def"],
                "used_count": 2,
                "used_chars": 240,
                "written": ["mem_ghi"],
                "snippets": [
                    "User prefers Llama 3 model.",
                    "Temperature set to 0.2 for factual queries."
                ]
            }
        }
