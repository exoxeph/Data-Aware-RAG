"""
Memory API endpoints for managing conversation memory notes.

CRUD operations for memory notes across session/corpus/global scopes.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from rag_papers.memory.schemas import MemoryNote, MemoryQuery, MemoryScope
from rag_papers.memory.store import MemoryStore
from rag_papers.memory.recall import query_memories


router = APIRouter(prefix="/memory", tags=["memory"])


# Default memory store instance (can be dependency injected later)
_memory_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """Get or create memory store singleton."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


class AddMemoryRequest(BaseModel):
    """Request to add a new memory note."""
    
    text: str = Field(..., description="Memory note text (1-3 sentences)", min_length=1, max_length=2000)
    scope: MemoryScope = Field("session", description="Memory scope: session, corpus, or global")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    source: str = Field("manual", description="Source identifier (e.g., 'chat', 'doc:path', 'manual')")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Transfer learning uses pretrained models on new tasks.",
                "scope": "corpus",
                "tags": ["concept:transfer_learning", "entity:model"],
                "source": "chat",
                "meta": {"turn_id": "abc123"}
            }
        }


class AddMemoryResponse(BaseModel):
    """Response after adding a memory note."""
    
    note_id: str = Field(..., description="Generated note ID")
    message: str = Field(..., description="Success message")


class SearchMemoryRequest(BaseModel):
    """Request to search memory notes."""
    
    query: str = Field(..., description="Search query for semantic recall")
    scope: Optional[MemoryScope] = Field(None, description="Filter by scope")
    session_id: Optional[str] = Field(None, description="Session ID for session-scoped search")
    tags: List[str] = Field(default_factory=list, description="Filter by tags")
    limit: int = Field(5, description="Maximum number of results", ge=1, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is transfer learning?",
                "scope": "session",
                "session_id": "sess_abc123",
                "limit": 5
            }
        }


class SearchMemoryResponse(BaseModel):
    """Response with matched memory notes."""
    
    memories: List[MemoryNote] = Field(..., description="Matched memory notes with scores")
    count: int = Field(..., description="Number of results returned")


class ListMemoryRequest(BaseModel):
    """Request to list memory notes."""
    
    scope: Optional[MemoryScope] = Field(None, description="Filter by scope")
    tags: List[str] = Field(default_factory=list, description="Filter by tags")
    limit: int = Field(100, description="Maximum number of results", ge=1, le=1000)
    sort_by: str = Field("recency", description="Sort order: 'recency' or 'uses'")


class DeleteMemoryRequest(BaseModel):
    """Request to delete a memory note."""
    
    note_id: str = Field(..., description="Memory note ID to delete")


class DeleteMemoryResponse(BaseModel):
    """Response after deleting a memory note."""
    
    deleted: bool = Field(..., description="Whether note was deleted")
    message: str = Field(..., description="Status message")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/search", response_model=SearchMemoryResponse)
async def search_memories(request: SearchMemoryRequest):
    """
    Search memory notes using semantic recall.
    
    Scoring combines token overlap, cosine similarity, recency, and usage frequency.
    Returns top-k most relevant notes sorted by score.
    
    Args:
        request: Search parameters (query, scope, tags, limit)
    
    Returns:
        SearchMemoryResponse with matched notes
    
    Example:
        POST /memory/search
        {
            "query": "What is transfer learning?",
            "scope": "session",
            "session_id": "sess_abc123",
            "limit": 5
        }
        
        Response:
        {
            "memories": [
                {
                    "id": "mem_001",
                    "text": "Transfer learning uses pretrained models...",
                    "score": 0.85,
                    "tags": ["concept:transfer_learning"],
                    ...
                }
            ],
            "count": 1
        }
    """
    store = get_memory_store()
    
    # Use semantic recall
    try:
        memories = query_memories(
            query=request.query,
            memory_store=store,
            session_id=request.session_id or "default",
            top_k=request.limit,
            encoder=None  # Will use default encoder from recall.py
        )
        
        return SearchMemoryResponse(
            memories=memories,
            count=len(memories)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search failed: {str(e)}")


@router.post("/add", response_model=AddMemoryResponse)
async def add_memory(request: AddMemoryRequest):
    """
    Add a new memory note.
    
    Creates a memory note with the given text, scope, tags, and metadata.
    Returns the generated note ID.
    
    Args:
        request: Memory note details
    
    Returns:
        AddMemoryResponse with note_id
    
    Example:
        POST /memory/add
        {
            "text": "Transfer learning uses pretrained models.",
            "scope": "corpus",
            "tags": ["concept:transfer_learning"]
        }
        
        Response:
        {
            "note_id": "mem_12345",
            "message": "Memory note added successfully"
        }
    """
    store = get_memory_store()
    
    try:
        # Create memory note
        note = MemoryNote(
            text=request.text,
            scope=request.scope,
            tags=request.tags,
            source=request.source,
            meta=request.meta
        )
        
        # Store it
        note_id = store.add_note(note)
        
        return AddMemoryResponse(
            note_id=note_id,
            message="Memory note added successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add memory: {str(e)}")


@router.get("/list", response_model=SearchMemoryResponse)
async def list_memories(
    scope: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated
    limit: int = 100,
    sort_by: str = "recency"
):
    """
    List memory notes with optional filtering.
    
    Retrieves notes filtered by scope and/or tags, sorted by recency or usage.
    
    Query Parameters:
        scope: Filter by scope (session, corpus, global)
        tags: Comma-separated tags to filter by
        limit: Maximum results (default: 100)
        sort_by: Sort order - 'recency' or 'uses' (default: recency)
    
    Returns:
        SearchMemoryResponse with notes list
    
    Example:
        GET /memory/list?scope=corpus&tags=concept:transfer_learning&limit=10
        
        Response:
        {
            "memories": [...],
            "count": 10
        }
    """
    store = get_memory_store()
    
    try:
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        
        # List notes with filters
        memories = store.list_notes(
            scope=scope,
            tags=tag_list if tag_list else None,
            limit=limit,
            sort_by=sort_by  # type: ignore
        )
        
        return SearchMemoryResponse(
            memories=memories,
            count=len(memories)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {str(e)}")


@router.delete("/delete", response_model=DeleteMemoryResponse)
async def delete_memory(request: DeleteMemoryRequest):
    """
    Delete a memory note by ID.
    
    Permanently removes the specified memory note from storage.
    
    Args:
        request: Delete request with note_id
    
    Returns:
        DeleteMemoryResponse with deletion status
    
    Example:
        DELETE /memory/delete
        {
            "note_id": "mem_12345"
        }
        
        Response:
        {
            "deleted": true,
            "message": "Memory note deleted successfully"
        }
    """
    store = get_memory_store()
    
    try:
        # Attempt to delete
        deleted = store.delete_note(request.note_id)
        
        if deleted:
            return DeleteMemoryResponse(
                deleted=True,
                message="Memory note deleted successfully"
            )
        else:
            return DeleteMemoryResponse(
                deleted=False,
                message=f"Memory note {request.note_id} not found"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")


@router.post("/purge")
async def purge_session_memories(session_id: str):
    """
    Purge all session-scoped memories for a given session.
    
    Deletes all memory notes with scope='session' and matching session_id.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Dict with purge count
    
    Example:
        POST /memory/purge?session_id=sess_abc123
        
        Response:
        {
            "purged": 12,
            "message": "Purged 12 session memories"
        }
    """
    store = get_memory_store()
    
    try:
        # List all session notes
        session_notes = store.list_notes(scope="session", limit=10000)
        
        # Filter by session_id (stored in meta)
        purged = 0
        for note in session_notes:
            if note.meta.get("session_id") == session_id:
                if store.delete_note(note.id):
                    purged += 1
        
        return {
            "purged": purged,
            "message": f"Purged {purged} session memories"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to purge memories: {str(e)}")
