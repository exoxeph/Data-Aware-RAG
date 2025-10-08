"""
Unit tests for MemoryStore (CRUD operations and persistence).

Tests:
- add_note(): Create and persist memory notes
- get_note(): Retrieve by ID
- list_notes(): Filter by scope, tags, sort by uses/recency
- delete_note(): Remove notes
- update_usage(): Increment uses counter
"""

import pytest
import tempfile
import time
from pathlib import Path

from rag_papers.memory.schemas import MemoryNote, MemoryScope
from rag_papers.memory.store import MemoryStore


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_memory.db")
        yield db_path


@pytest.fixture
def memory_store(temp_db):
    """Create MemoryStore instance with temp database."""
    return MemoryStore(db_path=temp_db)


# ============================================================================
# Add Note Tests
# ============================================================================

def test_add_note_basic(memory_store):
    """Test basic note addition."""
    note = MemoryNote(
        text="Transfer learning uses pretrained models.",
        scope="session",
        tags=["concept:transfer_learning"],
        source="chat"
    )
    
    note_id = memory_store.add_note(note)
    
    assert note_id is not None
    assert note_id == note.id
    
    # Verify retrieval
    retrieved = memory_store.get_note(note_id)
    assert retrieved is not None
    assert retrieved.text == "Transfer learning uses pretrained models."
    assert retrieved.scope == "session"


def test_add_note_generates_id(memory_store):
    """Test that IDs are auto-generated if not provided."""
    note = MemoryNote(
        text="Neural networks are computational models.",
        scope="corpus",
        source="doc:neural_networks.txt"
    )
    
    note_id = memory_store.add_note(note)
    
    assert note_id is not None
    assert len(note_id) > 0


def test_add_multiple_notes(memory_store):
    """Test adding multiple notes."""
    notes = [
        MemoryNote(text=f"Note {i}", scope="session", source="test")
        for i in range(5)
    ]
    
    ids = [memory_store.add_note(n) for n in notes]
    
    assert len(ids) == 5
    assert len(set(ids)) == 5  # All unique


# ============================================================================
# Get Note Tests
# ============================================================================

def test_get_note_exists(memory_store):
    """Test retrieving existing note."""
    note = MemoryNote(
        text="Overfitting occurs when model memorizes training data.",
        scope="corpus",
        tags=["concept:overfitting"],
        source="chat"
    )
    
    note_id = memory_store.add_note(note)
    retrieved = memory_store.get_note(note_id)
    
    assert retrieved is not None
    assert retrieved.id == note_id
    assert retrieved.text == note.text
    assert retrieved.tags == note.tags


def test_get_note_not_exists(memory_store):
    """Test retrieving non-existent note."""
    result = memory_store.get_note("nonexistent_id")
    assert result is None


# ============================================================================
# List Notes Tests
# ============================================================================

def test_list_notes_all(memory_store):
    """Test listing all notes."""
    notes = [
        MemoryNote(text=f"Note {i}", scope="session", source="test")
        for i in range(3)
    ]
    
    for n in notes:
        memory_store.add_note(n)
    
    all_notes = memory_store.list_notes(limit=100)
    
    assert len(all_notes) >= 3


def test_list_notes_by_scope(memory_store):
    """Test filtering notes by scope."""
    # Add notes with different scopes
    session_note = MemoryNote(text="Session note", scope="session", source="test")
    corpus_note = MemoryNote(text="Corpus note", scope="corpus", source="test")
    global_note = MemoryNote(text="Global note", scope="global", source="test")
    
    memory_store.add_note(session_note)
    memory_store.add_note(corpus_note)
    memory_store.add_note(global_note)
    
    # Filter by session
    session_notes = memory_store.list_notes(scope="session", limit=100)
    assert len(session_notes) >= 1
    assert all(n.scope == "session" for n in session_notes)
    
    # Filter by corpus
    corpus_notes = memory_store.list_notes(scope="corpus", limit=100)
    assert len(corpus_notes) >= 1
    assert all(n.scope == "corpus" for n in corpus_notes)


def test_list_notes_by_tags(memory_store):
    """Test filtering notes by tags."""
    note1 = MemoryNote(
        text="Note 1",
        scope="session",
        tags=["entity:optimizer", "concept:sgd"],
        source="test"
    )
    note2 = MemoryNote(
        text="Note 2",
        scope="session",
        tags=["entity:loss", "task:training"],
        source="test"
    )
    
    memory_store.add_note(note1)
    memory_store.add_note(note2)
    
    # Filter by one tag
    optimizer_notes = memory_store.list_notes(tags=["entity:optimizer"], limit=100)
    assert len(optimizer_notes) >= 1
    assert any("entity:optimizer" in n.tags for n in optimizer_notes)


def test_list_notes_sort_by_recency(memory_store):
    """Test sorting notes by recency."""
    # Add notes with delays to ensure different timestamps
    note1 = MemoryNote(text="Old note", scope="session", source="test")
    note1_id = memory_store.add_note(note1)
    
    time.sleep(0.01)  # Small delay
    
    note2 = MemoryNote(text="New note", scope="session", source="test")
    note2_id = memory_store.add_note(note2)
    
    # List by recency (default)
    notes = memory_store.list_notes(scope="session", sort_by="recency", limit=10)
    
    # Most recent should be first
    note_ids = [n.id for n in notes]
    assert note_ids.index(note2_id) < note_ids.index(note1_id)


def test_list_notes_sort_by_uses(memory_store):
    """Test sorting notes by usage count."""
    note1 = MemoryNote(text="Popular note", scope="session", source="test")
    note1_id = memory_store.add_note(note1)
    
    note2 = MemoryNote(text="Unpopular note", scope="session", source="test")
    note2_id = memory_store.add_note(note2)
    
    # Increment uses for note1
    for _ in range(5):
        memory_store.update_usage(note1_id)
    
    # List by uses
    notes = memory_store.list_notes(scope="session", sort_by="uses", limit=10)
    
    # Most used should be first
    note_ids = [n.id for n in notes]
    assert note_ids.index(note1_id) < note_ids.index(note2_id)


def test_list_notes_limit(memory_store):
    """Test limiting result count."""
    # Add many notes
    for i in range(10):
        note = MemoryNote(text=f"Note {i}", scope="session", source="test")
        memory_store.add_note(note)
    
    # Request only 3
    notes = memory_store.list_notes(scope="session", limit=3)
    
    assert len(notes) <= 3


# ============================================================================
# Delete Note Tests
# ============================================================================

def test_delete_note_exists(memory_store):
    """Test deleting existing note."""
    note = MemoryNote(text="Temporary note", scope="session", source="test")
    note_id = memory_store.add_note(note)
    
    # Verify it exists
    assert memory_store.get_note(note_id) is not None
    
    # Delete it
    deleted = memory_store.delete_note(note_id)
    assert deleted is True
    
    # Verify it's gone
    assert memory_store.get_note(note_id) is None


def test_delete_note_not_exists(memory_store):
    """Test deleting non-existent note."""
    deleted = memory_store.delete_note("nonexistent_id")
    assert deleted is False


# ============================================================================
# Update Usage Tests
# ============================================================================

def test_update_usage(memory_store):
    """Test incrementing usage counter."""
    note = MemoryNote(text="Test note", scope="session", source="test")
    note_id = memory_store.add_note(note)
    
    # Initial uses should be 0
    retrieved = memory_store.get_note(note_id)
    assert retrieved.uses == 0
    
    # Update usage
    memory_store.update_usage(note_id)
    
    # Verify increment
    retrieved = memory_store.get_note(note_id)
    assert retrieved.uses == 1
    
    # Update again
    memory_store.update_usage(note_id)
    retrieved = memory_store.get_note(note_id)
    assert retrieved.uses == 2


def test_update_usage_nonexistent(memory_store):
    """Test updating usage of non-existent note (should not error)."""
    # Should not raise exception
    memory_store.update_usage("nonexistent_id")


# ============================================================================
# Persistence Tests
# ============================================================================

def test_persistence_across_instances(temp_db):
    """Test that notes persist across store instances."""
    # Create note in first instance
    store1 = MemoryStore(db_path=temp_db)
    note = MemoryNote(text="Persistent note", scope="corpus", source="test")
    note_id = store1.add_note(note)
    
    # Create new instance with same DB
    store2 = MemoryStore(db_path=temp_db)
    
    # Retrieve note
    retrieved = store2.get_note(note_id)
    
    assert retrieved is not None
    assert retrieved.text == "Persistent note"
    assert retrieved.scope == "corpus"


def test_metadata_preservation(memory_store):
    """Test that metadata is preserved."""
    note = MemoryNote(
        text="Note with metadata",
        scope="session",
        source="chat",
        meta={"turn_id": "turn_123", "query": "What is ML?"}
    )
    
    note_id = memory_store.add_note(note)
    retrieved = memory_store.get_note(note_id)
    
    assert retrieved.meta == {"turn_id": "turn_123", "query": "What is ML?"}
