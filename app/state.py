"""
Session state management for RAG Cockpit.

Provides typed state container and helpers for managing app state.
Stage 7: Added caching, corpus versioning, and session persistence.
Stage 8: Added chat history and multi-turn conversation support.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path
import streamlit as st
import json
from datetime import datetime
import uuid

from rag_papers.retrieval.router_dag import Stage4Config
from rag_papers.retrieval.ensemble_retriever import EnsembleRetriever

# Stage 7 imports
from rag_papers.persist import KVStore

# Stage 8 imports
from rag_papers.persist.chat_history import ChatHistory, ChatMessage, create_session_history


@dataclass
class AppState:
    """Application state container."""
    
    # Corpus and indexes
    docs: list[str] = field(default_factory=list)
    bm25_index: Optional[Any] = None
    vector_store: Optional[Any] = None
    embed_model: Optional[Any] = None
    retriever: Optional[EnsembleRetriever] = None
    active_corpus: str = "data/corpus"
    index_built_at: Optional[str] = None
    
    # Generator settings
    generator_name: str = "mock"  # "ollama" | "mock"
    ollama_model: str = "llama3"
    ollama_available: bool = False
    
    # DAG configuration
    cfg: Stage4Config = field(default_factory=Stage4Config)
    
    # Last run results
    last_run: Optional[dict] = None
    
    # Session history
    run_history: list[dict] = field(default_factory=list)
    
    # Stage 7: Caching and persistence
    kv: Optional[KVStore] = None
    corpus_id: str = "empty"
    use_cache: bool = True
    last_cache_info: dict = field(default_factory=dict)
    session_file: Optional[Path] = None
    current_job_id: Optional[str] = None  # For tracking async ingestion
    
    # Stage 8: Chat history
    chat_history: Optional[ChatHistory] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    chat_model: str = "ollama:llama3"


def get_state() -> AppState:
    """
    Get or initialize application state from session.
    
    Returns:
        AppState instance
    """
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    return st.session_state.app_state


def reset_indexes():
    """Clear retriever and indexes when corpus changes."""
    state = get_state()
    state.docs = []
    state.bm25_index = None
    state.vector_store = None
    state.embed_model = None
    state.retriever = None
    state.index_built_at = None
    st.session_state.app_state = state


def update_config(**kwargs):
    """
    Update Stage4Config with new values.
    
    Args:
        **kwargs: Config parameters to update
    """
    state = get_state()
    for key, value in kwargs.items():
        if hasattr(state.cfg, key):
            setattr(state.cfg, key, value)
    st.session_state.app_state = state


def add_to_history(run_data: dict):
    """
    Add a run to session history.
    
    Args:
        run_data: Dict with query, answer, metrics, etc.
    """
    state = get_state()
    state.run_history.append(run_data)
    st.session_state.app_state = state


def check_ollama_availability(base_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama server is available.
    
    Args:
        base_url: Ollama API base URL
        
    Returns:
        True if available, False otherwise
    """
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def ensure_directories():
    """Ensure required directories exist."""
    dirs = [
        "data/corpus",
        "data/raw_pdfs",
        "data/cache",
        "data/indexes",
        "data/jobs",
        "data/uploads",
        "runs",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


# ===== Stage 7: New Helper Functions =====


def init_cache(cache_dir: Path = Path("data/cache")) -> KVStore:
    """
    Initialize KVStore for caching.
    
    Args:
        cache_dir: Directory for cache database
        
    Returns:
        KVStore instance
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "cache.db"
    return KVStore(db_path)


def get_cache_stats(kv: KVStore) -> dict:
    """
    Get cache statistics for all tables.
    
    Args:
        kv: KVStore instance
        
    Returns:
        Dict with stats per table
    """
    if not kv:
        return {}
    
    tables = ["embeddings", "retrieval", "answers", "sessions"]
    stats = {}
    
    for table in tables:
        table_stats = kv.stats(table)
        stats[table] = {
            "count": table_stats["count"],
            "size_kb": table_stats["total_bytes"] / 1024,
            "newest": table_stats.get("newest_ts", 0),
        }
    
    return stats


def export_session(run_history: list[dict], out_path: Path) -> None:
    """
    Export session history to JSONL file.
    
    Args:
        run_history: List of run dictionaries
        out_path: Output file path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for run in run_history:
            f.write(json.dumps(run) + "\n")


def load_latest_index() -> tuple[Optional[Path], Optional[str]]:
    """
    Find the most recently built index.
    
    Returns:
        Tuple of (index_dir, corpus_id) or (None, None)
    """
    indexes_dir = Path("data/indexes")
    
    if not indexes_dir.exists():
        return None, None
    
    # Find all meta.json files
    meta_files = list(indexes_dir.rglob("meta.json"))
    
    if not meta_files:
        return None, None
    
    # Sort by modification time
    meta_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Return the parent directory and corpus_id
    latest = meta_files[0]
    index_dir = latest.parent
    corpus_id = index_dir.name
    
    return index_dir, corpus_id


# ============================================================================
# Stage 8: Chat History Helpers
# ============================================================================

def init_chat_history():
    """Initialize chat history for current session."""
    state = get_state()
    
    if state.chat_history is None:
        state.chat_history = create_session_history(
            session_id=state.session_id,
            base_dir=Path("runs")
        )
    
    st.session_state.app_state = state
    return state.chat_history


def add_chat_message(role: str, content: str, metadata: dict = None) -> ChatMessage:
    """
    Add a message to chat history.
    
    Args:
        role: "user" or "assistant"
        content: Message content
        metadata: Optional metadata (sources, cache info, etc.)
    
    Returns:
        Created ChatMessage
    """
    state = get_state()
    
    if state.chat_history is None:
        init_chat_history()
    
    msg = state.chat_history.add(role, content, metadata)
    st.session_state.app_state = state
    
    return msg


def get_chat_messages(max_count: int = 50) -> list[ChatMessage]:
    """
    Get recent chat messages.
    
    Args:
        max_count: Maximum number of messages to return
    
    Returns:
        List of ChatMessage objects
    """
    state = get_state()
    
    if state.chat_history is None:
        return []
    
    return state.chat_history.get_recent_messages(count=max_count)


def clear_chat_history():
    """Clear all chat history for current session."""
    state = get_state()
    
    if state.chat_history is not None:
        state.chat_history.clear()
    
    # Reset to new session
    state.session_id = str(uuid.uuid4())[:8]
    state.chat_history = None
    
    st.session_state.app_state = state


def get_chat_context(max_messages: int = 10) -> str:
    """
    Get formatted chat context for LLM prompt.
    
    Args:
        max_messages: Maximum number of recent messages
    
    Returns:
        Formatted conversation history
    """
    state = get_state()
    
    if state.chat_history is None:
        return ""
    
    return state.chat_history.get_context(max_messages=max_messages)

