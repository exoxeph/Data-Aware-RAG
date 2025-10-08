"""
Session state management for RAG Cockpit.

Provides typed state container and helpers for managing app state.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path
import streamlit as st

from rag_papers.retrieval.router_dag import Stage4Config
from rag_papers.retrieval.ensemble_retriever import EnsembleRetriever


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
        "runs",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
