"""
Reusable UI components for RAG Cockpit.

Provides widgets for backend status, knobs, tables, charts, etc.
"""

import streamlit as st
import pandas as pd
from typing import Optional
from rag_papers.retrieval.router_dag import Stage4Config


def backend_status(ollama_ok: bool, generator_name: str):
    """
    Render backend status chips.
    
    Args:
        ollama_ok: Whether Ollama is available
        generator_name: Current generator ("ollama" or "mock")
    """
    if generator_name == "ollama":
        if ollama_ok:
            st.markdown(
                '<span class="metric-chip success">✓ Ollama Online</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="metric-chip error">✗ Ollama Offline</span>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<span class="metric-chip warning">⚡ Mock Mode</span>',
            unsafe_allow_html=True
        )


def knobs(cfg: Stage4Config, prefix: str = "") -> Stage4Config:
    """
    Render configuration knobs and return updated config.
    
    Args:
        cfg: Current Stage4Config
        prefix: Optional prefix for widget keys (for multiple instances)
        
    Returns:
        Updated Stage4Config
    """
    st.subheader("⚙️ Pipeline Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Retrieval**")
        bm25_weight = st.slider(
            "BM25 Weight",
            0.0, 1.0, float(cfg.bm25_weight_init),
            0.05,
            key=f"{prefix}bm25_weight",
            help="Weight for BM25 lexical search"
        )
        vector_weight = st.slider(
            "Vector Weight",
            0.0, 1.0, float(cfg.vector_weight_init),
            0.05,
            key=f"{prefix}vector_weight",
            help="Weight for dense vector search"
        )
        top_k_first = st.number_input(
            "Top-K First",
            1, 50, cfg.top_k_first,
            key=f"{prefix}top_k",
            help="Initial retrieval count"
        )
        rerank_top_k = st.number_input(
            "Rerank Top-K",
            1, 20, cfg.rerank_top_k,
            key=f"{prefix}rerank_top_k",
            help="After reranking, keep top-K"
        )
    
    with col2:
        st.markdown("**Processing**")
        prune_min_overlap = st.number_input(
            "Prune Min Overlap",
            0, 5, cfg.prune_min_overlap,
            key=f"{prefix}prune_overlap",
            help="Minimum token overlap for pruning"
        )
        prune_max_chars = st.number_input(
            "Prune Max Chars",
            500, 5000, cfg.prune_max_chars,
            100,
            key=f"{prefix}prune_max",
            help="Max chars per candidate before pruning"
        )
        context_max_chars = st.number_input(
            "Context Max Chars",
            1000, 10000, cfg.context_max_chars,
            500,
            key=f"{prefix}context_max",
            help="Final context budget"
        )
        
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Generation**")
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, float(cfg.temperature_init),
            0.05,
            key=f"{prefix}temperature",
            help="LLM sampling temperature"
        )
    
    with col4:
        st.markdown("**Verification**")
        accept_threshold = st.slider(
            "Accept Threshold",
            0.0, 1.0, float(cfg.accept_threshold),
            0.05,
            key=f"{prefix}accept_threshold",
            help="Minimum score to accept answer"
        )
    
    # Create updated config
    return Stage4Config(
        bm25_weight_init=bm25_weight,
        vector_weight_init=vector_weight,
        top_k_first=top_k_first,
        rerank_top_k=rerank_top_k,
        prune_min_overlap=prune_min_overlap,
        prune_max_chars=prune_max_chars,
        context_max_chars=context_max_chars,
        temperature_init=temperature,
        accept_threshold=accept_threshold,
        # Keep repair settings from original
        bm25_weight_on_repair=cfg.bm25_weight_on_repair,
        vector_weight_on_repair=cfg.vector_weight_on_repair,
        temperature_on_repair=cfg.temperature_on_repair,
        max_repairs=cfg.max_repairs,
    )


def retrieval_table(candidates: list, show_meta: bool = False):
    """
    Render retrieval results table.
    
    Args:
        candidates: List of candidates (text, score, meta)
        show_meta: Whether to show metadata columns
    """
    if not candidates:
        st.info("No candidates retrieved")
        return
    
    data = []
    for i, cand in enumerate(candidates, 1):
        if isinstance(cand, dict):
            text = cand.get("text", "")[:200]
            score = cand.get("score", 0.0)
            meta = cand.get("metadata", {})
        elif hasattr(cand, "text"):
            text = cand.text[:200]
            score = cand.score
            meta = getattr(cand, "metadata", {})
        else:
            text = str(cand)[:200]
            score = 0.0
            meta = {}
        
        row = {
            "Rank": i,
            "Score": f"{score:.3f}",
            "Preview": text + ("..." if len(str(text)) > 200 else ""),
        }
        
        if show_meta and meta:
            row["Source"] = meta.get("source", "N/A")
            row["Page"] = meta.get("page", "N/A")
        
        data.append(row)
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def timings_bar(timings: dict[str, float]):
    """
    Render horizontal bar chart of step timings.
    
    Args:
        timings: Dict mapping step name to milliseconds
    """
    if not timings:
        st.info("No timing data available")
        return
    
    # Create DataFrame
    data = [
        {"Step": step, "Time (ms)": f"{ms:.1f}", "Duration": ms}
        for step, ms in timings.items()
    ]
    df = pd.DataFrame(data)
    
    # Simple bar chart using st.bar_chart
    st.bar_chart(df.set_index("Step")["Duration"])
    
    # Also show table
    st.dataframe(
        df[["Step", "Time (ms)"]],
        use_container_width=True,
        hide_index=True
    )


def highlight_overlap(text: str, query: str, min_token_len: int = 4) -> str:
    """
    Highlight query tokens in text.
    
    Args:
        text: Text to highlight
        query: Query string
        min_token_len: Minimum token length to highlight
        
    Returns:
        HTML string with <mark> tags
    """
    import re
    
    # Extract tokens
    tokens = [
        t.lower() for t in re.findall(r'\w+', query)
        if len(t) >= min_token_len
    ]
    
    if not tokens:
        return text
    
    # Build pattern
    pattern = '|'.join(re.escape(t) for t in tokens)
    
    # Replace with marks (case-insensitive)
    highlighted = re.sub(
        f'({pattern})',
        r'<mark>\1</mark>',
        text,
        flags=re.IGNORECASE
    )
    
    return highlighted


def metric_card(label: str, value: str, chip_class: str = ""):
    """
    Render a metric as a styled chip.
    
    Args:
        label: Metric label
        value: Metric value
        chip_class: Additional CSS class (success, warning, error)
    """
    st.markdown(
        f'<div class="metric-chip {chip_class}">'
        f'<strong>{label}:</strong> {value}'
        f'</div>',
        unsafe_allow_html=True
    )


def step_path_display(path: list[str]):
    """
    Display execution path as connected steps.
    
    Args:
        path: List of step names
    """
    if not path:
        st.info("No execution path available")
        return
    
    # Step icons
    icons = {
        "retrieve": "🔍",
        "rerank": "📊",
        "prune": "✂️",
        "contextualize": "📝",
        "generate": "🤖",
        "verify": "✅",
        "repair": "🔧",
    }
    
    path_html = " → ".join([
        f"{icons.get(step, '•')} <code>{step}</code>"
        for step in path
    ])
    
    st.markdown(f"**Execution Path:** {path_html}", unsafe_allow_html=True)
