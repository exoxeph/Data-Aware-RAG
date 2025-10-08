"""
RAG Cockpit - Streamlit Application Entrypoint

A techno-styled interface for the RAG pipeline with:
- PDF upload and corpus management
- DAG execution with step-by-step inspection
- Live configuration tuning
- Telemetry and metrics visualization

Usage:
    poetry run streamlit run streamlit_app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.state import get_state, ensure_directories, check_ollama_availability


# Page configuration
st.set_page_config(
    page_title="RAG Cockpit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_file = Path(__file__).parent / "app" / "ui_theme.css"
if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize state
ensure_directories()
state = get_state()

# Check Ollama availability on first load
if "ollama_checked" not in st.session_state:
    state.ollama_available = check_ollama_availability()
    st.session_state.ollama_checked = True

# Sidebar
with st.sidebar:
    st.markdown("# 🤖 RAG Cockpit")
    st.markdown("---")
    
    # Corpus info
    st.markdown("### 📁 Corpus")
    st.text_input(
        "Active Directory",
        value=state.active_corpus,
        disabled=True,
        key="sidebar_corpus_display"
    )
    
    if state.retriever:
        st.success(f"✓ {len(state.docs)} docs indexed")
    else:
        st.warning("⚠️ No indexes built")
    
    st.markdown("---")
    
    # Generator backend
    st.markdown("### 🔧 Backend")
    
    backend_options = ["mock", "ollama"]
    backend_labels = {
        "mock": "🎭 Mock (Offline)",
        "ollama": "🦙 Ollama (Local)"
    }
    
    selected_backend = st.selectbox(
        "Generator",
        options=backend_options,
        index=backend_options.index(state.generator_name),
        format_func=lambda x: backend_labels[x],
        key="sidebar_backend"
    )
    
    if selected_backend != state.generator_name:
        state.generator_name = selected_backend
    
    if state.generator_name == "ollama":
        if state.ollama_available:
            st.success("✓ Ollama online")
        else:
            st.error("✗ Ollama offline")
            st.caption("Start with: `ollama serve`")
    
    st.markdown("---")
    
    # Quick links
    st.markdown("### 🔗 Links")
    
    if st.button("📖 API Docs", use_container_width=True):
        st.write("Start API: `poetry run rag-serve`")
        st.write("Then visit: http://localhost:8000/docs")
    
    if st.button("📂 Open Runs Folder", use_container_width=True):
        import subprocess
        import platform
        
        runs_dir = Path("runs").absolute()
        if platform.system() == "Windows":
            subprocess.Popen(f'explorer "{runs_dir}"')
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(runs_dir)])
        else:
            subprocess.Popen(["xdg-open", str(runs_dir)])
    
    st.markdown("---")
    st.caption("RAG Cockpit v1.0.0")

# Main content area
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🤖 RAG Cockpit</h1>
    <p style="font-size: 1.2rem; color: #00F5FF; font-family: monospace;">
        Techno-styled interface for the RAG pipeline
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Navigation guide
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h2>🏗️ Corpus</h2>
        <p>Upload PDFs, load text files, and build search indexes</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h2>🤖 Ask</h2>
        <p>Query your corpus and inspect every DAG step</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h2>📈 Metrics</h2>
        <p>View evaluation results and telemetry data</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick start guide
with st.expander("🚀 Quick Start Guide"):
    st.markdown("""
    ### Getting Started
    
    1. **Build Your Corpus** (🏗️ Corpus page)
       - Load existing text files from `data/corpus/`
       - Or upload PDFs (experimental)
       - Click "Build Indexes" to create BM25 + Vector search
    
    2. **Ask Questions** (🤖 Ask page)
       - Choose backend (Mock for testing, Ollama for real LLM)
       - Enter your query and select intent
       - Adjust configuration knobs
       - View step-by-step execution
    
    3. **Review Performance** (📈 Metrics page)
       - Load evaluation runs from `runs/`
       - View Accept@1, repair rates, latency
       - Inspect worst/best performing queries
    
    ### Backend Options
    
    - **Mock**: Offline mode, no LLM API calls, great for testing
    - **Ollama**: Local LLM inference (requires `ollama serve`)
    
    ### Tips
    
    - Corpus files should be in `data/corpus/*.txt` or `*.md`
    - Run evaluations with: `poetry run rag-eval --dataset qa_small --out runs/ --corpus-dir data/corpus`
    - Adjust BM25/Vector weights: higher BM25 (0.7+) for keyword matching
    - Lower accept threshold (0.5) to see more accepted answers with Mock generator
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.6; padding: 2rem 0;">
    <p style="font-family: monospace; font-size: 0.9rem;">
        Built with Streamlit | Powered by Sentence Transformers & Rank-BM25
    </p>
</div>
""", unsafe_allow_html=True)
