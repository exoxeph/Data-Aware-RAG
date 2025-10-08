"""
Corpus Management Page - Upload PDFs, ingest, and build indexes.
Stage 7: Added async ingestion with progress tracking and cache stats.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import glob
import time
import httpx

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.state import get_state, reset_indexes, ensure_directories, init_cache, get_cache_stats, load_latest_index
from app.components import metric_card

# Import RAG components
from rag_papers.index.build_bm25 import build_bm25
from rag_papers.index.build_vector_store import build_vector_store
from rag_papers.retrieval.ensemble_retriever import EnsembleRetriever
from rag_papers.persist import load_indexes, corpus_id_from_dir


def load_corpus_from_dir(corpus_dir: str) -> list[str]:
    """Load all text files from corpus directory."""
    docs = []
    patterns = [
        f"{corpus_dir}/*.txt",
        f"{corpus_dir}/*.md",
        f"{corpus_dir}/**/*.txt",
        f"{corpus_dir}/**/*.md",
    ]
    
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if content:
                        docs.append(content)
            except Exception as e:
                st.warning(f"Could not read {Path(path).name}: {e}")
    
    return docs


# ===== Stage 7: Async Ingestion Helpers =====


def start_async_ingestion(corpus_dir: str, api_url: str = "http://localhost:8000") -> str:
    """
    Start async ingestion job via API.
    
    Returns:
        Job ID or None if failed
    """
    try:
        response = httpx.post(
            f"{api_url}/ingest/start",
            json={
                "corpus_dir": corpus_dir,
                "model_name": "all-MiniLM-L6-v2",
                "use_cache": True,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()["job_id"]
    except Exception as e:
        st.error(f"Failed to start ingestion: {e}")
        return None


def check_job_status(job_id: str, api_url: str = "http://localhost:8000") -> dict:
    """
    Check ingestion job status via API.
    
    Returns:
        Status dict with state, progress, message
    """
    try:
        response = httpx.get(
            f"{api_url}/ingest/status/{job_id}",
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to check status: {e}")
        return {"state": "failed", "progress": 0.0, "message": str(e)}


def main():
    st.title("🏗️ Corpus Management")
    st.markdown("Upload PDFs, ingest documents, and build search indexes.")
    
    ensure_directories()
    state = get_state()
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Corpus Source")
        
        # Corpus directory selector
        corpus_dir = st.text_input(
            "Corpus Directory",
            value=state.active_corpus,
            help="Directory containing .txt or .md files"
        )
        
        if corpus_dir != state.active_corpus:
            state.active_corpus = corpus_dir
            reset_indexes()
            st.rerun()
        
        # File uploader
        st.markdown("**Upload PDFs**")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            upload_dir = Path("data/raw_pdfs") / Path(corpus_dir).name
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            saved_count = 0
            for uploaded_file in uploaded_files:
                try:
                    file_path = upload_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1
                except Exception as e:
                    st.error(f"Failed to save {uploaded_file.name}: {e}")
            
            if saved_count > 0:
                st.success(f"✓ Saved {saved_count} PDF(s) to {upload_dir}")
        
        # Ingest button
        st.markdown("---")
        if st.button("🔄 Ingest PDFs", use_container_width=True):
            pdf_dir = Path("data/raw_pdfs") / Path(corpus_dir).name
            if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
                st.warning("No PDFs found to ingest.")
            else:
                with st.spinner("Ingesting PDFs..."):
                    try:
                        # Simple ingestion: use marker or read raw text
                        # For now, show placeholder (full integration would use IngestionService)
                        st.info(
                            f"Found {len(list(pdf_dir.glob('*.pdf')))} PDFs. "
                            "Full PDF ingestion requires marker integration. "
                            "Use text files in corpus directory for now."
                        )
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
        
        # Manual text input (advanced)
        with st.expander("⚙️ Advanced: Direct Text Input"):
            manual_text = st.text_area(
                "Paste text corpus",
                height=150,
                help="Quick way to add documents without files"
            )
            if st.button("Add to Corpus"):
                if manual_text.strip():
                    corpus_file = Path(corpus_dir) / "manual_input.txt"
                    corpus_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(corpus_file, "a", encoding="utf-8") as f:
                        f.write("\n\n" + manual_text.strip())
                    st.success("✓ Added to corpus")
                    reset_indexes()
                    st.rerun()
    
    with col2:
        st.subheader("🔨 Index Management")
        
        # Load corpus button
        if st.button("📖 Load Corpus", use_container_width=True):
            with st.spinner(f"Loading documents from {corpus_dir}..."):
                try:
                    docs = load_corpus_from_dir(corpus_dir)
                    if not docs:
                        st.error(f"No .txt or .md files found in {corpus_dir}")
                    else:
                        state.docs = docs
                        state.retriever = None  # Reset retriever
                        st.success(f"✓ Loaded {len(docs)} documents")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load corpus: {e}")
        
        # Build indexes button
        if st.button("⚡ Build Indexes", use_container_width=True, type="primary"):
            if not state.docs:
                st.error("Load corpus first!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Build BM25
                    status_text.text("Building BM25 index...")
                    progress_bar.progress(0.2)
                    bm25_index = build_bm25(state.docs)
                    state.bm25_index = bm25_index
                    
                    # Build vector store
                    status_text.text("Building vector store (this may take a minute)...")
                    progress_bar.progress(0.4)
                    vector_store, embed_model = build_vector_store(state.docs)
                    state.vector_store = vector_store
                    state.embed_model = embed_model
                    
                    # Create retriever
                    status_text.text("Creating ensemble retriever...")
                    progress_bar.progress(0.8)
                    retriever = EnsembleRetriever(
                        bm25_index=bm25_index,
                        vector_store=vector_store,
                        model=embed_model,
                        text_chunks=state.docs
                    )
                    state.retriever = retriever
                    state.index_built_at = datetime.now().isoformat()
                    
                    progress_bar.progress(1.0)
                    status_text.text("✓ Indexes built successfully!")
                    st.success(f"✓ Built indexes for {len(state.docs)} documents")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Index building failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Stage 7: Async Ingestion Section
        st.markdown("---")
        st.markdown("**🚀 Async Ingestion (Stage 7)**")
        
        async_col1, async_col2 = st.columns(2)
        
        with async_col1:
            if st.button("⚡ Start Async Ingestion", use_container_width=True):
                job_id = start_async_ingestion(corpus_dir)
                if job_id:
                    state.current_job_id = job_id
                    st.success(f"✓ Job started: {job_id[:8]}...")
                    st.rerun()
        
        with async_col2:
            if st.button("🔄 Check Status", use_container_width=True):
                if state.current_job_id:
                    st.rerun()
                else:
                    st.info("No active job")
        
        # Job status display
        if state.current_job_id:
            status = check_job_status(state.current_job_id)
            
            # Progress bar with cyber theme
            progress = status.get("progress", 0.0)
            st.progress(progress, text=f"{progress*100:.1f}%")
            
            # Status message with neon style
            state_emoji = {
                "queued": "⏳",
                "running": "⚙️",
                "succeeded": "✅",
                "failed": "❌",
            }
            emoji = state_emoji.get(status.get("state", "unknown"), "❓")
            st.markdown(f"**{emoji} {status.get('state', 'unknown').upper()}** • {status.get('message', 'No message')}")
            
            # Auto-refresh while running
            if status.get("state") in ["queued", "running"]:
                time.sleep(1)
                st.rerun()
            
            # Load indexes on success
            if status.get("state") == "succeeded":
                payload = status.get("payload", {})
                if "meta" in payload:
                    meta = payload["meta"]
                    st.success(f"🎉 Ingestion complete! {meta.get('doc_count', 0)} docs indexed.")
                    
                    # Show cache stats
                    if "embed_cache" in payload:
                        cache = payload["embed_cache"]
                        st.info(f"Cache: {cache.get('hits', 0)} hits, {cache.get('misses', 0)} misses ({cache.get('hit_rate', 0)*100:.1f}%)")
                
                if st.button("📥 Load Built Indexes"):
                    index_dir = Path(payload.get("index_dir", ""))
                    if index_dir.exists():
                        with st.spinner("Loading indexes..."):
                            try:
                                bm25, vectors, texts, meta = load_indexes(index_dir)
                                if bm25 and vectors is not None and texts:
                                    state.docs = texts
                                    state.bm25_index = bm25
                                    state.vector_store = vectors
                                    state.corpus_id = meta.corpus_id
                                    st.success(f"✓ Loaded {len(texts)} docs from {index_dir.name}")
                                    state.current_job_id = None
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Failed to load indexes: {e}")
        
        # Load Latest Index button
        if st.button("📥 Load Latest Index", use_container_width=True):
            index_dir, corpus_id = load_latest_index()
            if index_dir:
                with st.spinner(f"Loading indexes from {corpus_id}..."):
                    try:
                        bm25, vectors, texts, meta = load_indexes(index_dir)
                        if bm25 and vectors is not None and texts:
                            state.docs = texts
                            state.bm25_index = bm25
                            state.vector_store = vectors
                            state.corpus_id = corpus_id
                            
                            # Create retriever
                            from sentence_transformers import SentenceTransformer
                            embed_model = SentenceTransformer(meta.embed_model_name)
                            retriever = EnsembleRetriever(
                                bm25_index=bm25,
                                vector_store=vectors,
                                model=embed_model,
                                text_chunks=texts
                            )
                            state.retriever = retriever
                            state.embed_model = embed_model
                            state.index_built_at = meta.built_at
                            
                            st.success(f"✓ Loaded {len(texts)} docs from {corpus_id}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load indexes: {e}")
            else:
                st.info("No indexes found in data/indexes/")
        
        # Index health card
        st.markdown("---")
        st.markdown("**Index Status**")
        
        if state.retriever:
            metric_card("Status", "✓ Ready", "success")
            metric_card("Documents", str(len(state.docs)), "")
            if state.index_built_at:
                built_time = datetime.fromisoformat(state.index_built_at)
                metric_card("Built", built_time.strftime("%Y-%m-%d %H:%M"), "")
            if state.embed_model:
                metric_card("Model", "sentence-transformers", "")
        else:
            metric_card("Status", "✗ Not Built", "error")
            if state.docs:
                st.info(f"{len(state.docs)} documents loaded. Build indexes to enable search.")
            else:
                st.info("Load corpus first, then build indexes.")
        
        # Corpus stats
        if state.docs:
            with st.expander("📊 Corpus Statistics"):
                total_chars = sum(len(doc) for doc in state.docs)
                avg_chars = total_chars // len(state.docs) if state.docs else 0
                
                st.metric("Total Documents", len(state.docs))
                st.metric("Total Characters", f"{total_chars:,}")
                st.metric("Avg Doc Length", f"{avg_chars:,} chars")
                
                # Preview first doc
                st.markdown("**First Document Preview:**")
                st.text(state.docs[0][:300] + "...")
    
    # Stage 7: Cache Stats Sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💾 Cache Statistics")
        
        # Initialize cache if needed
        if not state.kv:
            try:
                state.kv = init_cache()
            except Exception as e:
                st.error(f"Cache init failed: {e}")
        
        if state.kv:
            try:
                cache_stats = get_cache_stats(state.kv)
                
                for table, stats in cache_stats.items():
                    with st.expander(f"📊 {table.title()}", expanded=False):
                        st.metric("Entries", stats["count"])
                        st.metric("Size", f"{stats['size_kb']:.1f} KB")
                
                # Corpus ID display
                if state.corpus_id != "empty":
                    st.info(f"🔖 Corpus: `{state.corpus_id[:12]}...`")
            except Exception as e:
                st.error(f"Cache stats error: {e}")


if __name__ == "__main__":
    main()
