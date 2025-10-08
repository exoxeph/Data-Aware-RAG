"""
Ask Page - Run queries through the DAG and inspect results.
Stage 7: Added caching toggle, cache hit badges, and session export.
"""

import streamlit as st
import sys
import json
from pathlib import Path
from datetime import datetime
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.state import get_state, add_to_history, export_session
from app.components import (
    backend_status, knobs, retrieval_table, timings_bar,
    highlight_overlap, metric_card, step_path_display
)
from app.adapters import OllamaGenerator

from rag_papers.retrieval.router_dag import run_plan
from rag_papers.generation.generator import MockGenerator


def get_generator(state):
    """Get the appropriate generator based on state."""
    if state.generator_name == "ollama":
        try:
            return OllamaGenerator(model=state.ollama_model)
        except RuntimeError as e:
            st.warning(f"⚠️ {str(e)} Falling back to Mock generator.")
            state.generator_name = "mock"
            return MockGenerator()
    else:
        return MockGenerator()


def extract_context_metadata(ctx: dict) -> dict:
    """Extract metrics from context dict."""
    meta = ctx.get("meta", {})
    verify_dict = ctx.get("verify", {})
    
    return {
        "path": meta.get("path", []),
        "timings": meta.get("timings", {}),
        "verify_score": float(verify_dict.get("score", 0.0)),
        "accepted": verify_dict.get("score", 0.0) >= 0.72,  # Will use cfg threshold
        "retrieved_count": len(ctx.get("candidates", [])),
        "context_chars": len(ctx.get("context_text", "")),
        "pruned_chars": 0,  # Calculate if needed
        "candidates": ctx.get("candidates", []),
        "pruned_candidates": ctx.get("pruned_candidates", []),
        "context_text": ctx.get("context_text", ""),
        "verify_dict": verify_dict,
    }


def main():
    st.title("🤖 Ask the RAG")
    st.markdown("Query your corpus and inspect every step of the pipeline.")
    
    state = get_state()
    
    # Check if indexes are built
    if not state.retriever:
        st.error("⚠️ Indexes not built! Go to 🏗️ Corpus page and build indexes first.")
        st.stop()
    
    # Backend selector
    st.markdown("### Backend")
    col_b1, col_b2 = st.columns([1, 3])
    
    with col_b1:
        backend_choice = st.radio(
            "Generator",
            options=["mock", "ollama"],
            index=0 if state.generator_name == "mock" else 1,
            format_func=lambda x: "🎭 Mock (Offline)" if x == "mock" else "🦙 Ollama (Local)",
            key="backend_radio"
        )
        
        if backend_choice != state.generator_name:
            state.generator_name = backend_choice
    
    with col_b2:
        backend_status(state.ollama_available, state.generator_name)
        
        if state.generator_name == "ollama":
            state.ollama_model = st.selectbox(
                "Model",
                options=["llama3", "mistral", "phi", "llama2"],
                index=0,
                key="ollama_model_select"
            )
    
    # Stage 7: Cache Toggle
    st.markdown("---")
    cache_col1, cache_col2, cache_col3 = st.columns([1, 1, 2])
    
    with cache_col1:
        state.use_cache = st.checkbox(
            "💾 Use Cache",
            value=state.use_cache,
            help="Enable answer caching for faster repeat queries"
        )
    
    with cache_col2:
        if state.use_cache:
            st.markdown(
                '<div style="background: linear-gradient(90deg, #00F5FF, #FF00D4); '
                'padding: 4px 12px; border-radius: 8px; text-align: center; '
                'font-weight: bold; color: black; margin-top: 8px;">'
                'CACHE ON</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background: #333; padding: 4px 12px; '
                'border-radius: 8px; text-align: center; color: #888; margin-top: 8px;">'
                'CACHE OFF</div>',
                unsafe_allow_html=True
            )
    
    with cache_col3:
        # Show last cache info if available
        if state.last_cache_info:
            cache_badges = []
            if state.last_cache_info.get("answer_hit"):
                cache_badges.append("🎯 Answer Hit")
            if state.last_cache_info.get("retrieval_hit"):
                cache_badges.append("🔍 Retrieval Hit")
            embed_hits = state.last_cache_info.get("embed_hits", 0)
            if embed_hits > 0:
                cache_badges.append(f"🧩 {embed_hits} Embed Hits")
            
            if cache_badges:
                st.markdown(
                    f'<div style="margin-top: 8px; color: #00F5FF;">'
                    f'{" • ".join(cache_badges)}</div>',
                    unsafe_allow_html=True
                )
    
    st.markdown("---")
    
    # Query panel
    col_q1, col_q2 = st.columns([2, 1])
    
    with col_q1:
        query = st.text_input(
            "Query",
            placeholder="What is transfer learning?",
            key="query_input"
        )
    
    with col_q2:
        intent = st.selectbox(
            "Intent",
            options=["auto", "definition", "explanation", "comparison", "how_to", "summarization"],
            index=0,
            key="intent_select"
        )
    
    # Configuration knobs
    with st.expander("⚙️ Configuration", expanded=False):
        updated_cfg = knobs(state.cfg, prefix="ask_")
        state.cfg = updated_cfg
    
    # Run button
    run_button = st.button("▶️ Run Plan", type="primary", use_container_width=True)
    
    if run_button and query:
        start_time = time.time()
        
        with st.spinner("Running DAG pipeline..."):
            try:
                # Get generator
                generator = get_generator(state)
                
                # Run plan
                actual_intent = intent if intent != "auto" else "definition"
                answer, ctx = run_plan(
                    query=query,
                    intent=actual_intent,
                    retriever=state.retriever,
                    generator=generator,
                    cfg=state.cfg
                )
                
                elapsed = time.time() - start_time
                
                # Extract metadata
                metadata = extract_context_metadata(ctx)
                metadata["accepted"] = metadata["verify_score"] >= state.cfg.accept_threshold
                
                # Stage 7: Extract cache info if present
                cache_info = ctx.get("cache", {})
                if cache_info:
                    state.last_cache_info = cache_info
                else:
                    state.last_cache_info = {}
                
                # Save to state
                state.last_run = {
                    "query": query,
                    "intent": actual_intent,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_ms": elapsed * 1000,
                    **metadata
                }
                
                # Add to history
                add_to_history({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "query": query[:50],
                    "score": metadata["verify_score"],
                    "accepted": metadata["accepted"],
                    "repair": "repair" in metadata["path"],
                    "retrieved": metadata["retrieved_count"],
                    "latency_ms": elapsed * 1000,
                })
                
                st.success(f"✓ Completed in {elapsed:.2f}s")
                
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Results display
    if state.last_run:
        st.markdown("---")
        st.markdown("## 📊 Results")
        
        run = state.last_run
        
        # Three-column layout
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### 💬 Answer")
            
            # Status badges
            if run["accepted"]:
                metric_card("Status", "✓ Accepted", "success")
            else:
                metric_card("Status", "✗ Rejected", "error")
            
            metric_card("Score", f"{run['verify_score']:.3f}", "success" if run["accepted"] else "warning")
            
            if "repair" in run["path"]:
                metric_card("Repair", "Used", "warning")
            
            # Answer display
            st.markdown("**Generated Answer:**")
            st.code(run["answer"], language="markdown")
            
            # Copy button hint
            st.caption("Use the copy button in the code block above ↗️")
        
        with col2:
            st.markdown("### 📚 Context & Sources")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Retrieved", "Pruned", "Final Context", "Verify"])
            
            with tab1:
                st.markdown(f"**Retrieved {run['retrieved_count']} candidates**")
                show_meta = st.checkbox("Show metadata", key="meta_retrieved")
                retrieval_table(run["candidates"], show_meta=show_meta)
            
            with tab2:
                pruned = run.get("pruned_candidates", [])
                st.markdown(f"**Pruned: {len(pruned)} candidates**")
                if pruned:
                    show_meta_p = st.checkbox("Show metadata", key="meta_pruned")
                    retrieval_table(pruned, show_meta=show_meta_p)
                else:
                    st.info("No pruned candidates available")
            
            with tab3:
                context_text = run.get("context_text", "")
                st.markdown(f"**Final context: {len(context_text)} chars**")
                
                highlight_check = st.checkbox("Highlight query terms", key="highlight_context")
                
                if highlight_check:
                    highlighted = highlight_overlap(context_text, run["query"])
                    st.markdown(highlighted, unsafe_allow_html=True)
                else:
                    st.text_area(
                        "Context",
                        value=context_text,
                        height=300,
                        disabled=True,
                        key="context_display"
                    )
            
            with tab4:
                verify = run.get("verify_dict", {})
                st.json(verify)
        
        with col3:
            st.markdown("### 🔄 Execution Plan")
            
            # Path display
            step_path_display(run["path"])
            
            st.markdown(f"**Latency: {run['elapsed_ms']:.1f} ms**")
            
            # Timings
            st.markdown("**Step Timings:**")
            timings_bar(run["timings"])
            
            # Raw context expander
            with st.expander("🔍 Raw Context (JSON)"):
                # Create a clean version for display
                display_ctx = {
                    "query": run["query"],
                    "intent": run["intent"],
                    "path": run["path"],
                    "timings": run["timings"],
                    "verify_score": run["verify_score"],
                    "accepted": run["accepted"],
                    "retrieved_count": run["retrieved_count"],
                    "context_chars": run["context_chars"],
                }
                st.json(display_ctx)
        
        # History section
        st.markdown("---")
        st.markdown("### 📜 Session History")
        
        if state.run_history:
            import pandas as pd
            
            df = pd.DataFrame(state.run_history)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            col_h1, col_h2 = st.columns([1, 3])
            
            with col_h1:
                if st.button("💾 Save Conversation", use_container_width=True):
                    # Stage 7: Use helper function
                    export_dir = Path("runs") / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    export_file = export_dir / "session.jsonl"
                    
                    try:
                        export_session(state.run_history, export_file)
                        state.session_file = export_file
                        
                        # Neon success message
                        st.markdown(
                            '<div style="background: linear-gradient(90deg, #00FF88, #00F5FF); '
                            'padding: 8px; border-radius: 8px; text-align: center; '
                            'font-weight: bold; color: black; margin-top: 8px;">'
                            f'✓ Saved to {export_file.name}</div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            with col_h2:
                st.caption(f"{len(state.run_history)} queries in this session")
        else:
            st.info("No queries run yet in this session")
    
    else:
        st.info("👆 Enter a query and click 'Run Plan' to see results")


if __name__ == "__main__":
    main()
