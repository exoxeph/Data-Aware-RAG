"""
💬 Chat Page - Multi-Turn RAG Conversations

Interactive chat interface with:
- History-aware context
- Real-time citations
- Cache acceleration
- Neon glassmorphism styling
"""

import streamlit as st
import requests
import time
from pathlib import Path

# Import state management
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from state import (
    get_state,
    init_chat_history,
    add_chat_message,
    get_chat_messages,
    clear_chat_history
)

# Configure page
st.set_page_config(
    page_title="💬 Chat - RAG Cockpit",
    page_icon="💬",
    layout="wide"
)

# Load custom CSS
css_file = Path(__file__).parent.parent / "ui_theme.css"
if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Additional chat-specific styles
st.markdown("""
<style>
    /* Chat message containers */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: messageSlideIn 0.3s ease-out;
    }
    
    .chat-user {
        background: linear-gradient(135deg, 
            rgba(0, 255, 255, 0.1),
            rgba(0, 200, 255, 0.05)
        );
        border-left: 3px solid #00ffff;
        margin-left: 2rem;
    }
    
    .chat-assistant {
        background: linear-gradient(135deg,
            rgba(255, 0, 255, 0.1),
            rgba(200, 0, 255, 0.05)
        );
        border-left: 3px solid #ff00ff;
        margin-right: 2rem;
    }
    
    .chat-role {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    .chat-content {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .chat-timestamp {
        font-size: 0.75rem;
        opacity: 0.5;
        margin-top: 0.5rem;
    }
    
    .source-snippet {
        background: rgba(0, 255, 100, 0.05);
        border-left: 2px solid #00ff64;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    
    .cache-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .cache-hit {
        background: rgba(0, 255, 100, 0.2);
        color: #00ff64;
        border: 1px solid #00ff64;
    }
    
    .cache-miss {
        background: rgba(255, 100, 0, 0.2);
        color: #ff6400;
        border: 1px solid #ff6400;
    }
    
    @keyframes messageSlideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-block;
        font-size: 2rem;
        animation: typing 1.5s infinite;
    }
    
    @keyframes typing {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize state
state = get_state()
init_chat_history()

# Sidebar configuration
with st.sidebar:
    st.markdown("### 💬 Chat Configuration")
    
    # Corpus selection
    state.corpus_id = st.text_input(
        "Corpus ID",
        value=state.corpus_id,
        help="Identifier for the document corpus"
    )
    
    # Model selection
    state.chat_model = st.selectbox(
        "Model",
        options=["ollama:llama3", "ollama:mistral", "mock"],
        index=0
    )
    
    # Cache toggle
    state.use_cache = st.toggle(
        "Use Cache",
        value=state.use_cache,
        help="Enable answer and retrieval caching"
    )
    
    # Streaming toggle
    enable_streaming = st.toggle(
        "Stream Tokens (SSE)",
        value=False,
        help="Stream response word-by-word in real-time"
    )
    
    # Top-k results
    top_k = st.slider(
        "Top K Results",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of documents to retrieve"
    )
    
    st.markdown("---")
    
    # Session info
    st.markdown(f"**Session:** `{state.session_id}`")
    
    if state.chat_history:
        st.markdown(f"**Messages:** {len(state.chat_history)}")
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        clear_chat_history()
        st.rerun()
    
    st.markdown("---")
    
    # Cache stats (if available)
    if state.last_cache_info:
        st.markdown("### 📊 Last Query Cache")
        
        if state.last_cache_info.get("answer_hit"):
            st.success("✓ Answer Cache Hit")
        if state.last_cache_info.get("retrieval_hit"):
            st.success("✓ Retrieval Cache Hit")
        
        embed_hits = state.last_cache_info.get("embed_hits", 0)
        embed_misses = state.last_cache_info.get("embed_misses", 0)
        if embed_hits + embed_misses > 0:
            hit_rate = embed_hits / (embed_hits + embed_misses) * 100
            st.metric("Embedding Hit Rate", f"{hit_rate:.1f}%")
    
    # Memory stats (Stage 10)
    if hasattr(state, 'last_memory_info') and state.last_memory_info:
        st.markdown("### 🧠 Last Query Memory")
        
        mem_info = state.last_memory_info
        
        # Memory status
        if mem_info.get("enabled", False):
            st.success("✅ Memory ON")
        else:
            st.warning("⚠️ Memory OFF")
        
        # Usage stats
        used_count = mem_info.get("used_count", 0)
        used_chars = mem_info.get("chars", 0)
        written_count = mem_info.get("written_count", 0)
        
        if used_count > 0:
            st.info(f"📥 Used: {used_count} notes ({used_chars} chars)")
        
        if written_count > 0:
            st.info(f"📤 Wrote: {written_count} notes")
        
        # Show actual memory text (if available)
        if mem_info.get("memory_text"):
            with st.expander("View Memory Context"):
                st.markdown(mem_info["memory_text"])

# Main chat interface
st.markdown("# 💬 RAG Chat")
st.markdown("*Multi-turn conversations with your document corpus*")

# Display chat history
messages = get_chat_messages()

if not messages:
    st.info("👋 Start a conversation! Ask a question about your corpus.")
else:
    # Render messages
    for msg in messages:
        role_color = "#00ffff" if msg.role == "user" else "#ff00ff"
        role_class = "chat-user" if msg.role == "user" else "chat-assistant"
        role_icon = "👤" if msg.role == "user" else "🤖"
        
        # Format timestamp
        from datetime import datetime
        ts = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")
        
        # Build message HTML
        message_html = f"""
        <div class="chat-message {role_class}">
            <div class="chat-role" style="color: {role_color}">
                {role_icon} {msg.role.upper()}
                <span class="chat-timestamp">{ts}</span>
            </div>
            <div class="chat-content">{msg.content}</div>
        """
        
        # Add cache badge for assistant messages
        if msg.role == "assistant" and msg.metadata:
            cache_info = msg.metadata.get("cache", {})
            if cache_info.get("answer_hit"):
                message_html += '<span class="cache-badge cache-hit">⚡ CACHED</span>'
            else:
                message_html += '<span class="cache-badge cache-miss">🔄 FRESH</span>'
        
        message_html += "</div>"
        
        st.markdown(message_html, unsafe_allow_html=True)
        
        # Show sources in expander for assistant messages
        if msg.role == "assistant" and msg.metadata:
            sources = msg.metadata.get("sources", [])
            if sources:
                with st.expander(f"📚 View {len(sources)} Sources"):
                    for idx, source in enumerate(sources, 1):
                        st.markdown(f"""
                        <div class="source-snippet">
                            <strong>Source {idx}</strong> (score: {source.get('score', 0):.3f})<br>
                            {source.get('text', 'N/A')[:300]}...
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Add user message
    add_chat_message("user", user_input)
    
    # Show typing indicator
    with st.spinner("🤔 Thinking..."):
        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            '<div class="typing-indicator">⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏</div>',
            unsafe_allow_html=True
        )
        
        # Prepare chat request
        history_turns = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "query": user_input,
            "history": history_turns,
            "corpus_id": state.corpus_id,
            "session_id": state.session_id,
            "use_cache": state.use_cache,
            "refresh_cache": False,
            "top_k": top_k,
            "model": state.chat_model
        }
        
        try:
            # Call chat API (streaming or non-streaming)
            if enable_streaming:
                # Streaming mode with SSE
                response = requests.post(
                    "http://localhost:8000/api/chat/stream",
                    json=payload,
                    timeout=60,
                    stream=True
                )
                
                if response.status_code == 200:
                    # Create placeholder for incremental updates
                    message_placeholder = st.empty()
                    full_answer = ""
                    sources = []
                    metadata = {}
                    
                    # Parse SSE stream
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        line = line.decode('utf-8')
                        
                        # Parse SSE format: "event: <type>" and "data: <json>"
                        if line.startswith('event:'):
                            event_type = line.split(':', 1)[1].strip()
                        elif line.startswith('data:'):
                            try:
                                import json
                                data_str = line.split(':', 1)[1].strip()
                                event_data = json.loads(data_str)
                                
                                if event_type == 'token':
                                    # Append token and update display
                                    token = event_data.get('t', '')
                                    full_answer += token
                                    
                                    # Show typing indicator + partial answer
                                    typing_placeholder.markdown(
                                        f'<div class="typing-indicator">🤖 Generating...</div>',
                                        unsafe_allow_html=True
                                    )
                                    message_placeholder.markdown(
                                        f"**Assistant:** {full_answer}▌",
                                        unsafe_allow_html=False
                                    )
                                
                                elif event_type == 'sources':
                                    sources = event_data.get('sources', [])
                                
                                elif event_type == 'verify':
                                    metadata['verification'] = event_data
                                
                                elif event_type == 'done':
                                    # Final answer
                                    full_answer = event_data.get('answer', full_answer)
                                    metadata['cached'] = event_data.get('cached', False)
                                    metadata['duration_ms'] = event_data.get('duration_ms', 0)
                                    
                                    typing_placeholder.empty()
                                    message_placeholder.empty()
                                    
                                    # Add to history
                                    add_chat_message(
                                        "assistant",
                                        full_answer,
                                        metadata={
                                            "sources": sources,
                                            "cache": {"answer_hit": metadata.get('cached', False)},
                                            "latency_ms": metadata.get('duration_ms', 0)
                                        }
                                    )
                                    
                                    # Show success
                                    if metadata.get('cached'):
                                        st.success(f"✨ Streamed from cache in {metadata['duration_ms']:.1f}ms")
                                    else:
                                        st.success(f"✅ Streamed new answer in {metadata['duration_ms']:.1f}ms")
                                    
                                    st.rerun()
                                
                                elif event_type == 'error':
                                    st.error(f"❌ Error: {event_data.get('message')}")
                                    typing_placeholder.empty()
                                    break
                            
                            except Exception as parse_error:
                                st.warning(f"⚠️ Parse error: {parse_error}")
                                continue
                else:
                    typing_placeholder.empty()
                    st.error(f"❌ Streaming API Error: {response.status_code}")
            
            else:
                # Non-streaming mode (original)
                response = requests.post(
                    "http://localhost:8000/api/chat",
                    json=payload,
                    timeout=30
                )
                
                typing_placeholder.empty()
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract response
                    answer = data.get("answer", "No answer received")
                    sources = data.get("sources", [])
                    cache_info = data.get("cache", {})
                    latency = data.get("latency_ms", 0)
                    memory_info = data.get("memory", {})
                    
                    # Store memory info in state
                    if memory_info:
                        state.last_memory_info = {
                            "enabled": True,
                            "used_count": memory_info.get("used_count", 0),
                            "chars": memory_info.get("chars", 0),
                            "written_count": memory_info.get("written_count", 0),
                            "memory_text": None  # Would need to fetch separately
                        }
                    
                    # Add assistant message with metadata
                    add_chat_message(
                        "assistant",
                        answer,
                        metadata={
                            "sources": sources,
                            "cache": cache_info,
                            "latency_ms": latency
                        }
                    )
                    
                    # Update cache info in state
                    state.last_cache_info = cache_info
                    
                    # Show success with latency
                    if cache_info.get("answer_hit"):
                        st.success(f"✨ Answer retrieved from cache in {latency:.1f}ms")
                    else:
                        st.success(f"✅ Generated new answer in {latency:.1f}ms")
                    
                    # Rerun to show new message
                    st.rerun()
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
        
        except requests.exceptions.ConnectionError:
            typing_placeholder.empty()
            st.error("❌ Cannot connect to API server. Start with: `poetry run rag-serve`")
        except Exception as e:
            typing_placeholder.empty()
            st.error(f"❌ Error: {str(e)}")

# Footer with tips
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.6; font-size: 0.9rem;">
    💡 <strong>Tips:</strong> 
    Use "Clear History" to start fresh • 
    Enable cache for faster responses • 
    Cyan = your messages, Magenta = AI responses
</div>
""", unsafe_allow_html=True)
