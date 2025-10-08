"""
🧠 Memory Management Page

View, search, and manage conversation memories across sessions.
"""

import streamlit as st
import requests
import time
from typing import List, Dict, Any

# Page config
st.set_page_config(
    page_title="Memory - RAG Papers",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Memory Management")
st.markdown("""
Manage long-term conversation memories with semantic recall and summarization.
""")

# API base URL
API_BASE = st.session_state.get("api_base", "http://localhost:8000/api")

# ============================================================================
# State Management
# ============================================================================

if "memory_enabled" not in st.session_state:
    st.session_state.memory_enabled = True

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"sess_{int(time.time())}"


# ============================================================================
# Memory Controls
# ============================================================================

st.subheader("Memory Controls")

col1, col2, col3 = st.columns(3)

with col1:
    # Toggle memory
    memory_enabled = st.checkbox(
        "Enable Memory",
        value=st.session_state.memory_enabled,
        help="Enable or disable memory injection and writing"
    )
    st.session_state.memory_enabled = memory_enabled
    
    if memory_enabled:
        st.success("✅ Memory is ON")
    else:
        st.warning("⚠️ Memory is OFF")

with col2:
    # Current session
    st.text_input(
        "Current Session ID",
        value=st.session_state.current_session_id,
        key="session_id_input",
        help="Session identifier for memory scoping"
    )
    st.session_state.current_session_id = st.session_state.session_id_input

with col3:
    # Purge session memories
    if st.button("🗑️ Purge Session Memory", type="secondary"):
        try:
            response = requests.post(
                f"{API_BASE}/memory/purge",
                params={"session_id": st.session_state.current_session_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ {data['message']}")
            else:
                st.error(f"Failed to purge: {response.text}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ============================================================================
# Memory Search
# ============================================================================

st.markdown("---")
st.subheader("🔍 Search Memories")

search_col1, search_col2 = st.columns([3, 1])

with search_col1:
    search_query = st.text_input(
        "Search Query",
        placeholder="What is transfer learning?",
        help="Semantic search across memory notes"
    )

with search_col2:
    search_scope = st.selectbox(
        "Scope",
        options=["session", "corpus", "global"],
        index=0,
        help="Memory scope to search in"
    )
    
    search_limit = st.number_input(
        "Limit",
        min_value=1,
        max_value=100,
        value=10,
        help="Maximum results"
    )

if st.button("Search", type="primary"):
    if search_query:
        try:
            response = requests.post(
                f"{API_BASE}/memory/search",
                json={
                    "query": search_query,
                    "scope": search_scope,
                    "session_id": st.session_state.current_session_id,
                    "limit": search_limit
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                memories = data.get("memories", [])
                
                st.success(f"Found {len(memories)} memories")
                
                if memories:
                    for i, mem in enumerate(memories, 1):
                        with st.expander(f"**{i}. Score: {mem.get('score', 0):.3f}** - {mem.get('text', '')[:80]}..."):
                            st.markdown(f"**Full Text:** {mem.get('text', '')}")
                            st.markdown(f"**Tags:** {', '.join(mem.get('tags', []))}")
                            st.markdown(f"**Source:** {mem.get('source', 'unknown')}")
                            st.markdown(f"**Uses:** {mem.get('uses', 0)} | **Created:** {time.ctime(mem.get('created_at', 0))}")
                            
                            # Delete button
                            if st.button(f"Delete", key=f"delete_{mem['id']}"):
                                del_response = requests.delete(
                                    f"{API_BASE}/memory/delete",
                                    json={"note_id": mem["id"]}
                                )
                                if del_response.status_code == 200:
                                    st.success("Deleted!")
                                    st.rerun()
                else:
                    st.info("No memories found")
            else:
                st.error(f"Search failed: {response.text}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Enter a search query")


# ============================================================================
# Add Memory Note
# ============================================================================

st.markdown("---")
st.subheader("➕ Add Memory Note")

add_col1, add_col2 = st.columns([3, 1])

with add_col1:
    note_text = st.text_area(
        "Memory Text",
        placeholder="Transfer learning uses pretrained models on new tasks...",
        help="Concise memory note (1-3 sentences)"
    )

with add_col2:
    note_scope = st.selectbox(
        "Scope",
        options=["session", "corpus", "global"],
        index=0,
        help="Memory scope"
    )
    
    note_tags = st.text_input(
        "Tags (comma-separated)",
        placeholder="concept:transfer, entity:model",
        help="Tags for categorization"
    )

if st.button("Add Note", type="primary"):
    if note_text:
        try:
            tags = [t.strip() for t in note_tags.split(",") if t.strip()]
            
            response = requests.post(
                f"{API_BASE}/memory/add",
                json={
                    "text": note_text,
                    "scope": note_scope,
                    "tags": tags,
                    "source": "manual",
                    "meta": {"session_id": st.session_state.current_session_id}
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ {data['message']} (ID: {data['note_id']})")
            else:
                st.error(f"Failed to add note: {response.text}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Enter memory text")


# ============================================================================
# List All Session Memories
# ============================================================================

st.markdown("---")
st.subheader("📋 All Session Memories")

list_col1, list_col2, list_col3 = st.columns(3)

with list_col1:
    list_scope = st.selectbox(
        "Filter Scope",
        options=["session", "corpus", "global", "all"],
        index=0,
        key="list_scope"
    )

with list_col2:
    list_sort = st.selectbox(
        "Sort By",
        options=["recency", "uses"],
        index=0,
        key="list_sort"
    )

with list_col3:
    list_limit = st.number_input(
        "Max Results",
        min_value=1,
        max_value=1000,
        value=100,
        key="list_limit"
    )

if st.button("Load Memories", type="primary"):
    try:
        params = {
            "limit": list_limit,
            "sort_by": list_sort
        }
        
        if list_scope != "all":
            params["scope"] = list_scope
        
        response = requests.get(
            f"{API_BASE}/memory/list",
            params=params
        )
        
        if response.status_code == 200:
            data = response.json()
            memories = data.get("memories", [])
            
            st.info(f"Loaded {len(memories)} memories")
            
            if memories:
                # Display as table
                table_data = []
                for mem in memories:
                    table_data.append({
                        "Text": mem.get("text", "")[:100] + "..." if len(mem.get("text", "")) > 100 else mem.get("text", ""),
                        "Scope": mem.get("scope", ""),
                        "Tags": ", ".join(mem.get("tags", [])[:3]),
                        "Uses": mem.get("uses", 0),
                        "Source": mem.get("source", ""),
                        "Created": time.strftime("%Y-%m-%d %H:%M", time.localtime(mem.get("created_at", 0)))
                    })
                
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=False
                )
            else:
                st.info("No memories found")
        else:
            st.error(f"Failed to load: {response.text}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")


# ============================================================================
# Memory Stats
# ============================================================================

st.markdown("---")
st.subheader("📊 Memory Statistics")

if st.button("Refresh Stats"):
    try:
        # Get counts for each scope
        stats_data = {}
        
        for scope in ["session", "corpus", "global"]:
            response = requests.get(
                f"{API_BASE}/memory/list",
                params={"scope": scope, "limit": 10000}
            )
            
            if response.status_code == 200:
                data = response.json()
                count = data.get("count", 0)
                memories = data.get("memories", [])
                
                # Calculate total uses
                total_uses = sum(m.get("uses", 0) for m in memories)
                
                stats_data[scope] = {
                    "count": count,
                    "total_uses": total_uses
                }
        
        # Display stats
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            st.metric(
                "Session Memories",
                stats_data.get("session", {}).get("count", 0),
                delta=f"{stats_data.get('session', {}).get('total_uses', 0)} uses"
            )
        
        with stat_col2:
            st.metric(
                "Corpus Memories",
                stats_data.get("corpus", {}).get("count", 0),
                delta=f"{stats_data.get('corpus', {}).get('total_uses', 0)} uses"
            )
        
        with stat_col3:
            st.metric(
                "Global Memories",
                stats_data.get("global", {}).get("count", 0),
                delta=f"{stats_data.get('global', {}).get('total_uses', 0)} uses"
            )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
