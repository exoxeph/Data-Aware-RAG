"""Contextualization module for combining retrieved chunks into coherent context."""
from __future__ import annotations
from typing import List, Optional


def contextualize(
    retrieved_chunks: List[str], 
    query: str, 
    max_length: int = 4000,
    include_query: bool = True
) -> str:
    """
    Combine retrieved text chunks into a coherent context for the LLM.

    Args:
        retrieved_chunks (List[str]): List of retrieved text chunks.
        query (str): The query provided by the user.
        max_length (int): Maximum length of the context in characters.
        include_query (bool): Whether to include the query in the context.

    Returns:
        str: The contextualized prompt for the LLM.
    """
    if not retrieved_chunks:
        if include_query:
            return f"Query: {query}\n\nNo relevant context found."
        return "No relevant context found."
    
    # Filter out empty chunks
    valid_chunks = [chunk.strip() for chunk in retrieved_chunks if chunk and chunk.strip()]
    
    if not valid_chunks:
        if include_query:
            return f"Query: {query}\n\nNo relevant context found."
        return "No relevant context found."
    
    # Build context
    context_parts = []
    
    if include_query:
        context_parts.append(f"Query: {query}\n")
    
    context_parts.append("Relevant Context:")
    
    # Add chunks with separators
    current_length = len("\n".join(context_parts))
    added_chunks = []
    
    for i, chunk in enumerate(valid_chunks):
        chunk_header = f"\n\n[Context {i+1}]\n"
        chunk_content = chunk
        chunk_text = chunk_header + chunk_content
        
        # Check if adding this chunk would exceed max length
        if current_length + len(chunk_text) > max_length:
            # If we haven't added any chunks yet, add a truncated version
            if not added_chunks:
                remaining_space = max_length - current_length - len(chunk_header)
                if remaining_space > 100:  # Only add if we have reasonable space
                    truncated_chunk = chunk_content[:remaining_space-3] + "..."
                    added_chunks.append(chunk_header + truncated_chunk)
            break
        
        added_chunks.append(chunk_text)
        current_length += len(chunk_text)
    
    # Combine all parts
    context_parts.extend(added_chunks)
    
    return "\n".join(context_parts)


def contextualize_with_metadata(
    retrieved_chunks: List[str], 
    query: str,
    chunk_metadata: Optional[List[dict]] = None,
    max_length: int = 4000,
    include_sources: bool = True
) -> str:
    """
    Enhanced contextualization that includes metadata about sources.
    
    Args:
        retrieved_chunks: List of retrieved text chunks
        query: The user's query
        chunk_metadata: Optional metadata for each chunk (file, page, etc.)
        max_length: Maximum context length
        include_sources: Whether to include source information
        
    Returns:
        Formatted context string with metadata
    """
    if not retrieved_chunks:
        return f"Query: {query}\n\nNo relevant context found."
    
    # Filter valid chunks
    valid_chunks = []
    valid_metadata = []
    
    for i, chunk in enumerate(retrieved_chunks):
        if chunk and chunk.strip():
            valid_chunks.append(chunk.strip())
            if chunk_metadata and i < len(chunk_metadata):
                valid_metadata.append(chunk_metadata[i])
            else:
                valid_metadata.append({})
    
    if not valid_chunks:
        return f"Query: {query}\n\nNo relevant context found."
    
    # Build context
    context_parts = [f"Query: {query}\n", "Relevant Context:"]
    current_length = len("\n".join(context_parts))
    
    for i, chunk in enumerate(valid_chunks):
        # Create chunk header with metadata
        metadata = valid_metadata[i] if i < len(valid_metadata) else {}
        
        if include_sources and metadata:
            source_info = []
            if 'file' in metadata:
                source_info.append(f"File: {metadata['file']}")
            if 'page' in metadata:
                source_info.append(f"Page: {metadata['page']}")
            
            if source_info:
                chunk_header = f"\n\n[Context {i+1} - {', '.join(source_info)}]\n"
            else:
                chunk_header = f"\n\n[Context {i+1}]\n"
        else:
            chunk_header = f"\n\n[Context {i+1}]\n"
        
        chunk_text = chunk_header + chunk
        
        # Check length constraint
        if current_length + len(chunk_text) > max_length:
            if i == 0:  # Always include at least one chunk
                remaining_space = max_length - current_length - len(chunk_header)
                if remaining_space > 100:
                    truncated = chunk[:remaining_space-3] + "..."
                    context_parts.append(chunk_header + truncated)
            break
        
        context_parts.append(chunk_text)
        current_length += len(chunk_text)
    
    return "\n".join(context_parts)


def format_for_llm(
    retrieved_chunks: List[str], 
    query: str,
    system_prompt: Optional[str] = None
) -> str:
    """
    Format retrieved context specifically for LLM consumption.
    
    Args:
        retrieved_chunks: Retrieved text chunks
        query: User query
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Formatted prompt ready for LLM
    """
    context = contextualize(retrieved_chunks, query, include_query=False)
    
    parts = []
    
    if system_prompt:
        parts.append(system_prompt)
    
    parts.extend([
        context,
        f"\nQuestion: {query}",
        "\nPlease provide a comprehensive answer based on the context above."
    ])
    
    return "\n".join(parts)


def make_context_string(title: str, section_path: str, neighbor_terms: list[str]) -> str:
    """Generate context string for search results.
    
    Args:
        title: Document or paper title
        section_path: Hierarchical section path (e.g., "Methods > Architecture")
        neighbor_terms: List of related terms found nearby
        
    Returns:
        Formatted context string with title, section, and terms
    """
    # Remove duplicates while preserving order
    terms = ", ".join(dict.fromkeys(neighbor_terms))
    return f"{title} | {section_path} | {terms}"