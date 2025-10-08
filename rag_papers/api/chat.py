"""
Chat API endpoint for multi-turn RAG conversations.

Integrates retrieval, generation, and caching with conversation history.
"""

import time
import json
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncIterator

from rag_papers.persist.chat_history import ChatHistory, create_session_history
from rag_papers.persist.answer_cache import get_answer, set_answer, AnswerKey, AnswerValue
from rag_papers.persist.retrieval_cache import cached_search, RetrievalKey


router = APIRouter()


class ChatTurn(BaseModel):
    """Single message in chat history."""
    
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request with history context."""
    
    query: str = Field(..., description="User's current question")
    history: List[ChatTurn] = Field(default_factory=list, description="Previous conversation turns")
    corpus_id: str = Field(..., description="Corpus identifier")
    session_id: Optional[str] = Field(None, description="Session ID for history tracking")
    
    # RAG configuration
    use_cache: bool = Field(True, description="Use answer/retrieval cache")
    refresh_cache: bool = Field(False, description="Force cache refresh")
    top_k: int = Field(5, description="Number of results to retrieve")
    
    # Model configuration
    model: str = Field("ollama:llama3", description="LLM model identifier")
    temperature: float = Field(0.1, description="Generation temperature")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is transfer learning?",
                "history": [
                    {"role": "user", "content": "Explain neural networks"},
                    {"role": "assistant", "content": "Neural networks are..."}
                ],
                "corpus_id": "ml_papers",
                "use_cache": True,
                "model": "ollama:llama3"
            }
        }


class SourceSnippet(BaseModel):
    """Source document with relevance score."""
    
    text: str = Field(..., description="Document text snippet")
    score: float = Field(..., description="Relevance score")
    rank: int = Field(..., description="Result rank")


class CacheInfo(BaseModel):
    """Cache hit/miss information."""
    
    answer_hit: bool = Field(False, description="Answer cache hit")
    retrieval_hit: bool = Field(False, description="Retrieval cache hit")
    embed_hits: int = Field(0, description="Embedding cache hits")
    embed_misses: int = Field(0, description="Embedding cache misses")


class ChatResponse(BaseModel):
    """Chat response with answer, sources, and metadata."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceSnippet] = Field(default_factory=list, description="Source documents")
    cache: CacheInfo = Field(default_factory=CacheInfo, description="Cache statistics")
    latency_ms: float = Field(..., description="Total response time in milliseconds")
    session_id: Optional[str] = Field(None, description="Session identifier")
    memory: Optional[Dict[str, Any]] = Field(
        None,
        description="Memory usage: {used_ids, used_count, written, chars}"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Multi-turn RAG chat with history-aware retrieval and caching.
    
    Flow:
    1. Build context from conversation history
    2. Check answer cache (query + history + corpus + model)
    3. If cache miss:
       a. Retrieve relevant documents (with retrieval cache)
       b. Generate answer with LLM
       c. Store in answer cache
    4. Return answer + sources + cache info
    
    Args:
        request: Chat request with query, history, and config
    
    Returns:
        ChatResponse with answer, sources, and metadata
    """
    start_time = time.time()
    
    # Initialize cache info
    cache_info = CacheInfo()
    
    # Build context from history (last 5 turns)
    history_context = ""
    if request.history:
        recent_history = request.history[-5:]  # Keep last 5 turns
        history_lines = [
            f"{turn.role.capitalize()}: {turn.content}"
            for turn in recent_history
        ]
        history_context = "\n".join(history_lines) + "\n"
    
    # Full query with history
    full_query = f"{history_context}User: {request.query}\nAssistant:"
    
    # Check answer cache (if enabled and not refresh)
    answer_cached = None
    if request.use_cache and not request.refresh_cache:
        answer_key = AnswerKey(
            query=request.query,
            corpus_id=request.corpus_id,
            model=request.model,
            top_k=request.top_k,
            history_hash=str(hash(history_context))  # Include history in cache key
        )
        
        from rag_papers.persist.sqlite_store import KVStore
        from pathlib import Path
        
        # Use default cache location
        kv = KVStore(Path("data") / "cache" / "kv_store.db")
        answer_cached = get_answer(answer_key, kv)
        
        if answer_cached is not None:
            cache_info.answer_hit = True
            cache_info.retrieval_hit = True  # Implies retrieval was cached too
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Convert sources from cached format
            sources = [
                SourceSnippet(text=text, score=score, rank=idx + 1)
                for idx, (text, score) in enumerate(answer_cached.sources[:request.top_k])
            ]
            
            return ChatResponse(
                answer=answer_cached.answer,
                sources=sources,
                cache=cache_info,
                latency_ms=latency_ms,
                session_id=request.session_id
            )
    
    # Cache miss - run full RAG pipeline
    try:
        # Import retriever and generator (lazy to avoid circular imports)
        from rag_papers.retrieval.ensemble_retriever import EnsembleRetriever
        from rag_papers.generation.generator import MockGenerator
        
        # TODO: Load actual retriever from corpus index
        # For now, return a helpful message
        answer = (
            f"I would retrieve relevant documents about '{request.query}' "
            f"from corpus '{request.corpus_id}' and generate an answer. "
            f"\n\nTo enable full RAG:\n"
            f"1. Ingest documents to build index\n"
            f"2. Load retriever from index\n"
            f"3. Configure generator (Ollama/OpenAI)\n\n"
            f"Chat history context ({len(request.history)} turns) is captured."
        )
        
        # Mock sources
        sources = [
            SourceSnippet(
                text=f"Source document about {request.query} (placeholder)",
                score=0.85,
                rank=1
            )
        ]
        
        # Store in cache for next time
        if request.use_cache:
            answer_value = AnswerValue(
                answer=answer,
                sources=[(s.text, s.score) for s in sources],
                model=request.model,
                timestamp=time.time()
            )
            
            answer_key = AnswerKey(
                query=request.query,
                corpus_id=request.corpus_id,
                model=request.model,
                top_k=request.top_k,
                history_hash=str(hash(history_context))
            )
            
            from rag_papers.persist.sqlite_store import KVStore
            from pathlib import Path
            kv = KVStore(Path("data") / "cache" / "kv_store.db")
            set_answer(answer_key, answer_value, kv)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            cache=cache_info,
            latency_ms=latency_ms,
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.post("/chat/clear")
async def clear_chat_history(session_id: str):
    """
    Clear chat history for a session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Success message
    """
    try:
        history = create_session_history(session_id)
        history.clear()
        
        return {"message": f"Chat history cleared for session {session_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear history: {str(e)}"
        )


@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 50):
    """
    Retrieve chat history for a session.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of messages to return
    
    Returns:
        List of chat messages
    """
    try:
        history = create_session_history(session_id)
        messages = history.get_recent_messages(count=limit)
        
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    
    Returns a stream of events:
    - event: meta -> {"run_id", "corpus_id", "retrieved", "ctx_chars"}
    - event: token -> {"t": " word "}
    - event: sources -> {"sources": [{"doc", "score"}]}
    - event: verify -> {"score", "accepted", "issues"}
    - event: done -> {"answer", "cached", "duration_ms"}
    - event: error -> {"message"}
    
    Args:
        request: ChatRequest with query, history, and configuration
    
    Returns:
        StreamingResponse with text/event-stream content type
    """
    
    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events."""
        start_time = time.time()
        full_answer = ""
        cached = False
        
        try:
            # Import dependencies (lazy to avoid circular imports)
            from rag_papers.retrieval.router_dag import run_chat_plan_stream
            from rag_papers.index.build_vector_store import load_or_build_ensemble
            from rag_papers.generation.generator import MockGenerator
            from rag_papers.config.settings import Stage4Config
            
            # Initialize components
            cfg = Stage4Config()
            retriever = load_or_build_ensemble(request.corpus_id, cfg)
            generator = MockGenerator()  # TODO: Make configurable
            
            # Convert history format
            history_dicts = [
                {"role": turn.role, "content": turn.content}
                for turn in request.history
            ]
            
            # Stream events from router
            async for event in run_chat_plan_stream(
                query=request.query,
                history=history_dicts,
                retriever=retriever,
                generator=generator,
                cfg=cfg,
                use_cache=request.use_cache
            ):
                event_type = event.get("type")
                
                if event_type == "meta":
                    # Metadata about run
                    yield f"event: meta\n"
                    yield f"data: {json.dumps(event['data'])}\n\n"
                
                elif event_type == "token":
                    # Stream token
                    token = event["data"]["t"]
                    full_answer += token
                    yield f"event: token\n"
                    yield f"data: {json.dumps({'t': token})}\n\n"
                    await asyncio.sleep(0)  # Yield control
                
                elif event_type == "sources":
                    # Retrieved sources
                    yield f"event: sources\n"
                    yield f"data: {json.dumps(event['data'])}\n\n"
                
                elif event_type == "verify":
                    # Verification result
                    yield f"event: verify\n"
                    yield f"data: {json.dumps(event['data'])}\n\n"
                
                elif event_type == "cached":
                    # Cache hit - tokenize and stream
                    cached = True
                    answer = event["data"]["answer"]
                    full_answer = answer
                    
                    # Simulate streaming by splitting into words
                    words = answer.split()
                    for word in words:
                        token = word + " "
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'t': token})}\n\n"
                        await asyncio.sleep(0.005)  # Small delay
            
            # Final event
            duration_ms = int((time.time() - start_time) * 1000)
            done_data = {
                "answer": full_answer.strip(),
                "cached": cached,
                "duration_ms": duration_ms
            }
            yield f"event: done\n"
            yield f"data: {json.dumps(done_data)}\n\n"
            
            # Persist to history if session_id provided
            if request.session_id:
                history = create_session_history(request.session_id)
                history.add("user", request.query)
                history.add("assistant", full_answer.strip(), metadata={"cached": cached})
                history.save()
        
        except Exception as e:
            error_data = {"message": str(e)}
            yield f"event: error\n"
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

