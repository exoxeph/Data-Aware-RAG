"""
Pydantic schemas for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AskRequest(BaseModel):
    """Request model for /ask endpoint."""
    
    query: str = Field(..., description="User query")
    intent: str = Field(default="unknown", description="Query intent (definition, explanation, comparison, how_to, summarization, unknown)")
    bm25_weight: Optional[float] = Field(default=None, description="BM25 weight for ensemble retrieval")
    vector_weight: Optional[float] = Field(default=None, description="Vector weight for ensemble retrieval")
    temperature: Optional[float] = Field(default=None, description="LLM temperature for generation")
    top_k: Optional[int] = Field(default=None, description="Number of documents to retrieve")
    rerank_top_k: Optional[int] = Field(default=None, description="Number of documents after reranking")
    accept_threshold: Optional[float] = Field(default=None, description="Acceptance threshold for verification")
    use_cache: bool = Field(default=True, description="Whether to use caching (Stage 7)")
    refresh: bool = Field(default=False, description="Force cache refresh (Stage 7)")


class AskResponse(BaseModel):
    """Response model for /ask endpoint."""
    
    answer: str = Field(..., description="Generated answer")
    score: float = Field(..., description="Verification score")
    accepted: bool = Field(..., description="Whether answer was accepted")
    path: List[str] = Field(..., description="Execution path through DAG")
    timings: Dict[str, float] = Field(..., description="Step timings in milliseconds")
    sources: List[str] = Field(default_factory=list, description="Source document references")
    retrieved_count: int = Field(default=0, description="Number of documents retrieved")
    context_chars: int = Field(default=0, description="Number of context characters used")
    repair_used: bool = Field(default=False, description="Whether repair step was triggered")
    cache: Optional[Dict[str, Any]] = Field(default=None, description="Cache hit/miss info (Stage 7)")


class PlanRequest(BaseModel):
    """Request model for /plan endpoint."""
    
    intent: str = Field(..., description="Query intent to plan for")


class PlanResponse(BaseModel):
    """Response model for /plan endpoint."""
    
    steps: List[str] = Field(..., description="Planned execution steps")
    intent: str = Field(..., description="Query intent")
    materialized_config: Dict[str, Any] = Field(..., description="Materialized configuration")


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    
    status: str = Field(..., description="Service status")
    stage: str = Field(..., description="Implementation stage")
    components: Dict[str, bool] = Field(default_factory=dict, description="Component availability")


class MetricsResponse(BaseModel):
    """Response model for /metrics endpoint."""
    
    total_queries: int = Field(default=0, description="Total queries processed")
    accept_at_1: float = Field(default=0.0, description="Acceptance rate")
    avg_score: float = Field(default=0.0, description="Average verification score")
    repair_rate: float = Field(default=0.0, description="Repair usage rate")
    recent_results: List[Dict] = Field(default_factory=list, description="Recent evaluation results")


# ===== Stage 7: Async Ingestion & Caching Schemas =====


class IngestStartRequest(BaseModel):
    """Request model for /ingest/start endpoint."""
    
    corpus_dir: str = Field(..., description="Path to corpus directory")
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    use_cache: bool = Field(default=True, description="Whether to use embedding cache")


class IngestStartResponse(BaseModel):
    """Response model for /ingest/start endpoint."""
    
    job_id: str = Field(..., description="Job ID for tracking")
    message: str = Field(default="Ingestion job queued", description="Status message")


class JobStatusResponse(BaseModel):
    """Response model for /ingest/status endpoint."""
    
    id: str = Field(..., description="Job ID")
    state: str = Field(..., description="Job state: queued, running, succeeded, failed")
    progress: float = Field(..., description="Progress from 0.0 to 1.0")
    message: str = Field(..., description="Current status message")
    started_at: Optional[str] = Field(default=None, description="ISO timestamp when job started")
    finished_at: Optional[str] = Field(default=None, description="ISO timestamp when job finished")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Job-specific data")


class JobListResponse(BaseModel):
    """Response model for /ingest/list endpoint."""
    
    jobs: List[JobStatusResponse] = Field(..., description="List of jobs")


class CachePurgeRequest(BaseModel):
    """Request model for /cache/purge endpoint."""
    
    tables: List[str] = Field(..., description="Cache tables to purge: embeddings, retrieval, answers, sessions")


class CachePurgeResponse(BaseModel):
    """Response model for /cache/purge endpoint."""
    
    ok: bool = Field(..., description="Whether purge succeeded")
    purged: Dict[str, int] = Field(..., description="Number of entries purged per table")


class CacheInfo(BaseModel):
    """Cache hit/miss information."""
    
    answer_hit: bool = Field(default=False, description="Answer cache hit")
    retrieval_hit: bool = Field(default=False, description="Retrieval cache hit")
    embed_hits: int = Field(default=0, description="Number of embedding cache hits")
    embed_misses: int = Field(default=0, description="Number of embedding cache misses")
