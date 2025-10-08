"""Main FastAPI application and server startup."""

from fastapi import FastAPI, HTTPException, Depends
from typing import Optional, Any
from pathlib import Path
import uvicorn
import time

from .schemas import (
    AskRequest,
    AskResponse,
    PlanRequest,
    PlanResponse,
    HealthResponse,
    MetricsResponse,
    IngestStartRequest,
    IngestStartResponse,
    JobStatusResponse,
    JobListResponse,
    CachePurgeRequest,
    CachePurgeResponse,
)
from ..retrieval.router_dag import (
    run_plan,
    plan_for_intent,
    _materialize,
    Stage4Config,
)
from ..generation.generator import MockGenerator

# Stage 7: Import persistence and job management
from ..persist import (
    get_corpus_paths,
    KVStore,
    AnswerKey,
    AnswerValue,
    get_answer,
    set_answer,
    corpus_id_from_dir,
)
from ..ops import JobManager, run_ingest

# Stage 8: Import chat API
from .chat import router as chat_router

# Stage 10: Import memory API
# Temporarily disabled until memory module is fully implemented
# from .memory import router as memory_router

app = FastAPI(
    title="RAG Papers API",
    description="Data-Aware RAG for PDF Research Papers with Multi-Turn Chat",
    version="1.2.0",
)

# Include chat router
app.include_router(chat_router, prefix="/api", tags=["chat"])

# Include memory router
# Temporarily disabled
# app.include_router(memory_router, prefix="/api", tags=["memory"])

# Global state for dependencies (initialized on startup)
_docs: Optional[list[str]] = None
_retriever: Optional[Any] = None
_generator: Optional[Any] = None
_config: Optional[Stage4Config] = None

# Stage 7: Global job manager and cache
_job_manager: Optional[JobManager] = None
_kv_store: Optional[KVStore] = None
_corpus_id: str = "empty"


def get_docs():
    """Dependency to get document corpus."""
    if _docs is None:
        raise HTTPException(status_code=503, detail="Document corpus not initialized")
    return _docs


def get_retriever():
    """Dependency to get retriever."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return _retriever


def get_generator():
    """Dependency to get generator."""
    if _generator is None:
        # Fallback to MockGenerator if not initialized
        return MockGenerator()
    return _generator


def get_config():
    """Dependency to get config."""
    if _config is None:
        return Stage4Config()
    return _config


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global _docs, _retriever, _generator, _config, _job_manager, _kv_store, _corpus_id
    
    # Initialize with minimal setup
    # In production, this would load actual docs and build retriever
    _docs = [
        "Sample document about machine learning and neural networks.",
        "Transfer learning involves using pretrained models.",
        "Fine-tuning adapts models to specific tasks.",
    ]
    
    # Use MockGenerator by default
    _generator = MockGenerator()
    
    # Default config
    _config = Stage4Config()
    
    # Stage 7: Initialize job manager and cache
    jobs_dir = Path("data/jobs")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    _job_manager = JobManager(jobs_dir / "jobs.jsonl")
    
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _kv_store = KVStore(cache_dir / "cache.db")
    
    _corpus_id = "default"
    
    # Note: Retriever initialization would go here in production
    # For now, we'll handle missing retriever gracefully in endpoints


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _job_manager, _kv_store
    
    if _job_manager:
        _job_manager.shutdown()
    
    if _kv_store:
        _kv_store.close()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Papers API is running",
        "version": "1.0.0",
        "stage": "4-ready",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        stage="4-ready",
        components={
            "docs": _docs is not None,
            "retriever": _retriever is not None,
            "generator": _generator is not None,
            "config": _config is not None,
        },
    )


@app.post("/plan", response_model=PlanResponse)
async def plan(
    request: PlanRequest,
    config: Stage4Config = Depends(get_config),
):
    """
    Get execution plan for a given intent.
    
    Returns the planned steps and materialized configuration.
    """
    # Get plan for intent
    steps = plan_for_intent(request.intent)
    step_names = [step["name"] for step in steps]
    
    # Materialize config
    materialized = _materialize(
        cfg=config,
        intent=request.intent,
        overrides={},
    )
    
    return PlanResponse(
        steps=step_names,
        intent=request.intent,
        materialized_config=materialized,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    docs: list[str] = Depends(get_docs),
    retriever: Any = Depends(get_retriever),
    generator: Any = Depends(get_generator),
    base_config: Stage4Config = Depends(get_config),
):
    """
    Ask a question and get an answer through the DAG pipeline.
    
    Stage 7: Supports caching with use_cache and refresh params.
    Uses MockGenerator by default for offline operation.
    """
    global _kv_store, _corpus_id
    
    # Build config overrides from request
    overrides = {}
    if request.bm25_weight is not None:
        overrides["bm25_weight"] = request.bm25_weight
    if request.vector_weight is not None:
        overrides["vector_weight"] = request.vector_weight
    if request.temperature is not None:
        overrides["temperature"] = request.temperature
    if request.top_k is not None:
        overrides["top_k"] = request.top_k
    if request.rerank_top_k is not None:
        overrides["rerank_top_k"] = request.rerank_top_k
    if request.accept_threshold is not None:
        overrides["accept_threshold"] = request.accept_threshold
    
    # Create config with overrides (map to Stage4Config parameters)
    cfg = Stage4Config(
        bm25_weight_init=overrides.get("bm25_weight", base_config.bm25_weight_init),
        vector_weight_init=overrides.get("vector_weight", base_config.vector_weight_init),
        temperature_init=overrides.get("temperature", base_config.temperature_init),
        top_k_first=overrides.get("top_k", base_config.top_k_first),
        rerank_top_k=overrides.get("rerank_top_k", base_config.rerank_top_k),
        prune_min_overlap=base_config.prune_min_overlap,
        prune_max_chars=base_config.prune_max_chars,
        context_max_chars=base_config.context_max_chars,
        accept_threshold=overrides.get("accept_threshold", base_config.accept_threshold),
        bm25_weight_on_repair=base_config.bm25_weight_on_repair,
        vector_weight_on_repair=base_config.vector_weight_on_repair,
        temperature_on_repair=base_config.temperature_on_repair,
        max_repairs=base_config.max_repairs,
    )
    
    # Handle missing retriever gracefully
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. Service requires retriever setup.",
        )
    
    # Stage 7: Check answer cache
    cache_info = {
        "answer_hit": False,
        "retrieval_hit": False,
        "embed_hits": 0,
        "embed_misses": 0,
    }
    
    use_cache = request.use_cache and not request.refresh
    
    if use_cache and _kv_store:
        # Build cache key
        answer_key = AnswerKey(
            query=request.query,
            intent=request.intent,
            corpus_id=_corpus_id,
            model=generator.__class__.__name__,
            cfg={
                "bm25_weight": cfg.bm25_weight_init,
                "vector_weight": cfg.vector_weight_init,
                "temperature": cfg.temperature_init,
                "top_k": cfg.top_k_first,
                "rerank_top_k": cfg.rerank_top_k,
                "prune_min_overlap": cfg.prune_min_overlap,
                "prune_max_chars": cfg.prune_max_chars,
                "context_max_chars": cfg.context_max_chars,
                "accept_threshold": cfg.accept_threshold,
            },
        )
        
        # Try to get from cache
        cached_answer = get_answer(answer_key, _kv_store)
        
        if cached_answer:
            cache_info["answer_hit"] = True
            
            return AskResponse(
                answer=cached_answer.answer,
                score=cached_answer.score,
                accepted=cached_answer.accepted,
                path=cached_answer.path,
                timings=cached_answer.timings,
                sources=[],
                retrieved_count=0,
                context_chars=cached_answer.context_chars,
                repair_used="repair" in cached_answer.path,
                cache=cache_info,
            )
    
    start_time = time.perf_counter()
    
    try:
        # Run the DAG pipeline
        answer, context = run_plan(
            query=request.query,
            intent=request.intent,
            docs=docs,
            retriever=retriever,
            generator=generator,
            cfg=cfg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    
    end_time = time.perf_counter()
    total_ms = (end_time - start_time) * 1000
    
    # Extract metadata from context
    path = context.get("meta", {}).get("path", context.get("path", []))
    
    # Get verify score from nested structure
    verify_dict = context.get("verify", {})
    verify_score = verify_dict.get("score", context.get("verify_score", 0.0))
    
    accepted = context.get("accepted", False)
    repair_used = "repair" in path
    retrieved_count = context.get("retrieved_count", len(context.get("candidates", [])))
    context_chars = len(context.get("context_text", ""))
    
    # Build timings dict
    timings_dict = context.get("meta", {}).get("timings", context.get("timings", {}))
    timings = {step: timings_dict.get(step, 0.0) for step in path}
    
    # Extract sources if available
    sources = context.get("sources", [])
    
    # Stage 7: Cache the answer
    if use_cache and _kv_store:
        answer_value = AnswerValue(
            answer=answer,
            score=verify_score,
            accepted=accepted,
            context_chars=context_chars,
            path=path,
            timings=timings,
        )
        set_answer(answer_key, answer_value, _kv_store)
    
    return AskResponse(
        answer=answer,
        score=verify_score,
        accepted=accepted,
        path=path,
        timings=timings,
        sources=sources,
        retrieved_count=retrieved_count,
        context_chars=context_chars,
        repair_used=repair_used,
        cache=cache_info,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """
    Get evaluation metrics.
    
    Returns metrics from recent evaluation runs if available.
    In production, this would query DuckDB for stored results.
    """
    # Placeholder implementation
    # In production, this would query eval_results table from DuckDB
    return MetricsResponse(
        total_queries=0,
        accept_at_1=0.0,
        avg_score=0.0,
        repair_rate=0.0,
        recent_results=[],
    )


# ===== Stage 7: Async Ingestion Endpoints =====


@app.post("/ingest/start", response_model=IngestStartResponse)
async def ingest_start(request: IngestStartRequest):
    """
    Start async corpus ingestion job.
    
    Creates a background job that:
    1. Scans corpus directory
    2. Builds BM25 index
    3. Builds vector store (with caching)
    4. Saves indexes to disk
    """
    global _job_manager
    
    if not _job_manager:
        raise HTTPException(status_code=503, detail="Job manager not initialized")
    
    corpus_dir = Path(request.corpus_dir)
    
    if not corpus_dir.exists():
        raise HTTPException(status_code=404, detail=f"Corpus directory not found: {corpus_dir}")
    
    # Create worker function
    async def worker(job, manager):
        await run_ingest(
            job=job,
            manager=manager,
            corpus_dir=corpus_dir,
            model_name=request.model_name,
            use_cache=request.use_cache,
        )
    
    # Submit job
    job_id = _job_manager.submit_job(
        job_type="ingest",
        worker_fn=worker,
        payload={
            "corpus_dir": str(corpus_dir),
            "model_name": request.model_name,
            "use_cache": request.use_cache,
        },
    )
    
    return IngestStartResponse(
        job_id=job_id,
        message="Ingestion job queued successfully",
    )


@app.get("/ingest/status/{job_id}", response_model=JobStatusResponse)
async def ingest_status(job_id: str):
    """Get status of an ingestion job."""
    global _job_manager
    
    if not _job_manager:
        raise HTTPException(status_code=503, detail="Job manager not initialized")
    
    job = _job_manager.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    return JobStatusResponse(
        id=job.id,
        state=job.state,
        progress=job.progress,
        message=job.message,
        started_at=job.started_at,
        finished_at=job.finished_at,
        payload=job.payload,
    )


@app.get("/ingest/list", response_model=JobListResponse)
async def ingest_list(state: Optional[str] = None):
    """
    List all ingestion jobs.
    
    Query params:
        state: Filter by state (queued, running, succeeded, failed)
    """
    global _job_manager
    
    if not _job_manager:
        raise HTTPException(status_code=503, detail="Job manager not initialized")
    
    jobs = _job_manager.list(state=state)
    
    return JobListResponse(
        jobs=[
            JobStatusResponse(
                id=j.id,
                state=j.state,
                progress=j.progress,
                message=j.message,
                started_at=j.started_at,
                finished_at=j.finished_at,
                payload=j.payload,
            )
            for j in jobs
        ]
    )


# ===== Stage 7: Cache Management Endpoints =====


@app.post("/cache/purge", response_model=CachePurgeResponse)
async def cache_purge(request: CachePurgeRequest):
    """
    Purge cache tables.
    
    Tables: embeddings, retrieval, answers, sessions
    """
    global _kv_store
    
    if not _kv_store:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    valid_tables = {"embeddings", "retrieval", "answers", "sessions"}
    invalid = set(request.tables) - valid_tables
    
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid table names: {invalid}. Valid: {valid_tables}",
        )
    
    purged = {}
    for table in request.tables:
        count = _kv_store.purge_table(table)
        purged[table] = count
    
    # Vacuum to reclaim space
    _kv_store.vacuum()
    
    return CachePurgeResponse(
        ok=True,
        purged=purged,
    )


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics for all tables."""
    global _kv_store
    
    if not _kv_store:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    tables = ["embeddings", "retrieval", "answers", "sessions"]
    stats = {}
    
    for table in tables:
        stats[table] = _kv_store.stats(table)
    
    return {"tables": stats}



def run():
    """Run the development server."""
    uvicorn.run("rag_papers.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()