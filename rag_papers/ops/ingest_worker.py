"""
Async ingestion worker - builds indexes with progress tracking.

Handles corpus loading, BM25 build, vector store build, and persistence.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .jobs import JobStatus, JobManager


async def run_ingest(
    job: "JobStatus",
    manager: "JobManager",
    corpus_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    use_cache: bool = True,
) -> None:
    """
    Run async corpus ingestion with progress updates.
    
    Steps:
        1. Scan files (10%)
        2. Build BM25 index (40%)
        3. Build vector store (80%)
        4. Save indexes (100%)
    
    Args:
        job: JobStatus object to update
        manager: JobManager for progress updates
        corpus_dir: Directory containing corpus files
        model_name: Embedding model name
        use_cache: Whether to use embedding cache
    
    Updates job.payload with:
        - corpus_dir: str
        - corpus_id: str
        - doc_count: int
        - meta: IndexMeta dict (on success)
        - error: str (on failure)
    """
    from rag_papers.index.build_bm25 import build_bm25
    from rag_papers.index.build_vector_store import build_vector_store
    from rag_papers.persist import (
        get_corpus_paths,
        ensure_dirs,
        save_indexes,
        IndexMeta,
        KVStore,
        CachingEncoder,
    )
    from sentence_transformers import SentenceTransformer
    
    corpus_dir = Path(corpus_dir)
    
    # Update payload
    job.payload["corpus_dir"] = str(corpus_dir)
    
    # Step 1: Scan files (10%)
    manager.update_progress(job.id, 0.1, "Scanning corpus directory...")
    await asyncio.sleep(0.1)  # Allow event loop to process
    
    # Get corpus paths
    cp = get_corpus_paths(corpus_dir)
    ensure_dirs(cp)
    
    job.payload["corpus_id"] = cp.idx_dir.name
    
    # Load texts
    texts = []
    for ext in ["*.txt", "*.md"]:
        for path in corpus_dir.rglob(ext):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    
    if not texts:
        raise ValueError(f"No .txt or .md files found in {corpus_dir}")
    
    job.payload["doc_count"] = len(texts)
    manager.update_progress(job.id, 0.15, f"Found {len(texts)} documents")
    
    # Step 2: Build BM25 (40%)
    manager.update_progress(job.id, 0.2, "Building BM25 index...")
    
    # Run in thread pool (CPU-bound)
    bm25 = await asyncio.to_thread(build_bm25, texts)
    
    manager.update_progress(job.id, 0.4, "BM25 index complete")
    
    # Step 3: Build vector store (80%)
    manager.update_progress(job.id, 0.45, f"Loading embedding model: {model_name}...")
    
    # Load model
    model = await asyncio.to_thread(SentenceTransformer, model_name)
    
    # Wrap with cache if enabled
    if use_cache:
        kv = KVStore(cp.cache_db_path)
        encoder = CachingEncoder(model, kv, model_name)
        manager.update_progress(job.id, 0.5, "Building vectors (with cache)...")
    else:
        encoder = model
        manager.update_progress(job.id, 0.5, "Building vectors...")
    
    # Encode
    vectors = await asyncio.to_thread(
        encoder.encode,
        texts,
        batch_size=32,
        show_progress_bar=False,
    )
    
    # Get cache stats if available
    if use_cache and hasattr(encoder, "get_stats"):
        stats = encoder.get_stats()
        job.payload["embed_cache"] = stats
        kv.close()
    
    manager.update_progress(job.id, 0.8, f"Encoded {len(vectors)} documents")
    
    # Step 4: Save indexes (100%)
    manager.update_progress(job.id, 0.85, "Saving indexes to disk...")
    
    meta = IndexMeta(
        corpus_id=cp.idx_dir.name,
        doc_count=len(texts),
        built_at=datetime.utcnow().isoformat(),
        bm25_tokenizer="default",
        embed_model_name=model_name,
        dim=vectors.shape[1],
    )
    
    await asyncio.to_thread(
        save_indexes,
        cp.idx_dir,
        bm25,
        vectors,
        texts,
        meta,
    )
    
    manager.update_progress(job.id, 1.0, "Ingestion complete")
    
    # Store metadata in payload
    job.payload["meta"] = meta.to_dict()
    job.payload["index_dir"] = str(cp.idx_dir)
