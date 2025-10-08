"""
Evaluation runner that executes the DAG pipeline on eval datasets.
"""

import time
from typing import Optional, Any

from ..retrieval.router_dag import run_plan, Stage4Config
from .schemas import EvalDataset, EvalResult, EvalReport, StepTiming
from .metrics import compute_metrics
from .telemetry import new_run_id, log_step, persist_duckdb


def evaluate_dataset(
    dataset: EvalDataset,
    retriever: Any,  # EnsembleRetriever
    generator: Any,  # BaseGenerator (Mock or real)
    cfg: Stage4Config,
    run_id: Optional[str] = None,
    duckdb_path: Optional[str] = None,
) -> EvalReport:
    """
    Evaluate a dataset by running each query through the DAG pipeline.
    
    Args:
        dataset: Evaluation dataset with queries
        retriever: EnsembleRetriever instance
        generator: Generator instance (MockGenerator for offline eval)
        cfg: Stage4Config with pipeline parameters
        run_id: Optional run ID; auto-generated if not provided
        duckdb_path: Optional DuckDB path for persisting telemetry
        
    Returns:
        EvalReport with results and aggregated metrics
        
    Example:
        >>> from rag_papers.eval import evaluate_dataset, load_dataset
        >>> from rag_papers.retrieval.router_dag import Stage4Config
        >>> from rag_papers.generation.generator import MockGenerator
        >>> 
        >>> dataset = load_dataset("qa_small")
        >>> cfg = Stage4Config()
        >>> generator = MockGenerator()
        >>> 
        >>> # retriever setup omitted for brevity
        >>> report = evaluate_dataset(
        ...     dataset=dataset,
        ...     retriever=retriever,
        ...     generator=generator,
        ...     cfg=cfg,
        ...     duckdb_path="data/telemetry.duckdb"
        ... )
        >>> 
        >>> print(f"Accept@1: {report.metrics['accept_at_1']:.2%}")
        >>> print(f"Repair rate: {report.metrics['repair_rate']:.2%}")
    """
    if run_id is None:
        run_id = new_run_id()
    
    results: list[EvalResult] = []
    duckdb_records: list[dict] = []
    
    for item in dataset.items:
        # Run the DAG pipeline
        start_time = time.perf_counter()
        
        answer, context = run_plan(
            query=item.query,
            intent=item.intent,
            retriever=retriever,
            generator=generator,
            cfg=cfg,
        )
        
        end_time = time.perf_counter()
        total_ms = (end_time - start_time) * 1000
        
        # Extract metadata from context
        # Path and timings are in meta sub-dict
        meta = context.get("meta", {})
        path = meta.get("path", [])
        step_timings_dict = meta.get("timings", {})
        
        # Verify score is in verify sub-dict
        verify_dict = context.get("verify", {})
        verify_score = float(verify_dict.get("score", 0.0))
        accepted = verify_score >= cfg.accept_threshold
        repair_used = "repair" in path
        
        # Extract step timings from context
        timings = [
            StepTiming(name=step, ms=step_timings_dict.get(step, 0.0) * 1000)  # Convert to ms
            for step in path
        ]
        
        # Extract retrieval and context stats
        retrieved_count = len(context.get("candidates", []))
        pruned_count = len(context.get("pruned_candidates", []))
        context_text = context.get("context_text", "")
        context_chars = len(context_text)
        
        # Estimate pruned chars (difference between candidates and pruned candidates)
        candidate_texts = [c.text if hasattr(c, 'text') else str(c) for c in context.get("candidates", [])]
        total_candidate_chars = sum(len(t) for t in candidate_texts)
        pruned_chars = max(0, total_candidate_chars - context_chars)
        
        # Build result
        result = EvalResult(
            id=item.id,
            answer=answer,
            accepted=accepted,
            verify_score=verify_score,
            repair_used=repair_used,
            steps=path,
            timings=timings,
            retrieved_count=retrieved_count,
            pruned_chars=pruned_chars,
            context_chars=context_chars,
            meta={
                "query": item.query,
                "intent": item.intent,
                "run_id": run_id,
            },
        )
        results.append(result)
        
        # Log each step
        for timing in timings:
            log_step(
                run_id=run_id,
                step_name=timing.name,
                ms=timing.ms,
                extra={
                    "query_id": item.id,
                    "query": item.query,
                    "intent": item.intent,
                },
            )
        
        # Prepare DuckDB record
        if duckdb_path:
            duckdb_records.append({
                "run_id": run_id,
                "query_id": item.id,
                "query": item.query,
                "intent": item.intent,
                "answer": answer,
                "accepted": accepted,
                "verify_score": verify_score,
                "repair_used": repair_used,
                "path": "|".join(path),
                "total_ms": total_ms,
                "retrieved_count": retrieved_count,
                "pruned_chars": pruned_chars,
                "context_chars": context_chars,
            })
    
    # Persist to DuckDB if requested
    if duckdb_path and duckdb_records:
        try:
            persist_duckdb(duckdb_path, "eval_results", duckdb_records)
        except Exception as e:
            # Log but don't fail the evaluation
            print(f"Warning: Failed to persist to DuckDB: {e}")
    
    # Compute aggregate metrics
    metrics = compute_metrics(results)
    
    return EvalReport(
        dataset=dataset.name,
        results=results,
        metrics=metrics,
    )
