"""
Metrics computation for evaluation results.
"""

import statistics
from typing import List, Dict

from .schemas import EvalResult


def accept_at_1(results: List[EvalResult]) -> float:
    """
    Compute accept@1 metric: fraction of queries with accepted=True.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Acceptance rate between 0.0 and 1.0
    """
    if not results:
        return 0.0
    
    accepted_count = sum(1 for r in results if r.accepted)
    return accepted_count / len(results)


def avg_verify_score(results: List[EvalResult]) -> float:
    """
    Compute average verification score across all results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Mean verify score, or 0.0 if no results
    """
    if not results:
        return 0.0
    
    return statistics.mean(r.verify_score for r in results)


def repair_rate(results: List[EvalResult]) -> float:
    """
    Compute repair rate: fraction of queries that used the repair step.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Repair usage rate between 0.0 and 1.0
    """
    if not results:
        return 0.0
    
    repair_count = sum(1 for r in results if r.repair_used)
    return repair_count / len(results)


def latency_stats(results: List[EvalResult]) -> Dict[str, float]:
    """
    Compute latency statistics (p50, p95, mean, max) across all results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dict with keys: latency_p50_ms, latency_p95_ms, latency_mean_ms, latency_max_ms
    """
    if not results:
        return {
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_mean_ms": 0.0,
            "latency_max_ms": 0.0,
        }
    
    # Sum all step timings for each result to get total latency
    total_latencies = [sum(t.ms for t in r.timings) for r in results]
    
    return {
        "latency_p50_ms": statistics.median(total_latencies),
        "latency_p95_ms": statistics.quantiles(total_latencies, n=20)[18] if len(total_latencies) >= 20 else max(total_latencies),
        "latency_mean_ms": statistics.mean(total_latencies),
        "latency_max_ms": max(total_latencies),
    }


def avg_context_chars(results: List[EvalResult]) -> float:
    """
    Compute average context character count across all results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Mean context chars, or 0.0 if no results
    """
    if not results:
        return 0.0
    
    return statistics.mean(r.context_chars for r in results)


def avg_pruned_chars(results: List[EvalResult]) -> float:
    """
    Compute average pruned character count across all results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Mean pruned chars, or 0.0 if no results
    """
    if not results:
        return 0.0
    
    return statistics.mean(r.pruned_chars for r in results)


def avg_retrieved_count(results: List[EvalResult]) -> float:
    """
    Compute average number of retrieved documents across all results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Mean retrieved count, or 0.0 if no results
    """
    if not results:
        return 0.0
    
    return statistics.mean(r.retrieved_count for r in results)


def compute_metrics(results: List[EvalResult]) -> Dict[str, float]:
    """
    Compute all metrics for a list of evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dict with all computed metrics
    """
    metrics = {
        "accept_at_1": accept_at_1(results),
        "avg_score": avg_verify_score(results),
        "repair_rate": repair_rate(results),
        "avg_context_chars": avg_context_chars(results),
        "avg_pruned_chars": avg_pruned_chars(results),
        "avg_retrieved_count": avg_retrieved_count(results),
        "total_queries": float(len(results)),
    }
    
    # Add latency stats
    metrics.update(latency_stats(results))
    
    return metrics
