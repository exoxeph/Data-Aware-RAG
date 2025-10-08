"""
Evaluation harness for Stage 4 DAG pipeline.

Provides offline eval datasets, metrics computation, telemetry, and reporting.
"""

from .schemas import (
    EvalItem,
    EvalDataset,
    StepTiming,
    EvalResult,
    EvalReport,
)
from .datasets import load_dataset
from .metrics import (
    accept_at_1,
    avg_verify_score,
    repair_rate,
    latency_stats,
    avg_context_chars,
    avg_pruned_chars,
    compute_metrics,
)
from .runner import evaluate_dataset
from .reporter import save_report, save_html

__all__ = [
    "EvalItem",
    "EvalDataset",
    "StepTiming",
    "EvalResult",
    "EvalReport",
    "load_dataset",
    "accept_at_1",
    "avg_verify_score",
    "repair_rate",
    "latency_stats",
    "avg_context_chars",
    "avg_pruned_chars",
    "compute_metrics",
    "evaluate_dataset",
    "save_report",
    "save_html",
]
