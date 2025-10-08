"""
Pydantic schemas for evaluation harness.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class EvalItem(BaseModel):
    """Single evaluation item with query, intent, and expected outputs."""
    
    id: str
    query: str
    intent: str  # one of: definition, explanation, comparison, how_to, summarization, unknown
    expected_keywords: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class EvalDataset(BaseModel):
    """Collection of evaluation items."""
    
    name: str
    items: List[EvalItem]


class StepTiming(BaseModel):
    """Timing information for a single DAG step."""
    
    name: str
    ms: float


class EvalResult(BaseModel):
    """Results from evaluating a single query through the DAG."""
    
    id: str
    answer: str
    accepted: bool
    verify_score: float
    repair_used: bool
    steps: List[str]
    timings: List[StepTiming]
    retrieved_count: int
    pruned_chars: int
    context_chars: int
    meta: Dict[str, str] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """Aggregated report from evaluating a dataset."""
    
    dataset: str
    results: List[EvalResult]
    metrics: Dict[str, float]  # accept@1, avg_score, repair_rate, p50/p95_latency, etc.
