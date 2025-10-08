"""
Unit tests for evaluation metrics.
"""

import pytest
from rag_papers.eval.schemas import EvalResult, StepTiming
from rag_papers.eval.metrics import (
    accept_at_1,
    avg_verify_score,
    repair_rate,
    latency_stats,
    avg_context_chars,
    avg_pruned_chars,
    avg_retrieved_count,
    compute_metrics,
)


@pytest.fixture
def sample_results():
    """Create sample evaluation results for testing."""
    return [
        EvalResult(
            id="q1",
            answer="Answer 1",
            accepted=True,
            verify_score=0.85,
            repair_used=False,
            steps=["retrieve", "rerank", "generate", "verify"],
            timings=[
                StepTiming(name="retrieve", ms=100.0),
                StepTiming(name="rerank", ms=50.0),
                StepTiming(name="generate", ms=200.0),
                StepTiming(name="verify", ms=50.0),
            ],
            retrieved_count=12,
            pruned_chars=500,
            context_chars=4000,
            meta={"query": "test query 1", "intent": "definition"},
        ),
        EvalResult(
            id="q2",
            answer="Answer 2",
            accepted=False,
            verify_score=0.65,
            repair_used=True,
            steps=["retrieve", "rerank", "generate", "verify", "repair"],
            timings=[
                StepTiming(name="retrieve", ms=110.0),
                StepTiming(name="rerank", ms=55.0),
                StepTiming(name="generate", ms=210.0),
                StepTiming(name="verify", ms=45.0),
                StepTiming(name="repair", ms=150.0),
            ],
            retrieved_count=12,
            pruned_chars=600,
            context_chars=3500,
            meta={"query": "test query 2", "intent": "explanation"},
        ),
        EvalResult(
            id="q3",
            answer="Answer 3",
            accepted=True,
            verify_score=0.92,
            repair_used=False,
            steps=["retrieve", "rerank", "prune", "generate", "verify"],
            timings=[
                StepTiming(name="retrieve", ms=95.0),
                StepTiming(name="rerank", ms=48.0),
                StepTiming(name="prune", ms=30.0),
                StepTiming(name="generate", ms=190.0),
                StepTiming(name="verify", ms=52.0),
            ],
            retrieved_count=8,
            pruned_chars=800,
            context_chars=3000,
            meta={"query": "test query 3", "intent": "comparison"},
        ),
    ]


def test_accept_at_1(sample_results):
    """Test accept@1 metric calculation."""
    result = accept_at_1(sample_results)
    assert result == 2/3  # 2 out of 3 accepted
    assert 0.0 <= result <= 1.0


def test_accept_at_1_empty():
    """Test accept@1 with empty results."""
    assert accept_at_1([]) == 0.0


def test_accept_at_1_all_accepted():
    """Test accept@1 when all results are accepted."""
    results = [
        EvalResult(
            id="q1",
            answer="A",
            accepted=True,
            verify_score=0.9,
            repair_used=False,
            steps=[],
            timings=[],
            retrieved_count=10,
            pruned_chars=0,
            context_chars=0,
        ),
        EvalResult(
            id="q2",
            answer="B",
            accepted=True,
            verify_score=0.8,
            repair_used=False,
            steps=[],
            timings=[],
            retrieved_count=10,
            pruned_chars=0,
            context_chars=0,
        ),
    ]
    assert accept_at_1(results) == 1.0


def test_avg_verify_score(sample_results):
    """Test average verification score calculation."""
    result = avg_verify_score(sample_results)
    expected = (0.85 + 0.65 + 0.92) / 3
    assert result == pytest.approx(expected)


def test_avg_verify_score_empty():
    """Test avg verify score with empty results."""
    assert avg_verify_score([]) == 0.0


def test_repair_rate(sample_results):
    """Test repair rate calculation."""
    result = repair_rate(sample_results)
    assert result == 1/3  # 1 out of 3 used repair


def test_repair_rate_empty():
    """Test repair rate with empty results."""
    assert repair_rate([]) == 0.0


def test_repair_rate_none():
    """Test repair rate when no repairs used."""
    results = [
        EvalResult(
            id="q1",
            answer="A",
            accepted=True,
            verify_score=0.9,
            repair_used=False,
            steps=[],
            timings=[],
            retrieved_count=10,
            pruned_chars=0,
            context_chars=0,
        ),
    ]
    assert repair_rate(results) == 0.0


def test_latency_stats(sample_results):
    """Test latency statistics calculation."""
    stats = latency_stats(sample_results)
    
    # Total latencies: 400ms, 570ms, 415ms
    assert "latency_p50_ms" in stats
    assert "latency_p95_ms" in stats
    assert "latency_mean_ms" in stats
    assert "latency_max_ms" in stats
    
    assert stats["latency_p50_ms"] == 415.0  # median
    assert stats["latency_mean_ms"] == pytest.approx((400 + 570 + 415) / 3)
    assert stats["latency_max_ms"] == 570.0


def test_latency_stats_empty():
    """Test latency stats with empty results."""
    stats = latency_stats([])
    assert stats["latency_p50_ms"] == 0.0
    assert stats["latency_p95_ms"] == 0.0
    assert stats["latency_mean_ms"] == 0.0
    assert stats["latency_max_ms"] == 0.0


def test_avg_context_chars(sample_results):
    """Test average context chars calculation."""
    result = avg_context_chars(sample_results)
    expected = (4000 + 3500 + 3000) / 3
    assert result == pytest.approx(expected)


def test_avg_context_chars_empty():
    """Test avg context chars with empty results."""
    assert avg_context_chars([]) == 0.0


def test_avg_pruned_chars(sample_results):
    """Test average pruned chars calculation."""
    result = avg_pruned_chars(sample_results)
    expected = (500 + 600 + 800) / 3
    assert result == pytest.approx(expected)


def test_avg_pruned_chars_empty():
    """Test avg pruned chars with empty results."""
    assert avg_pruned_chars([]) == 0.0


def test_avg_retrieved_count(sample_results):
    """Test average retrieved count calculation."""
    result = avg_retrieved_count(sample_results)
    expected = (12 + 12 + 8) / 3
    assert result == pytest.approx(expected)


def test_avg_retrieved_count_empty():
    """Test avg retrieved count with empty results."""
    assert avg_retrieved_count([]) == 0.0


def test_compute_metrics(sample_results):
    """Test comprehensive metrics computation."""
    metrics = compute_metrics(sample_results)
    
    # Check all expected keys
    assert "accept_at_1" in metrics
    assert "avg_score" in metrics
    assert "repair_rate" in metrics
    assert "avg_context_chars" in metrics
    assert "avg_pruned_chars" in metrics
    assert "avg_retrieved_count" in metrics
    assert "total_queries" in metrics
    assert "latency_p50_ms" in metrics
    assert "latency_p95_ms" in metrics
    assert "latency_mean_ms" in metrics
    assert "latency_max_ms" in metrics
    
    # Check values
    assert metrics["accept_at_1"] == 2/3
    assert metrics["avg_score"] == pytest.approx((0.85 + 0.65 + 0.92) / 3)
    assert metrics["repair_rate"] == 1/3
    assert metrics["total_queries"] == 3.0


def test_compute_metrics_empty():
    """Test compute metrics with empty results."""
    metrics = compute_metrics([])
    
    assert metrics["accept_at_1"] == 0.0
    assert metrics["avg_score"] == 0.0
    assert metrics["repair_rate"] == 0.0
    assert metrics["total_queries"] == 0.0
