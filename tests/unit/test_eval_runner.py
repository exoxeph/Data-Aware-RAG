"""
Unit tests for evaluation runner.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_papers.eval.schemas import EvalDataset, EvalItem
from rag_papers.eval.runner import evaluate_dataset
from rag_papers.retrieval.router_dag import Stage4Config


@pytest.fixture
def sample_dataset():
    """Create sample evaluation dataset."""
    return EvalDataset(
        name="test_dataset",
        items=[
            EvalItem(
                id="q1",
                query="What is transfer learning?",
                intent="definition",
                expected_keywords=["pretrained", "model"],
            ),
            EvalItem(
                id="q2",
                query="How do neural networks work?",
                intent="explanation",
                expected_keywords=["layers", "nodes"],
            ),
        ],
    )


@pytest.fixture
def mock_docs():
    """Create mock document corpus."""
    return [
        "Document about transfer learning.",
        "Document about neural networks.",
    ]


@pytest.fixture
def mock_retriever():
    """Create mock retriever."""
    retriever = Mock()
    retriever.retrieve = Mock(return_value=[
        {"doc": "Doc 1", "score": 0.9},
        {"doc": "Doc 2", "score": 0.8},
    ])
    return retriever


@pytest.fixture
def mock_generator():
    """Create mock generator."""
    generator = Mock()
    generator.generate = Mock(return_value="Mock answer")
    return generator


@pytest.fixture
def mock_config():
    """Create mock config."""
    return Stage4Config()


@patch("rag_papers.eval.runner.run_plan")
def test_evaluate_dataset_basic(
    mock_run_plan,
    sample_dataset,
    mock_docs,
    mock_retriever,
    mock_generator,
    mock_config,
):
    """Test basic evaluation dataset execution."""
    # Setup mock run_plan to return deterministic results
    mock_run_plan.side_effect = [
        (
            "Answer 1",
            {
                "path": ["retrieve", "generate", "verify"],
                "verify_score": 0.85,
                "accepted": True,
                "timings": {"retrieve": 100, "generate": 200, "verify": 50},
                "retrieved_count": 10,
                "pruned_chars": 500,
                "context_chars": 4000,
            },
        ),
        (
            "Answer 2",
            {
                "path": ["retrieve", "generate", "verify", "repair"],
                "verify_score": 0.72,
                "accepted": True,
                "timings": {"retrieve": 110, "generate": 210, "verify": 55, "repair": 150},
                "retrieved_count": 12,
                "pruned_chars": 600,
                "context_chars": 3500,
            },
        ),
    ]
    
    # Run evaluation
    report = evaluate_dataset(
        dataset=sample_dataset,
        retriever=mock_retriever,
        generator=mock_generator,
        cfg=mock_config,
    )
    
    # Verify report structure
    assert report.dataset == "test_dataset"
    assert len(report.results) == 2
    assert "accept_at_1" in report.metrics
    assert "avg_score" in report.metrics
    
    # Verify run_plan was called correctly
    assert mock_run_plan.call_count == 2
    
    # Check first call
    first_call = mock_run_plan.call_args_list[0]
    assert first_call[1]["query"] == "What is transfer learning?"
    assert first_call[1]["intent"] == "definition"
    
    # Check results
    result1 = report.results[0]
    assert result1.id == "q1"
    assert result1.answer == "Answer 1"
    assert result1.accepted is True
    assert result1.verify_score == 0.85
    assert result1.repair_used is False
    
    result2 = report.results[1]
    assert result2.id == "q2"
    assert result2.answer == "Answer 2"
    assert result2.accepted is True
    assert result2.verify_score == 0.72
    assert result2.repair_used is True


@patch("rag_papers.eval.runner.run_plan")
def test_evaluate_dataset_with_run_id(
    mock_run_plan,
    sample_dataset,
    mock_docs,
    mock_retriever,
    mock_generator,
    mock_config,
):
    """Test evaluation with custom run_id."""
    mock_run_plan.return_value = (
        "Answer",
        {
            "path": ["retrieve", "generate", "verify"],
            "verify_score": 0.80,
            "accepted": True,
            "timings": {"retrieve": 100, "generate": 200, "verify": 50},
            "retrieved_count": 10,
            "pruned_chars": 0,
            "context_chars": 4000,
        },
    )
    
    custom_run_id = "test-run-123"
    
    report = evaluate_dataset(
        dataset=sample_dataset,
        retriever=mock_retriever,
        generator=mock_generator,
        cfg=mock_config,
        run_id=custom_run_id,
    )
    
    # Verify run_id is in metadata
    assert report.results[0].meta["run_id"] == custom_run_id


@patch("rag_papers.eval.runner.run_plan")
@patch("rag_papers.eval.runner.persist_duckdb")
def test_evaluate_dataset_with_duckdb(
    mock_persist,
    mock_run_plan,
    sample_dataset,
    mock_docs,
    mock_retriever,
    mock_generator,
    mock_config,
):
    """Test evaluation with DuckDB persistence."""
    mock_run_plan.return_value = (
        "Answer",
        {
            "path": ["retrieve", "generate", "verify"],
            "verify_score": 0.80,
            "accepted": True,
            "timings": {"retrieve": 100, "generate": 200, "verify": 50},
            "retrieved_count": 10,
            "pruned_chars": 0,
            "context_chars": 4000,
        },
    )
    
    duckdb_path = "test.duckdb"
    
    report = evaluate_dataset(
        dataset=sample_dataset,
        retriever=mock_retriever,
        generator=mock_generator,
        cfg=mock_config,
        duckdb_path=duckdb_path,
    )
    
    # Verify persist_duckdb was called
    assert mock_persist.called
    call_args = mock_persist.call_args
    assert call_args[0][0] == duckdb_path
    assert call_args[0][1] == "eval_results"
    assert len(call_args[0][2]) == 2  # 2 records


@patch("rag_papers.eval.runner.run_plan")
def test_evaluate_dataset_metrics(
    mock_run_plan,
    sample_dataset,
    mock_docs,
    mock_retriever,
    mock_generator,
    mock_config,
):
    """Test that metrics are computed correctly."""
    mock_run_plan.side_effect = [
        (
            "Answer 1",
            {
                "path": ["retrieve", "generate", "verify"],
                "verify_score": 0.90,
                "accepted": True,
                "timings": {"retrieve": 100, "generate": 200, "verify": 50},
                "retrieved_count": 10,
                "pruned_chars": 500,
                "context_chars": 4000,
            },
        ),
        (
            "Answer 2",
            {
                "path": ["retrieve", "generate", "verify"],
                "verify_score": 0.70,
                "accepted": False,
                "timings": {"retrieve": 110, "generate": 210, "verify": 55},
                "retrieved_count": 12,
                "pruned_chars": 600,
                "context_chars": 3500,
            },
        ),
    ]
    
    report = evaluate_dataset(
        dataset=sample_dataset,
        retriever=mock_retriever,
        generator=mock_generator,
        cfg=mock_config,
    )
    
    # Check metrics
    assert report.metrics["accept_at_1"] == 0.5  # 1/2 accepted
    assert report.metrics["avg_score"] == pytest.approx((0.90 + 0.70) / 2)
    assert report.metrics["repair_rate"] == 0.0  # No repairs
    assert report.metrics["total_queries"] == 2.0


@patch("rag_papers.eval.runner.run_plan")
def test_evaluate_dataset_extracts_timing(
    mock_run_plan,
    sample_dataset,
    mock_docs,
    mock_retriever,
    mock_generator,
    mock_config,
):
    """Test that step timings are extracted correctly."""
    mock_run_plan.return_value = (
        "Answer",
        {
            "path": ["retrieve", "rerank", "generate", "verify"],
            "verify_score": 0.85,
            "accepted": True,
            "timings": {
                "retrieve": 120.5,
                "rerank": 45.2,
                "generate": 205.8,
                "verify": 52.1,
            },
            "retrieved_count": 10,
            "pruned_chars": 0,
            "context_chars": 4000,
        },
    )
    
    report = evaluate_dataset(
        dataset=sample_dataset,
        retriever=mock_retriever,
        generator=mock_generator,
        cfg=mock_config,
    )
    
    # Check first result's timings
    result = report.results[0]
    assert len(result.timings) == 4
    assert result.timings[0].name == "retrieve"
    assert result.timings[0].ms == 120.5
    assert result.timings[1].name == "rerank"
    assert result.timings[1].ms == 45.2


@patch("rag_papers.eval.runner.run_plan")
@patch("rag_papers.eval.runner.persist_duckdb")
def test_evaluate_dataset_duckdb_error_handling(
    mock_persist,
    mock_run_plan,
    sample_dataset,
    mock_docs,
    mock_retriever,
    mock_generator,
    mock_config,
    capsys,
):
    """Test that DuckDB errors don't fail the evaluation."""
    mock_run_plan.return_value = (
        "Answer",
        {
            "path": ["retrieve", "generate", "verify"],
            "verify_score": 0.80,
            "accepted": True,
            "timings": {"retrieve": 100, "generate": 200, "verify": 50},
            "retrieved_count": 10,
            "pruned_chars": 0,
            "context_chars": 4000,
        },
    )
    
    # Make persist_duckdb raise an error
    mock_persist.side_effect = Exception("DuckDB error")
    
    # Should not raise, just print warning
    report = evaluate_dataset(
        dataset=sample_dataset,
        retriever=mock_retriever,
        generator=mock_generator,
        cfg=mock_config,
        duckdb_path="test.duckdb",
    )
    
    # Evaluation should still succeed
    assert len(report.results) == 2
    
    # Check that warning was printed
    captured = capsys.readouterr()
    assert "Warning" in captured.out or "DuckDB" in captured.out
