"""Unit test for query routing and planning."""

from rag_papers.retrieval.router_dag import plan_query, Plan


def test_plan_query_text_only():
    """Test query planning for text-only queries."""
    query = "What are the main findings of this research?"
    plan = plan_query(query)
    
    assert plan.use_text is True
    assert plan.use_tables is False
    assert plan.use_figures is False


def test_plan_query_with_tables():
    """Test query planning that should include tables."""
    query = "Compare AUC scores between different models on CIFAR-10"
    plan = plan_query(query)
    
    assert plan.use_text is True
    assert plan.use_tables is True  # Should detect 'auc'
    assert plan.use_figures is False


def test_plan_query_with_figures():
    """Test query planning that should include figures."""
    query = "Show the architecture diagram and explain the encoder-decoder framework"
    plan = plan_query(query)
    
    assert plan.use_text is True
    assert plan.use_tables is False
    assert plan.use_figures is True  # Should detect 'architecture', 'encoder', 'decoder'


def test_plan_query_multimodal():
    """Test query planning that requires all modalities."""
    query = "Compare the AUC results and show the model architecture pipeline with confidence intervals"
    plan = plan_query(query)
    
    assert plan.use_text is True
    assert plan.use_tables is True  # Should detect 'auc', 'confidence interval'
    assert plan.use_figures is True  # Should detect 'architecture', 'pipeline'


def test_plan_query_statistical_terms():
    """Test detection of statistical terms that suggest table usage."""
    query = "What is the 95% confidence interval with p<0.05 significance?"
    plan = plan_query(query)
    
    assert plan.use_text is True
    assert plan.use_tables is True  # Should detect '95%', 'p<'
    assert plan.use_figures is False


def test_plan_dataclass():
    """Test the Plan dataclass default values."""
    plan = Plan()
    assert plan.use_text is True
    assert plan.use_tables is False
    assert plan.use_figures is False
    
    plan_custom = Plan(use_text=False, use_tables=True, use_figures=True)
    assert plan_custom.use_text is False
    assert plan_custom.use_tables is True
    assert plan_custom.use_figures is True