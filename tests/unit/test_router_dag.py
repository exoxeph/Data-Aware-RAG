"""Unit tests for DAG-based orchestration layer."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

from rag_papers.retrieval.router_dag import (
    plan_for_intent,
    exec_retrieve,
    exec_rerank,
    exec_prune,
    exec_contextualize,
    exec_generate,
    exec_verify,
    exec_repair,
    run_plan,
    build_prompt_for_intent,
    score_answer,
    _materialize,
    Candidate,
    Context,
    Stage4Config,
    Plan,
    PlanStep
)


class TestPlanForIntent:
    """Test intent-based plan generation."""
    
    def test_definition_intent(self):
        """Test plan for definition queries."""
        plan = plan_for_intent("definition")
        assert len(plan["steps"]) > 0
        step_names = [s["name"] for s in plan["steps"]]
        assert "retrieve" in step_names
        assert "rerank" in step_names
        assert "prune" in step_names
        assert "generate" in step_names
        assert "verify" in step_names
    
    def test_explanation_intent(self):
        """Test plan for explanation queries."""
        plan = plan_for_intent("explanation")
        assert len(plan["steps"]) > 0
        step_names = [s["name"] for s in plan["steps"]]
        assert "retrieve" in step_names
        assert "generate" in step_names
    
    def test_comparison_intent(self):
        """Test plan for comparison queries."""
        plan = plan_for_intent("comparison")
        assert len(plan["steps"]) > 0
        step_names = [s["name"] for s in plan["steps"]]
        assert "retrieve" in step_names
        assert "contextualize" in step_names
        # Check that comparison mode is specified
        contextualize_step = [s for s in plan["steps"] if s["name"] == "contextualize"][0]
        assert "mode" in contextualize_step["params"]
    
    def test_how_to_intent(self):
        """Test plan for how-to queries."""
        plan = plan_for_intent("how_to")
        assert len(plan["steps"]) > 0
        step_names = [s["name"] for s in plan["steps"]]
        assert "retrieve" in step_names
        assert "generate" in step_names
    
    def test_unknown_intent_fallback(self):
        """Test fallback plan for unknown intents."""
        plan = plan_for_intent("random_unknown_intent")
        assert len(plan["steps"]) > 0
        step_names = [s["name"] for s in plan["steps"]]
        assert "retrieve" in step_names


class TestMaterialize:
    """Test configuration materialization."""
    
    def test_materialize_simple_values(self):
        """Test materialization of simple values."""
        cfg = Stage4Config(top_k_first=20, rerank_top_k=10)
        params = {"k": 5, "temperature": 0.3}
        result = _materialize(params, cfg)
        assert result["k"] == 5
        assert result["temperature"] == 0.3
    
    def test_materialize_config_placeholders(self):
        """Test materialization of cfg placeholders."""
        cfg = Stage4Config(top_k_first=20, rerank_top_k=10)
        params = {"k": "cfg.top_k_first", "top_k": "cfg.rerank_top_k"}
        result = _materialize(params, cfg)
        assert result["k"] == 20
        assert result["top_k"] == 10
    
    def test_materialize_mixed_params(self):
        """Test materialization of mixed parameters."""
        cfg = Stage4Config(prune_min_overlap=2)
        params = {
            "min_overlap": "cfg.prune_min_overlap",
            "max_chars": 1000,
            "mode": "comparison"
        }
        result = _materialize(params, cfg)
        assert result["min_overlap"] == 2
        assert result["max_chars"] == 1000
        assert result["mode"] == "comparison"


class TestExecutors:
    """Test individual step executors."""
    
    def test_exec_retrieve(self):
        """Test retrieval executor."""
        mock_retriever = Mock()
        mock_retriever.search.return_value = [
            ("text1", 0.9, {"source": "doc1"}),
            ("text2", 0.8, {"source": "doc2"})
        ]
        
        ctx: Context = {"query": "test query", "intent": "definition", "meta": {}}
        cfg = Stage4Config()
        
        result = exec_retrieve(ctx, mock_retriever, cfg, k=5)
        
        assert "candidates" in result
        assert len(result["candidates"]) == 2
        assert result["candidates"][0].text == "text1"
        assert result["candidates"][0].score == 0.9
        mock_retriever.search.assert_called_once()
    
    def test_exec_rerank(self):
        """Test re-ranking executor."""
        ctx: Context = {
            "query": "test",
            "intent": "definition",
            "meta": {},
            "candidates": [
                Candidate(text="low", score=0.3),
                Candidate(text="high", score=0.9),
                Candidate(text="medium", score=0.6)
            ]
        }
        
        result = exec_rerank(ctx, top_k=2)
        
        assert len(result["candidates"]) == 2
        assert result["candidates"][0].score == 0.9  # highest first
        assert result["candidates"][1].score == 0.6
    
    def test_exec_prune(self):
        """Test pruning executor."""
        ctx: Context = {
            "query": "machine learning",
            "intent": "definition",
            "meta": {},
            "candidates": [
                Candidate(text="Machine learning is a field of AI. It involves algorithms.", score=0.9),
                Candidate(text="This is about gardening and plants.", score=0.5)
            ]
        }
        
        result = exec_prune(ctx, min_overlap=1, max_chars=500)
        
        assert "pruned_candidates" in result
        # Should keep relevant content
        assert len(result["pruned_candidates"]) >= 1
    
    def test_exec_contextualize(self):
        """Test contextualization executor."""
        ctx: Context = {
            "query": "test query",
            "intent": "definition",
            "meta": {},
            "candidates": [
                Candidate(text="First document.", score=0.9),
                Candidate(text="Second document.", score=0.8)
            ]
        }
        
        result = exec_contextualize(ctx, max_chars=1000)
        
        assert "context_text" in result
        assert isinstance(result["context_text"], str)
        assert len(result["context_text"]) > 0
    
    def test_exec_generate(self):
        """Test generation executor."""
        mock_generator = Mock()
        mock_output = Mock()
        mock_output.text = "Generated answer"
        mock_generator.generate.return_value = mock_output
        
        ctx: Context = {
            "query": "test query",
            "intent": "definition",
            "meta": {},
            "context_text": "Context for generation"
        }
        
        result = exec_generate(ctx, mock_generator, template="explain", temperature=0.2)
        
        assert "answer" in result
        assert result["answer"] == "Generated answer"
        mock_generator.generate.assert_called_once()
    
    def test_exec_verify(self):
        """Test verification executor."""
        ctx: Context = {
            "query": "test query",
            "intent": "definition",
            "meta": {},
            "answer": "This is a test answer",
            "context_text": "Context text here"
        }
        
        result = exec_verify(ctx)
        
        assert "verify" in result
        assert "score" in result["verify"]
        assert isinstance(result["verify"]["score"], float)
    
    def test_exec_repair_accepted_answer(self):
        """Test repair executor with already-accepted answer."""
        mock_retriever = Mock()
        mock_generator = Mock()
        
        ctx: Context = {
            "query": "test",
            "intent": "definition",
            "meta": {},
            "answer": "Good answer",
            "verify": {"score": 0.85}  # Above threshold
        }
        cfg = Stage4Config(accept_threshold=0.72)
        
        result = exec_repair(ctx, mock_retriever, mock_generator, cfg)
        
        # Should not trigger repair
        mock_retriever.search.assert_not_called()
        mock_generator.generate.assert_not_called()
    
    def test_exec_repair_triggers_on_low_score(self):
        """Test repair executor triggers on low scores."""
        mock_retriever = Mock()
        mock_retriever.search.return_value = [
            ("text1", 0.9, {}),
            ("text2", 0.8, {})
        ]
        
        mock_generator = Mock()
        mock_output = Mock()
        mock_output.text = "Repaired answer"
        mock_generator.generate.return_value = mock_output
        
        ctx: Context = {
            "query": "test",
            "intent": "definition",
            "meta": {},
            "answer": "Poor answer",
            "verify": {"score": 0.5},  # Below threshold
            "context_text": "Old context"
        }
        cfg = Stage4Config(accept_threshold=0.72)
        
        result = exec_repair(ctx, mock_retriever, mock_generator, cfg)
        
        # Should trigger repair
        mock_retriever.search.assert_called_once()
        assert result["answer"] == "Repaired answer"


class TestPromptBuilding:
    """Test prompt building helpers."""
    
    def test_build_prompt_default(self):
        """Test default prompt building."""
        prompt = build_prompt_for_intent(
            intent="definition",
            query="What is ML?",
            context_block="ML is machine learning.",
            template="explain"
        )
        assert "What is ML?" in prompt
        assert "ML is machine learning." in prompt
    
    def test_build_prompt_comparison(self):
        """Test comparison prompt building."""
        prompt = build_prompt_for_intent(
            intent="comparison",
            query="Compare A and B",
            context_block="Context",
            template="compare"
        )
        assert "compare" in prompt.lower()
        assert "Compare A and B" in prompt
    
    def test_build_prompt_how_to(self):
        """Test how-to prompt building."""
        prompt = build_prompt_for_intent(
            intent="how_to",
            query="How to train a model?",
            context_block="Context",
            template="how_to"
        )
        assert "step" in prompt.lower()
        assert "How to train a model?" in prompt
    
    def test_build_prompt_constrained(self):
        """Test constrained prompt building."""
        prompt = build_prompt_for_intent(
            intent="definition",
            query="Test query",
            context_block="Context",
            template="constrained"
        )
        assert "concise" in prompt.lower() or "strict" in prompt.lower() or "precise" in prompt.lower()


class TestScoreAnswer:
    """Test answer scoring."""
    
    def test_score_answer_returns_dict(self):
        """Test that score_answer returns proper dictionary."""
        result = score_answer(
            answer="This is a test answer.",
            context="Context for the answer.",
            query="Test query"
        )
        assert isinstance(result, dict)
        assert "score" in result
        assert "issues" in result
        assert "dimensions" in result
    
    def test_score_answer_score_range(self):
        """Test that scores are in valid range."""
        result = score_answer(
            answer="Test answer",
            context="Context",
            query="Query"
        )
        assert 0.0 <= result["score"] <= 1.0


class TestRunPlan:
    """Test full plan execution."""
    
    def test_run_plan_basic_execution(self):
        """Test basic plan execution."""
        mock_retriever = Mock()
        mock_retriever.search.return_value = [
            ("Relevant text about ML.", 0.9, {}),
            ("More context here.", 0.8, {})
        ]
        
        mock_generator = Mock()
        mock_output = Mock()
        mock_output.text = "Machine learning is a field of AI."
        mock_generator.generate.return_value = mock_output
        
        answer, ctx = run_plan(
            query="What is machine learning?",
            intent="definition",
            retriever=mock_retriever,
            generator=mock_generator
        )
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "meta" in ctx
        assert "path" in ctx["meta"]
        assert len(ctx["meta"]["path"]) > 0
    
    def test_run_plan_tracks_execution_path(self):
        """Test that execution path is tracked."""
        mock_retriever = Mock()
        mock_retriever.search.return_value = [("text", 0.9, {})]
        
        mock_generator = Mock()
        mock_output = Mock()
        mock_output.text = "Answer"
        mock_generator.generate.return_value = mock_output
        
        answer, ctx = run_plan(
            query="Test query",
            intent="definition",
            retriever=mock_retriever,
            generator=mock_generator
        )
        
        assert "path" in ctx["meta"]
        assert "retrieve" in ctx["meta"]["path"]
        assert "generate" in ctx["meta"]["path"]
    
    def test_run_plan_early_exit_on_acceptance(self):
        """Test that plan exits early when answer is accepted."""
        mock_retriever = Mock()
        mock_retriever.search.return_value = [
            ("High quality context.", 0.95, {})
        ]
        
        mock_generator = Mock()
        mock_output = Mock()
        # Generate a high-quality answer that will be accepted
        mock_output.text = "This is a comprehensive and accurate answer with good coverage."
        mock_generator.generate.return_value = mock_output
        
        cfg = Stage4Config(accept_threshold=0.3)  # Low threshold for test
        
        answer, ctx = run_plan(
            query="Test query",
            intent="definition",
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=cfg
        )
        
        # Verify step exists and has acceptable score
        if "verify" in ctx["meta"]["path"]:
            verify_idx = ctx["meta"]["path"].index("verify")
            # Repair should not be executed after acceptance
            if verify_idx < len(ctx["meta"]["path"]) - 1:
                # If there are more steps, none should be repair
                assert "repair" not in ctx["meta"]["path"][verify_idx + 1:]
    
    def test_run_plan_custom_config(self):
        """Test plan execution with custom configuration."""
        mock_retriever = Mock()
        mock_retriever.search.return_value = [("text", 0.9, {})]
        
        mock_generator = Mock()
        mock_output = Mock()
        mock_output.text = "Answer"
        mock_generator.generate.return_value = mock_output
        
        custom_cfg = Stage4Config(
            top_k_first=20,
            rerank_top_k=10,
            accept_threshold=0.85
        )
        
        answer, ctx = run_plan(
            query="Test",
            intent="definition",
            retriever=mock_retriever,
            generator=mock_generator,
            cfg=custom_cfg
        )
        
        assert isinstance(answer, str)
        assert "meta" in ctx


class TestStage4Config:
    """Test configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        cfg = Stage4Config()
        assert cfg.top_k_first == 12
        assert cfg.rerank_top_k == 8
        assert cfg.accept_threshold == 0.72
        assert cfg.max_repairs == 1
    
    def test_custom_config(self):
        """Test custom configuration."""
        cfg = Stage4Config(
            top_k_first=20,
            rerank_top_k=15,
            accept_threshold=0.8,
            temperature_init=0.3
        )
        assert cfg.top_k_first == 20
        assert cfg.rerank_top_k == 15
        assert cfg.accept_threshold == 0.8
        assert cfg.temperature_init == 0.3


class TestCandidateDataclass:
    """Test Candidate dataclass."""
    
    def test_candidate_creation(self):
        """Test creating candidate with required fields."""
        c = Candidate(text="test text", score=0.9)
        assert c.text == "test text"
        assert c.score == 0.9
        assert c.metadata == {}
    
    def test_candidate_with_metadata(self):
        """Test candidate with metadata."""
        c = Candidate(
            text="test",
            score=0.8,
            metadata={"source": "doc1", "page": 5}
        )
        assert c.metadata["source"] == "doc1"
        assert c.metadata["page"] == 5