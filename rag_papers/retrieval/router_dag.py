"""
DAG-based orchestration layer for RAG pipeline.

Provides plannable routing that executes retrieval, re-ranking, pruning,
contextualization, generation, verification, and repair steps based on
query intent.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional, Tuple, Any, TypedDict, AsyncIterator
import time

from rag_papers.retrieval.ensemble_retriever import EnsembleRetriever
from rag_papers.index.contextualize import contextualize
from rag_papers.generation.generator import BaseGenerator
from rag_papers.generation.prompts import QueryProcessor
from rag_papers.generation.verifier import ResponseVerifier
from rag_papers.retrieval.prune_sentences import sentence_prune


# Type definitions
StepName = Literal[
    "retrieve", "rerank", "prune", "contextualize",
    "generate", "verify", "repair",
    "step_back", "plan_subqueries", "merge_contrast"  # keep but default off
]


class PlanStep(TypedDict):
    """A single step in the execution plan."""
    name: StepName
    params: Dict[str, Any]


class Plan(TypedDict):
    """Execution plan with ordered steps."""
    steps: List[PlanStep]


@dataclass
class Candidate:
    """A retrieved document candidate."""
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Context(TypedDict, total=False):
    """Execution context passed through the DAG."""
    query: str
    intent: str
    candidates: List[Candidate]          # after retrieval/rerank
    pruned_candidates: List[Candidate]   # after prune
    context_text: str                    # from contextualizer
    answer: Optional[str]
    verify: Optional[Dict[str, Any]]
    meta: Dict[str, Any]                 # timings, params, path, etc.


@dataclass
class Stage4Config:
    """Configuration for Stage 4 orchestration."""
    # Retrieval and ranking
    top_k_first: int = 12
    rerank_top_k: int = 8
    prune_min_overlap: int = 1
    prune_max_chars: int = 2200
    context_max_chars: int = 4000
    accept_threshold: float = 0.72
    max_repairs: int = 1
    
    # Ensemble weights
    bm25_weight_init: float = 0.5
    vector_weight_init: float = 0.5
    bm25_weight_on_repair: float = 0.7
    vector_weight_on_repair: float = 0.3
    
    # Generation
    temperature_init: float = 0.2
    temperature_on_repair: float = 0.1
    
    # Stage 9: Streaming flags
    enable_streaming: bool = True
    max_history_turns: int = 5
    
    # Convenience property for top_k (used in many places)
    @property
    def top_k(self) -> int:
        """Alias for rerank_top_k for backward compatibility."""
        return self.rerank_top_k


# ============================================================================
# Planning / Routing
# ============================================================================

def plan_for_intent(intent: str) -> Plan:
    """
    Create execution plan based on query intent.
    
    Args:
        intent: Query intent (definition, explanation, comparison, etc.)
    
    Returns:
        Plan with ordered steps and parameters
    """
    if intent in {"definition", "explanation", "unknown"}:
        return {
            "steps": [
                {"name": "retrieve", "params": {"k": "cfg.top_k_first"}},
                {"name": "rerank", "params": {"top_k": "cfg.rerank_top_k"}},
                {"name": "prune", "params": {"min_overlap": "cfg.prune_min_overlap", "max_chars": "cfg.prune_max_chars"}},
                {"name": "contextualize", "params": {"max_chars": "cfg.context_max_chars"}},
                {"name": "generate", "params": {"template": "explain", "temperature": "cfg.temperature_init"}},
                {"name": "verify", "params": {}},
                {"name": "repair", "params": {}}  # will early-exit if accepted
            ]
        }
    elif intent == "comparison":
        return {
            "steps": [
                {"name": "retrieve", "params": {"k": "cfg.top_k_first"}},
                {"name": "rerank", "params": {"top_k": "cfg.rerank_top_k"}},
                {"name": "prune", "params": {"min_overlap": "cfg.prune_min_overlap", "max_chars": "cfg.prune_max_chars"}},
                {"name": "contextualize", "params": {"max_chars": "cfg.context_max_chars", "mode": "comparison"}},
                {"name": "generate", "params": {"template": "compare", "temperature": "cfg.temperature_init"}},
                {"name": "verify", "params": {}},
                {"name": "repair", "params": {}}
            ]
        }
    elif intent in {"how_to", "summarization"}:
        return {
            "steps": [
                {"name": "retrieve", "params": {"k": "cfg.top_k_first"}},
                {"name": "rerank", "params": {"top_k": "cfg.rerank_top_k"}},
                {"name": "prune", "params": {"min_overlap": "cfg.prune_min_overlap", "max_chars": "cfg.prune_max_chars"}},
                {"name": "contextualize", "params": {"max_chars": "cfg.context_max_chars", "mode": intent}},
                {"name": "generate", "params": {"template": intent, "temperature": "cfg.temperature_init"}},
                {"name": "verify", "params": {}},
                {"name": "repair", "params": {}}
            ]
        }
    else:
        # fallback route
        return {
            "steps": [
                {"name": "retrieve", "params": {"k": "cfg.top_k_first"}},
                {"name": "prune", "params": {"min_overlap": "cfg.prune_min_overlap", "max_chars": "cfg.prune_max_chars"}},
                {"name": "contextualize", "params": {"max_chars": "cfg.context_max_chars"}},
                {"name": "generate", "params": {"template": "explain", "temperature": "cfg.temperature_init"}},
                {"name": "verify", "params": {}}
            ]
        }


# ============================================================================
# Step Executors
# ============================================================================

def exec_retrieve(
    ctx: Context, 
    retriever: EnsembleRetriever, 
    cfg: Stage4Config, 
    k: int,
    **kwargs
) -> Context:
    """Execute retrieval step."""
    results = retriever.search(
        ctx["query"], 
        top_k=k, 
        bm25_weight=cfg.bm25_weight_init, 
        vector_weight=cfg.vector_weight_init
    )
    ctx["candidates"] = [
        Candidate(
            text=r[0], 
            score=float(r[1]), 
            metadata=r[2] if len(r) > 2 else {}
        ) 
        for r in results
    ]
    return ctx


def exec_rerank(ctx: Context, top_k: int, **kwargs) -> Context:
    """Execute re-ranking step (simple score-based sort)."""
    # Deterministic re-sort by descending score
    ctx["candidates"] = sorted(
        ctx.get("candidates", []), 
        key=lambda c: c.score, 
        reverse=True
    )[:top_k]
    return ctx


def exec_prune(
    ctx: Context, 
    min_overlap: int, 
    max_chars: int, 
    **kwargs
) -> Context:
    """Execute sentence pruning step."""
    pruned = []
    budget = max_chars
    
    for c in ctx.get("candidates", []):
        # Prune sentences in this candidate
        kept = sentence_prune(ctx["query"], c.text, min_overlap=min_overlap)
        if not kept:
            continue
        
        if budget <= 0:
            break
        
        # Cap by budget
        portion = kept[:max(0, budget)]
        pruned.append(Candidate(
            text=portion, 
            score=c.score, 
            metadata=c.metadata
        ))
        budget -= len(portion)
    
    ctx["pruned_candidates"] = pruned or ctx.get("candidates", [])
    return ctx


def exec_contextualize(
    ctx: Context, 
    max_chars: int, 
    mode: Optional[str] = None, 
    **kwargs
) -> Context:
    """Execute contextualization step."""
    texts = [
        c.text 
        for c in (ctx.get("pruned_candidates") or ctx.get("candidates", []))
    ]
    ctx["context_text"] = contextualize(
        texts, 
        ctx["query"], 
        max_length=max_chars
    )
    return ctx


def exec_generate(
    ctx: Context, 
    generator: BaseGenerator, 
    template: str, 
    temperature: float, 
    **kwargs
) -> Context:
    """Execute generation step."""
    prompt = build_prompt_for_intent(
        intent=ctx["intent"],
        query=ctx["query"],
        context_block=ctx.get("context_text", ""),
        template=template
    )
    
    # Generate with parameters
    out = generator.generate(prompt, config=None)
    ctx["answer"] = out.text if hasattr(out, "text") else str(out)
    return ctx


def exec_verify(ctx: Context, **kwargs) -> Context:
    """Execute verification step."""
    ctx["verify"] = score_answer(
        ctx.get("answer", ""),
        ctx.get("context_text", ""),
        ctx["query"]
    )
    return ctx


def exec_repair(
    ctx: Context, 
    retriever: EnsembleRetriever, 
    generator: BaseGenerator, 
    cfg: Stage4Config, 
    **kwargs
) -> Context:
    """Execute repair step (one attempt with tighter parameters)."""
    if not ctx.get("verify"):
        return ctx
    
    # Check if response is already acceptable
    if float(ctx["verify"].get("score", 0.0)) >= cfg.accept_threshold:
        return ctx  # accepted; no repair needed
    
    # One repair attempt: tighten retrieval and temperature
    results = retriever.search(
        ctx["query"], 
        top_k=cfg.rerank_top_k, 
        bm25_weight=cfg.bm25_weight_on_repair, 
        vector_weight=cfg.vector_weight_on_repair
    )
    ctx["candidates"] = [
        Candidate(
            text=r[0], 
            score=float(r[1]), 
            metadata=r[2] if len(r) > 2 else {}
        ) 
        for r in results
    ]
    
    # Stricter prune
    ctx = exec_prune(ctx, cfg.prune_min_overlap + 1, cfg.prune_max_chars)
    ctx = exec_contextualize(ctx, cfg.context_max_chars)
    ctx = exec_generate(ctx, generator, template="constrained", temperature=cfg.temperature_on_repair)
    ctx = exec_verify(ctx)
    
    return ctx


# Executor registry
EXECUTORS = {
    "retrieve": exec_retrieve,
    "rerank": exec_rerank,
    "prune": exec_prune,
    "contextualize": exec_contextualize,
    "generate": exec_generate,
    "verify": exec_verify,
    "repair": exec_repair,
}


# ============================================================================
# Helper Functions
# ============================================================================

def _materialize(params: Dict[str, Any], cfg: Stage4Config) -> Dict[str, Any]:
    """
    Materialize configuration placeholders in parameters.
    
    Replaces strings like "cfg.top_k_first" with actual config values.
    """
    out = {}
    for k, v in params.items():
        if isinstance(v, str) and v.startswith("cfg."):
            out[k] = getattr(cfg, v.split(".", 1)[1])
        else:
            out[k] = v
    return out


def build_prompt_for_intent(
    intent: str, 
    query: str, 
    context_block: str, 
    template: str = "explain"
) -> str:
    """
    Build prompt for LLM based on intent and template.
    
    Args:
        intent: Query intent
        query: User query
        context_block: Retrieved context
        template: Template type (explain, compare, how_to, etc.)
    
    Returns:
        Formatted prompt string
    """
    if template == "compare":
        return f"""Compare and contrast based on the context below.

Context:
{context_block}

Question: {query}

Provide a detailed comparison highlighting key similarities and differences:"""
    
    elif template == "how_to":
        return f"""Provide step-by-step instructions based on the context below.

Context:
{context_block}

Question: {query}

Step-by-step answer:"""
    
    elif template == "summarization":
        return f"""Summarize the key points from the context below.

Context:
{context_block}

Question: {query}

Summary:"""
    
    elif template == "constrained":
        return f"""Answer concisely and accurately based strictly on the context below.

Context:
{context_block}

Question: {query}

Precise answer:"""
    
    else:  # explain or default
        return f"""Answer the question based on the context below.

Context:
{context_block}

Question: {query}

Detailed answer:"""


def score_answer(answer: str, context: str, query: str) -> Dict[str, Any]:
    """
    Score answer quality using the verifier.
    
    Args:
        answer: Generated answer
        context: Context used for generation
        query: Original query
    
    Returns:
        Dictionary with score, issues, and dimensions
    """
    verifier = ResponseVerifier()
    quality = verifier.verify_response(answer, query, context)
    
    return {
        "score": quality.overall_score,
        "issues": [issue.value for issue in quality.issues],
        "dimensions": {
            "relevance": quality.relevance_score,
            "coherence": quality.coherence_score,
            "completeness": quality.completeness_score
        },
        "suggestions": quality.suggestions
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def run_plan(
    query: str, 
    intent: str, 
    retriever: EnsembleRetriever, 
    generator: BaseGenerator, 
    cfg: Optional[Stage4Config] = None
) -> Tuple[str, Context]:
    """
    Execute the full DAG plan for a query.
    
    Args:
        query: User query string
        intent: Query intent (definition, explanation, etc.)
        retriever: Ensemble retriever instance
        generator: LLM generator instance
        cfg: Configuration (uses defaults if None)
    
    Returns:
        Tuple of (answer, context) where context contains execution metadata
    
    Examples:
        >>> answer, ctx = run_plan("What is ML?", "definition", retriever, gen)
        >>> print(ctx["meta"]["path"])
        ['retrieve', 'rerank', 'prune', 'contextualize', 'generate', 'verify']
    """
    cfg = cfg or Stage4Config()
    ctx: Context = {
        "query": query, 
        "intent": intent, 
        "meta": {"path": [], "timings": {}}
    }
    
    plan = plan_for_intent(intent)
    
    for step in plan["steps"]:
        name = step["name"]
        step_start = time.time()
        
        ctx["meta"]["path"].append(name)
        params = _materialize(step["params"], cfg)
        
        # Execute step with appropriate arguments
        if name in {"retrieve", "repair"}:
            ctx = EXECUTORS[name](
                ctx, 
                retriever=retriever, 
                generator=generator, 
                cfg=cfg, 
                **params
            )
        elif name == "generate":
            ctx = EXECUTORS[name](
                ctx, 
                generator=generator, 
                **params
            )
        else:
            ctx = EXECUTORS[name](ctx, **params)
        
        # Record timing
        ctx["meta"]["timings"][name] = time.time() - step_start
        
        # Early exit if verify step accepts the answer
        if name == "verify" and ctx.get("verify"):
            score = float(ctx["verify"].get("score", 0))
            if score >= cfg.accept_threshold:
                break  # Answer accepted, skip repair
    
    return ctx.get("answer", ""), ctx


def run_chat_plan(
    query: str,
    history: List[Dict[str, str]],
    retriever: EnsembleRetriever,
    generator: BaseGenerator,
    cfg: Stage4Config,
    use_cache: bool = True
) -> Tuple[str, Context]:
    """
    Run RAG plan with conversation history context.
    
    Merges recent chat history into query context for history-aware retrieval
    and generation.
    
    Args:
        query: Current user query
        history: List of previous messages [{"role": "user/assistant", "content": "..."}]
        retriever: Document retriever
        generator: LLM generator
        cfg: Pipeline configuration
        use_cache: Whether to use caching
    
    Returns:
        Tuple of (answer, context) where context includes sources and metadata
    
    Example:
        >>> history = [
        ...     {"role": "user", "content": "What is transfer learning?"},
        ...     {"role": "assistant", "content": "Transfer learning is..."}
        ... ]
        >>> answer, ctx = run_chat_plan(
        ...     "How does it work?",
        ...     history,
        ...     retriever,
        ...     generator,
        ...     cfg
        ... )
    """
    # Build history prefix from recent messages (last 5 turns)
    recent_history = history[-5:] if len(history) > 5 else history
    
    if recent_history:
        history_lines = [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in recent_history
        ]
        history_prefix = "\n".join(history_lines) + "\n"
        
        # Augment query with history context
        contextualized_query = f"{history_prefix}User: {query}\nAssistant:"
    else:
        contextualized_query = query
    
    # Run standard plan with history-enriched query
    answer, ctx = run_plan(
        query=contextualized_query,
        retriever=retriever,
        generator=generator,
        cfg=cfg,
        use_cache=use_cache
    )
    
    # Store original query in metadata for tracking
    ctx["meta"]["original_query"] = query
    ctx["meta"]["history_turns"] = len(history)
    
    return answer, ctx


async def run_chat_plan_stream(
    query: str,
    history: List[Dict[str, str]],
    retriever: EnsembleRetriever,
    generator: BaseGenerator,
    cfg: Stage4Config,
    use_cache: bool = True
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream chat plan execution as SSE events.
    
    Yields events:
    - type: meta -> {"run_id", "corpus_id", "retrieved", "ctx_chars"}
    - type: token -> {"t": " word "}
    - type: sources -> {"sources": [{"doc", "score"}]}
    - type: verify -> {"score", "accepted", "issues"}
    - type: cached -> {"answer"} (if cache hit)
    - type: done -> implicit (caller handles)
    
    Args:
        query: User's current question
        history: Previous conversation turns [{"role", "content"}]
        retriever: Ensemble retriever
        generator: LLM generator with stream() method
        cfg: Stage4 configuration
        use_cache: Whether to check answer cache
    
    Yields:
        Dict events for SSE encoding
    """
    import time
    import uuid
    from rag_papers.persist.answer_cache import get_answer, set_answer, AnswerKey, AnswerValue
    from rag_papers.generation.verifier import ResponseVerifier
    
    run_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Build history context (last 5 turns)
    history_prefix = ""
    if history:
        recent_turns = history[-5:]  # Last 5 turns
        for turn in recent_turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_prefix += f"{role.capitalize()}: {content}\n"
        history_prefix += "\n"
    
    # Contextualize query
    contextualized_query = history_prefix + f"User: {query}"
    
    # Check answer cache first
    if use_cache:
        cache_key = AnswerKey(
            query=contextualized_query,
            corpus_id=retriever.corpus_id,
            model_id=generator.__class__.__name__
        )
        cached_answer = get_answer(cache_key)
        
        if cached_answer:
            # Cache hit - return immediately
            yield {
                "type": "cached",
                "data": {"answer": cached_answer.answer}
            }
            return
    
    # Cache miss - run full pipeline
    
    # 1. Retrieve documents
    results = retriever.search(contextualized_query, top_k=cfg.top_k)
    retrieved_count = len(results)
    
    # Yield meta event
    yield {
        "type": "meta",
        "data": {
            "run_id": run_id,
            "corpus_id": retriever.corpus_id,
            "retrieved": retrieved_count,
            "ctx_chars": sum(len(r.content) for r in results)
        }
    }
    
    # Yield sources event
    yield {
        "type": "sources",
        "data": {
            "sources": [
                {
                    "doc": r.document_id,
                    "score": r.score,
                    "snippet": r.content[:200] + "..." if len(r.content) > 200 else r.content
                }
                for r in results[:5]  # Top 5 sources
            ]
        }
    }
    
    # 2. Build prompt
    context_str = "\n\n".join([
        f"[Document {i+1}] {r.content}"
        for i, r in enumerate(results[:cfg.top_k])
    ])
    
    prompt = f"""Context:
{context_str}

Question: {query}

Answer the question based on the context above. Be concise and accurate."""
    
    # 3. Stream generation
    full_answer = ""
    for token in generator.stream(prompt):
        full_answer += token
        yield {
            "type": "token",
            "data": {"t": token}
        }
    
    # 4. Verify answer (optional)
    try:
        verifier = ResponseVerifier()
        verification = verifier.verify(full_answer, results)
        
        yield {
            "type": "verify",
            "data": {
                "score": verification.score,
                "accepted": verification.accepted,
                "issues": verification.issues
            }
        }
    except Exception:
        # Verifier optional - skip if fails
        pass
    
    # 5. Cache answer for future use
    if use_cache:
        answer_value = AnswerValue(
            answer=full_answer,
            sources=[r.document_id for r in results[:5]],
            metadata={
                "run_id": run_id,
                "duration_ms": int((time.time() - start_time) * 1000)
            }
        )
        set_answer(cache_key, answer_value)

