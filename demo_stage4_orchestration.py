"""
Demo script for Stage 4: DAG-based orchestration with planning and repair.

Demonstrates the complete pipeline with routing, re-ranking, pruning,
contextualization, generation, verification, and repair capabilities.
"""

import os
from pathlib import Path
from rag_papers.retrieval.ensemble_retriever import EnsembleRetriever
from rag_papers.generation.generator import MockGenerator
from rag_papers.retrieval.router_dag import (
    run_plan,
    Stage4Config,
    plan_for_intent
)


def main():
    """Run Stage 4 orchestration demo."""
    print("=" * 80)
    print("Stage 4: DAG-Based Orchestration Demo")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Setup (Mock components for demo)
    # ========================================================================
    
    print("Setting up components...")
    print()
    
    # Mock retriever with sample documents
    class DemoRetriever(EnsembleRetriever):
        """Demo retriever with hardcoded responses."""
        
        def __init__(self):
            # Don't call super().__init__() to avoid loading real indexes
            pass
        
        def search(self, query, top_k=10, bm25_weight=0.5, vector_weight=0.5):
            """Return mock results based on query."""
            if "neural network" in query.lower():
                return [
                    (
                        "Neural networks are computational models inspired by biological neurons. "
                        "They consist of layers of interconnected nodes that process information.",
                        0.92,
                        {"source": "doc1.pdf", "page": 5}
                    ),
                    (
                        "Deep learning uses neural networks with multiple hidden layers to learn "
                        "hierarchical representations of data.",
                        0.88,
                        {"source": "doc2.pdf", "page": 12}
                    ),
                    (
                        "Training neural networks involves adjusting weights through backpropagation "
                        "to minimize a loss function.",
                        0.85,
                        {"source": "doc3.pdf", "page": 8}
                    ),
                    (
                        "Convolutional neural networks are specialized for processing grid-like data "
                        "such as images.",
                        0.83,
                        {"source": "doc4.pdf", "page": 15}
                    ),
                ]
            elif "transformer" in query.lower():
                return [
                    (
                        "Transformers use self-attention mechanisms to process sequences in parallel. "
                        "They replaced RNNs as the dominant architecture for NLP.",
                        0.94,
                        {"source": "doc5.pdf", "page": 3}
                    ),
                    (
                        "The attention mechanism computes weighted relationships between all positions "
                        "in the input sequence.",
                        0.89,
                        {"source": "doc6.pdf", "page": 7}
                    ),
                ]
            else:
                return [
                    (
                        "Machine learning is a subfield of artificial intelligence focused on "
                        "algorithms that improve through experience.",
                        0.75,
                        {"source": "doc7.pdf", "page": 1}
                    ),
                ]
    
    retriever = DemoRetriever()
    generator = MockGenerator()
    
    # ========================================================================
    # Demo 1: Definition Query with Default Config
    # ========================================================================
    
    print("Demo 1: Definition Query")
    print("-" * 80)
    
    query1 = "What is a neural network?"
    intent1 = "definition"
    
    print(f"Query: {query1}")
    print(f"Intent: {intent1}")
    print()
    
    # Show the plan
    plan1 = plan_for_intent(intent1)
    print(f"Execution Plan: {[s['name'] for s in plan1['steps']]}")
    print()
    
    # Run the pipeline
    answer1, ctx1 = run_plan(
        query=query1,
        intent=intent1,
        retriever=retriever,
        generator=generator
    )
    
    print(f"Answer: {answer1}")
    print()
    print(f"Execution Path: {ctx1['meta']['path']}")
    print(f"Verification Score: {ctx1.get('verify', {}).get('score', 'N/A')}")
    print()
    print()
    
    # ========================================================================
    # Demo 2: Comparison Query with Custom Config
    # ========================================================================
    
    print("Demo 2: Comparison Query with Custom Configuration")
    print("-" * 80)
    
    query2 = "Compare neural networks and transformers"
    intent2 = "comparison"
    
    print(f"Query: {query2}")
    print(f"Intent: {intent2}")
    print()
    
    # Custom config with tighter parameters
    custom_cfg = Stage4Config(
        top_k_first=6,
        rerank_top_k=4,
        prune_min_overlap=2,
        accept_threshold=0.65,
        temperature_init=0.15
    )
    
    print(f"Custom Config:")
    print(f"  - top_k_first: {custom_cfg.top_k_first}")
    print(f"  - rerank_top_k: {custom_cfg.rerank_top_k}")
    print(f"  - prune_min_overlap: {custom_cfg.prune_min_overlap}")
    print(f"  - accept_threshold: {custom_cfg.accept_threshold}")
    print()
    
    # Show the plan
    plan2 = plan_for_intent(intent2)
    print(f"Execution Plan: {[s['name'] for s in plan2['steps']]}")
    print()
    
    # Run the pipeline
    answer2, ctx2 = run_plan(
        query=query2,
        intent=intent2,
        retriever=retriever,
        generator=generator,
        cfg=custom_cfg
    )
    
    print(f"Answer: {answer2}")
    print()
    print(f"Execution Path: {ctx2['meta']['path']}")
    print(f"Number of candidates retrieved: {len(ctx2.get('candidates', []))}")
    print(f"Verification Score: {ctx2.get('verify', {}).get('score', 'N/A')}")
    print()
    print()
    
    # ========================================================================
    # Demo 3: Show Timing Information
    # ========================================================================
    
    print("Demo 3: Timing Breakdown")
    print("-" * 80)
    
    query3 = "Explain transformer architecture"
    intent3 = "explanation"
    
    print(f"Query: {query3}")
    print(f"Intent: {intent3}")
    print()
    
    answer3, ctx3 = run_plan(
        query=query3,
        intent=intent3,
        retriever=retriever,
        generator=generator
    )
    
    print(f"Answer: {answer3}")
    print()
    print("Timing Breakdown (seconds):")
    for step, timing in ctx3['meta']['timings'].items():
        print(f"  - {step}: {timing:.4f}")
    print()
    total_time = sum(ctx3['meta']['timings'].values())
    print(f"Total Execution Time: {total_time:.4f} seconds")
    print()
    print()
    
    # ========================================================================
    # Demo 4: Show Repair Mechanism (simulated)
    # ========================================================================
    
    print("Demo 4: Repair Mechanism")
    print("-" * 80)
    print()
    print("The repair mechanism is triggered automatically when:")
    print("  1. Verification score < accept_threshold")
    print("  2. Repair tightens retrieval (higher BM25 weight)")
    print("  3. Temperature is lowered for more constrained generation")
    print()
    
    cfg_with_high_threshold = Stage4Config(accept_threshold=0.95)
    print(f"Using high threshold: {cfg_with_high_threshold.accept_threshold}")
    print(f"Initial weights: BM25={cfg_with_high_threshold.bm25_weight_init}, Vector={cfg_with_high_threshold.vector_weight_init}")
    print(f"Repair weights: BM25={cfg_with_high_threshold.bm25_weight_on_repair}, Vector={cfg_with_high_threshold.vector_weight_on_repair}")
    print(f"Initial temp: {cfg_with_high_threshold.temperature_init}, Repair temp: {cfg_with_high_threshold.temperature_on_repair}")
    print()
    
    answer4, ctx4 = run_plan(
        query="What is a neural network?",
        intent="definition",
        retriever=retriever,
        generator=generator,
        cfg=cfg_with_high_threshold
    )
    
    # Check if repair was executed
    if "repair" in ctx4['meta']['path']:
        print("✓ Repair was triggered!")
        repair_idx = ctx4['meta']['path'].index("repair")
        print(f"  Execution path: {' → '.join(ctx4['meta']['path'][:repair_idx + 1])}")
    else:
        print("✗ Answer was accepted without repair")
        print(f"  Verification score: {ctx4.get('verify', {}).get('score', 'N/A')}")
    
    print()
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("=" * 80)
    print("Stage 4 Orchestration Summary")
    print("=" * 80)
    print()
    print("Capabilities demonstrated:")
    print("  ✓ Intent-based planning (definition, comparison, explanation)")
    print("  ✓ Multi-step execution (retrieve → rerank → prune → contextualize → generate)")
    print("  ✓ Verification and quality scoring")
    print("  ✓ Automatic repair with tighter parameters")
    print("  ✓ Configurable thresholds and weights")
    print("  ✓ Execution path tracking and timing")
    print()
    print("The DAG-based orchestration provides:")
    print("  • Deterministic, testable routing")
    print("  • Quality-aware repair loops")
    print("  • Fine-grained control over retrieval and generation")
    print("  • Transparent execution with metadata tracking")
    print()


if __name__ == "__main__":
    main()
