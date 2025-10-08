"""
CLI for running evaluations on the RAG pipeline.

Usage:
    poetry run rag-eval --dataset qa_small --out runs/
    poetry run rag-eval --dataset qa_small --out runs/ --corpus-dir data/corpus
    poetry run rag-eval --dataset qa_small --out runs/ --duckdb data/telemetry.duckdb
"""

import argparse
import sys
import glob
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_papers.eval import (
    load_dataset,
    evaluate_dataset,
    save_report,
)
from rag_papers.retrieval.router_dag import Stage4Config
from rag_papers.generation.generator import MockGenerator
from rag_papers.eval.telemetry import new_run_id


def load_corpus(corpus_dir: str) -> list[str]:
    """Load all text documents from a corpus directory."""
    docs = []
    patterns = [
        os.path.join(corpus_dir, "*.txt"),
        os.path.join(corpus_dir, "*.md"),
        os.path.join(corpus_dir, "**", "*.txt"),
        os.path.join(corpus_dir, "**", "*.md"),
    ]
    
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if content:
                        docs.append(content)
                        print(f"  Loaded: {os.path.basename(path)}")
            except Exception as e:
                print(f"  Warning: Could not read {path}: {e}")
    
    return docs


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on RAG pipeline with offline dataset"
    )
    
    # Dataset and output
    parser.add_argument(
        "--dataset",
        type=str,
        default="qa_small",
        help="Dataset name (looks for data/eval/{dataset}.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Directory containing corpus documents (.txt, .md). If not provided, uses mock retriever.",
    )
    parser.add_argument(
        "--duckdb",
        type=str,
        default=None,
        help="Optional DuckDB path for telemetry persistence",
    )
    
    # Config overrides
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.5,
        help="BM25 weight for ensemble retrieval",
    )
    parser.add_argument(
        "--vector-weight",
        type=float,
        default=0.5,
        help="Vector weight for ensemble retrieval",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=8,
        help="Number of documents after reranking",
    )
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=0.72,
        help="Acceptance threshold for verification",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature",
    )
    parser.add_argument(
        "--prune-overlap",
        type=float,
        default=0.3,
        help="Overlap threshold for sentence pruning",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=8000,
        help="Maximum context characters",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MockGenerator",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    print(f"Loaded {len(dataset.items)} items from {args.dataset}")
    
    # Build retriever and docs
    if args.corpus_dir:
        print(f"\nLoading corpus from: {args.corpus_dir}")
        docs = load_corpus(args.corpus_dir)
        
        if not docs:
            print(f"Error: No .txt or .md files found in {args.corpus_dir}")
            return 1
        
        print(f"Loaded {len(docs)} documents")
        
        # Build real retriever with BM25 + vector store
        print("Building BM25 index...")
        from rag_papers.index.build_bm25 import build_bm25
        bm25_index = build_bm25(docs)
        
        print("Building vector store...")
        from rag_papers.index.build_vector_store import build_vector_store
        vector_store, model = build_vector_store(docs)
        
        print("Creating EnsembleRetriever...")
        from rag_papers.retrieval.ensemble_retriever import EnsembleRetriever
        retriever = EnsembleRetriever(
            bm25_index=bm25_index,
            vector_store=vector_store,
            model=model,
            text_chunks=docs
        )
        print(f"✓ Using real EnsembleRetriever with {len(docs)} documents")
    else:
        print("\nWARNING: No --corpus-dir provided, using mock retriever")
        print("Results will show all zeros. Use --corpus-dir data/corpus for real metrics.")
        docs = _get_default_docs()
        retriever = _get_mock_retriever()
        print(f"Using mock retriever (for testing only)")
    
    # Create config (map CLI args to Stage4Config parameters)
    cfg = Stage4Config(
        bm25_weight_init=args.bm25_weight,
        vector_weight_init=args.vector_weight,
        top_k_first=args.top_k,
        rerank_top_k=args.rerank_top_k,
        accept_threshold=args.accept_threshold,
        temperature_init=args.temperature,
        prune_min_overlap=args.prune_overlap,
        context_max_chars=args.max_context_chars,
    )
    
    # Create generator (MockGenerator for offline eval, no API keys needed)
    generator = MockGenerator()
    print(f"Using MockGenerator (no LLM API calls)")
    
    # Generate run ID
    run_id = new_run_id()
    print(f"\nRun ID: {run_id}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    try:
        report = evaluate_dataset(
            dataset=dataset,
            retriever=retriever,
            generator=generator,
            cfg=cfg,
            run_id=run_id,
            duckdb_path=args.duckdb,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save report
    print("\nSaving results...")
    try:
        out_dir = save_report(report, args.out, run_id=run_id)
        print(f"Results saved to: {out_dir}")
    except Exception as e:
        print(f"Error saving report: {e}")
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Dataset: {report.dataset}")
    print(f"Total Queries: {int(report.metrics['total_queries'])}")
    print(f"Accept@1: {report.metrics['accept_at_1']:.2%}")
    print(f"Avg Score: {report.metrics['avg_score']:.3f}")
    print(f"Repair Rate: {report.metrics['repair_rate']:.2%}")
    print(f"P50 Latency: {report.metrics['latency_p50_ms']:.1f} ms")
    print(f"P95 Latency: {report.metrics['latency_p95_ms']:.1f} ms")
    print(f"Avg Context: {report.metrics['avg_context_chars']:.0f} chars")
    print("=" * 60)
    
    return 0


def _get_default_docs() -> list[str]:
    """Get default document corpus for evaluation."""
    return [
        "Transfer learning is a machine learning technique where a model trained on one task is adapted for a second related task. "
        "It leverages pretrained models and fine-tuning to achieve better performance with less data.",
        
        "Neural networks are computing systems inspired by biological neural networks. "
        "They consist of layers of interconnected nodes that process information.",
        
        "Fine-tuning involves taking a pretrained model and training it further on a specific dataset. "
        "This allows the model to adapt its learned weights to the target task.",
        
        "Backpropagation is the key algorithm for training neural networks. "
        "It computes gradients of the loss function with respect to network weights.",
        
        "Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images. "
        "They use convolutional layers to detect spatial patterns.",
        
        "Transformers are neural network architectures based on self-attention mechanisms. "
        "They have revolutionized natural language processing tasks.",
        
        "Attention mechanisms allow models to focus on relevant parts of the input. "
        "Self-attention computes relationships between all positions in a sequence.",
        
        "Pretrained models are neural networks trained on large datasets. "
        "They can be used as starting points for transfer learning.",
        
        "Gradient descent is an optimization algorithm for finding model parameters. "
        "It iteratively moves in the direction of steepest descent of the loss function.",
        
        "Overfitting occurs when a model learns training data too well. "
        "Regularization techniques help prevent overfitting.",
    ]


def _get_mock_retriever():
    """Create a mock retriever for testing."""
    # Return a simple mock that returns all docs
    class MockRetriever:
        def search(self, query: str, top_k: int = 10, 
                  bm25_weight: float = 0.5, vector_weight: float = 0.5):
            # Return mock candidates as list of (text, score) tuples
            return [
                (doc, 0.8)
                for doc in _get_default_docs()[:top_k]
            ]
    
    return MockRetriever()


if __name__ == "__main__":
    sys.exit(main())
