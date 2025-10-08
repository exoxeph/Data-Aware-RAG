"""
CLI for batch processing queries through the RAG pipeline.

Usage:
    poetry run rag-batch --input queries.txt --out answers.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_papers.retrieval.router_dag import run_plan, Stage4Config
from rag_papers.generation.generator import MockGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Process batch queries through RAG pipeline"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file with queries (one per line, optional |intent suffix)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL file for answers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MockGenerator",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of documents to retrieve",
    )
    
    args = parser.parse_args()
    
    # Read input queries
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    queries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Parse query and optional intent
            if "|" in line:
                query, intent = line.rsplit("|", 1)
                query = query.strip()
                intent = intent.strip()
            else:
                query = line
                intent = "unknown"
            
            queries.append({
                "line": line_num,
                "query": query,
                "intent": intent,
            })
    
    print(f"Loaded {len(queries)} queries from {input_path}")
    
    # Create config and generator (map CLI args to Stage4Config parameters)
    cfg = Stage4Config(
        temperature_init=args.temperature,
        top_k_first=args.top_k,
    )
    generator = MockGenerator()
    
    # Create minimal setup (same as rag_eval.py)
    docs = _get_default_docs()
    retriever = _get_mock_retriever()
    
    # Process queries
    results = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(queries, 1):
            print(f"Processing {i}/{len(queries)}: {item['query'][:50]}...")
            
            try:
                answer, context = run_plan(
                    query=item["query"],
                    intent=item["intent"],
                    docs=docs,
                    retriever=retriever,
                    generator=generator,
                    cfg=cfg,
                )
                
                result = {
                    "line": item["line"],
                    "query": item["query"],
                    "intent": item["intent"],
                    "answer": answer,
                    "score": context.get("verify_score", 0.0),
                    "accepted": context.get("accepted", False),
                    "path": context.get("path", []),
                    "repair_used": "repair" in context.get("path", []),
                }
                
                # Write result immediately (streaming)
                f.write(json.dumps(result) + "\n")
                f.flush()
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing query: {e}")
                result = {
                    "line": item["line"],
                    "query": item["query"],
                    "intent": item["intent"],
                    "error": str(e),
                }
                f.write(json.dumps(result) + "\n")
                f.flush()
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total Queries: {len(queries)}")
    print(f"Successful: {len(results)}")
    print(f"Output: {out_path}")
    
    if results:
        accepted = sum(1 for r in results if r.get("accepted", False))
        avg_score = sum(r.get("score", 0) for r in results) / len(results)
        print(f"Accept Rate: {accepted}/{len(results)} ({accepted/len(results):.2%})")
        print(f"Avg Score: {avg_score:.3f}")
    
    print("=" * 60)
    
    return 0


def _get_default_docs() -> list[str]:
    """Get default document corpus (same as rag_eval.py)."""
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
    class MockRetriever:
        def retrieve(self, query: str, k: int = 10):
            return [
                {"doc": doc, "score": 0.8}
                for doc in _get_default_docs()[:k]
            ]
    
    return MockRetriever()


if __name__ == "__main__":
    sys.exit(main())
