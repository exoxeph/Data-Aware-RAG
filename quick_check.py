"""
Quick sanity check that the DAG pipeline works with real retrieval.

Usage:
    poetry run python quick_check.py
"""

from rag_papers.retrieval.ensemble_retriever import EnsembleRetriever
from rag_papers.index.build_bm25 import build_bm25
from rag_papers.index.build_vector_store import build_vector_store
from rag_papers.retrieval.router_dag import run_plan, Stage4Config
from rag_papers.generation.generator import MockGenerator

# Minimal corpus
docs = [
    "Transfer learning reuses weights from a source model trained on one task to initialize a target model for a related task. This approach reduces training time and data requirements.",
    "CNNs use convolution and pooling layers to process spatial data like images. Transformers use self-attention mechanisms to process sequential data.",
    "Backpropagation uses the chain rule to compute gradients of the loss function with respect to network weights, enabling gradient descent optimization.",
    "Neural networks consist of interconnected layers of neurons that transform inputs through weighted sums and activation functions.",
    "Fine-tuning adapts a pretrained model to a specific task by continuing training on task-specific data with a lower learning rate.",
]

print("Building indices...")
bm25 = build_bm25(docs)
vstore, model = build_vector_store(docs)
retriever = EnsembleRetriever(bm25, vstore, model, docs)
print(f"✓ Built retriever with {len(docs)} documents\n")

# Test queries
queries = [
    ("What is transfer learning?", "definition"),
    ("How do CNNs process images?", "explanation"),
    ("Compare CNNs and Transformers", "comparison"),
]

cfg = Stage4Config(
    bm25_weight_init=0.7,
    vector_weight_init=0.3,
    temperature_init=0.2,
    accept_threshold=0.5,  # Lower for mock generator
)

for query, intent in queries:
    print(f"Query: {query}")
    print(f"Intent: {intent}")
    
    answer, ctx = run_plan(
        query=query,
        intent=intent,
        retriever=retriever,
        generator=MockGenerator(),
        cfg=cfg
    )
    
    # Extract metrics the same way eval runner does
    meta = ctx.get("meta", {})
    verify_dict = ctx.get("verify", {})
    verify_score = float(verify_dict.get("score", 0.0))
    accepted = verify_score >= cfg.accept_threshold
    retrieved_count = len(ctx.get("candidates", []))
    context_text = ctx.get("context_text", "")
    context_chars = len(context_text)
    
    print(f"Answer: {answer[:100]}...")
    print(f"Retrieved: {retrieved_count} docs")
    print(f"Context chars: {context_chars}")
    print(f"Verify score: {verify_score:.3f}")
    print(f"Accepted: {accepted}")
    print(f"Path: {' -> '.join(meta.get('path', []))}")
    print("-" * 80)
