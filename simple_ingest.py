#!/usr/bin/env python3
"""
Simple corpus ingestion script.

Directly indexes documents from data/corpus without requiring API server.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_papers.ingest.loader import load_text_files
from rag_papers.ingest.embedder import EmbeddingModel
from rag_papers.index.faiss_index import FAISSIndex
from rag_papers.persist import corpus_id_from_dir

def main():
    """Index the corpus documents."""
    
    print("="*70)
    print("📚 Simple Corpus Ingestion")
    print("="*70)
    
    # Set paths
    corpus_dir = Path("data/corpus")
    index_dir = Path("data/indexes")
    index_dir.mkdir(parents=True, exist_ok=True)
    
    if not corpus_dir.exists():
        print(f"❌ Corpus directory not found: {corpus_dir}")
        return 1
    
    # Load documents
    print(f"\n📖 Loading documents from: {corpus_dir}")
    documents = load_text_files(corpus_dir)
    print(f"   Found {len(documents)} documents")
    
    if len(documents) == 0:
        print("❌ No documents found!")
        return 1
    
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc.doc_id} ({len(doc.text)} chars)")
    
    # Initialize embedding model
    print(f"\n🧠 Initializing embedding model: all-MiniLM-L6-v2")
    embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    
    # Generate corpus ID
    corpus_id = corpus_id_from_dir(corpus_dir)
    print(f"\n🔖 Corpus ID: {corpus_id}")
    
    # Create FAISS index
    print(f"\n⚡ Creating FAISS index...")
    index = FAISSIndex.create_from_docs(
        documents=documents,
        embedder=embedder,
        index_dir=index_dir,
        corpus_id=corpus_id
    )
    
    print(f"\n✅ Index created successfully!")
    print(f"   Dimension: {index.dimension}")
    print(f"   Document count: {len(documents)}")
    print(f"   Index directory: {index.index_dir}")
    
    # Test search
    print(f"\n🔍 Testing search...")
    query = "What are neural networks?"
    print(f"   Query: '{query}'")
    
    # Embed query
    query_embedding = embedder.embed_text(query)
    
    # Search
    results = index.search(query_embedding, top_k=3)
    print(f"\n   Top 3 results:")
    for i, result in enumerate(results, 1):
        doc_snippet = documents[result.doc_idx].text[:100].replace('\n', ' ')
        print(f"   {i}. Score: {result.score:.4f}")
        print(f"      Doc: {documents[result.doc_idx].doc_id}")
        print(f"      Snippet: {doc_snippet}...")
    
    print(f"\n✅ Ingestion complete!")
    print(f"\n💡 Next steps:")
    print(f"   1. Start API server: poetry run rag-serve")
    print(f"   2. Use Chat interface: poetry run streamlit run streamlit_app.py")
    print(f"   3. Or test with: python production_example.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
