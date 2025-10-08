"""
Simple corpus chat with Ollama.

Loads documents from data/corpus, builds an index, and allows interactive chat.
"""

from pathlib import Path
from rag_papers.retrieval.ensemble_retriever import EnsembleRetriever
from rag_papers.index.build_bm25 import build_bm25
from rag_papers.index.build_vector_store import build_vector_store
from rag_papers.retrieval.router_dag import run_chat_plan, Stage4Config
from rag_papers.generation.ollama_generator import OllamaGenerator

# Load corpus documents
corpus_dir = Path("data/corpus")
print("="*70)
print("📚 Loading corpus from:", corpus_dir)
print("="*70)

docs = []
doc_files = list(corpus_dir.glob("*.txt"))

if not doc_files:
    print("❌ No .txt files found in data/corpus")
    exit(1)

for doc_file in doc_files:
    content = doc_file.read_text(encoding="utf-8")
    docs.append(content)
    print(f"✓ Loaded: {doc_file.name} ({len(content)} chars)")

print(f"\n✅ Loaded {len(docs)} documents\n")

# Build indices
print("🔧 Building search indices...")
bm25 = build_bm25(docs)
vstore, model = build_vector_store(docs)
retriever = EnsembleRetriever(bm25, vstore, model, docs)
print(f"✓ Built BM25 and vector indices\n")

# Initialize Ollama generator
print("🤖 Initializing Ollama (llama3)...")
try:
    generator = OllamaGenerator(model="llama3")
    print("✓ Connected to Ollama\n")
except Exception as e:
    print(f"❌ Failed to connect to Ollama: {e}")
    print("\nMake sure Ollama is running:")
    print("  1. Install Ollama from https://ollama.ai")
    print("  2. Run: ollama run llama3")
    print("  3. Try this script again")
    exit(1)

# Configuration
cfg = Stage4Config(
    bm25_weight_init=0.7,
    vector_weight_init=0.3,
    temperature_init=0.2,
    accept_threshold=0.6,
    enable_memory=False,  # Disabled for now
)

# Interactive chat loop
print("="*70)
print("💬 Interactive RAG Chat (type 'quit' to exit)")
print("="*70)

history = []

while True:
    try:
        query = input("\n🧑 You: ").strip()
        
        if not query:
            continue
            
        if query.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        # Run chat with history
        print("🤖 Assistant: ", end="", flush=True)
        answer, ctx = run_chat_plan(
            query=query,
            history=history,
            retriever=retriever,
            generator=generator,
            cfg=cfg,
            use_cache=False,
            session_id="local_session",
            memory_integration=None
        )
        
        print(answer)
        
        # Show metadata
        meta = ctx.get("meta", {})
        verify_dict = ctx.get("verify", {})
        verify_score = float(verify_dict.get("score", 0.0))
        retrieved_count = len(ctx.get("candidates", []))
        
        print(f"\n📊 Retrieved: {retrieved_count} docs | Verify score: {verify_score:.3f}")
        
        # Update history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        
        # Keep last 10 messages (5 turns)
        if len(history) > 10:
            history = history[-10:]
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        continue
