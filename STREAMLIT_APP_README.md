# 🤖 RAG Cockpit - Streamlit UI

A techno-styled Streamlit interface for the RAG pipeline with complete DAG inspection and real-time configuration.

## Features

- **🏗️ Corpus Management**: Upload PDFs, load text files, build BM25 + vector indexes
- **🤖 Query Interface**: Ask questions and see every step of the DAG execution
- **⚙️ Live Configuration**: Tune BM25/vector weights, top-k, pruning, temperature, thresholds
- **📊 Step Inspection**: View retrieved docs, pruned candidates, final context, prompts, timings
- **📈 Telemetry**: Load evaluation runs, view metrics (Accept@1, repair rate, latency)
- **🦙 Ollama Integration**: Switch between Mock and Ollama for local LLM inference
- **🎨 Neon Theme**: Dark glassmorphism UI with cyan/magenta accents

## Quick Start

### 1. Launch the App

```bash
poetry run streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### 2. Build Your Corpus

Navigate to **🏗️ Corpus** page:

1. Load documents from `data/corpus/` (or specify custom directory)
2. Click **"Build Indexes"** to create BM25 and vector embeddings
3. Wait for completion (~30s for 10 documents)

### 3. Ask Questions

Navigate to **🤖 Ask** page:

1. Choose backend:
   - **Mock** (offline, no LLM) - for testing
   - **Ollama** (local LLM) - for real answers (requires `ollama serve`)

2. Enter your query (e.g., "What is transfer learning?")
3. Select intent or use "auto"
4. Adjust configuration knobs if needed
5. Click **"Run Plan"**

### 4. Inspect Results

View three columns:
- **Left**: Final answer with accept/reject status
- **Middle**: Retrieved docs, pruned candidates, final context
- **Right**: Execution path, step timings, raw JSON

### 5. View Metrics

Navigate to **📈 Metrics** page:

- Select a run from `runs/run_*` directories
- View Accept@1, avg score, repair rate, latency
- Inspect worst/best performing queries
- Export to DuckDB if available

## Backend Options

### Mock Generator (Default)

- No external dependencies
- Instant responses
- Perfect for testing pipeline
- Returns generic placeholder answers

### Ollama (Local LLM)

1. Install Ollama: https://ollama.ai
2. Start server: `ollama serve`
3. Pull a model: `ollama pull llama3`
4. Select "Ollama" in the app
5. Choose model (llama3, mistral, phi, etc.)

The app auto-detects Ollama and falls back to Mock if unavailable.

## Configuration Knobs

### Retrieval
- **BM25 Weight** (0-1): Lexical search weight
- **Vector Weight** (0-1): Dense embedding weight
- **Top-K First**: Initial retrieval count (1-50)
- **Rerank Top-K**: Final count after reranking (1-20)

### Processing
- **Prune Min Overlap**: Token overlap threshold (0-5)
- **Prune Max Chars**: Max chars per candidate (500-5000)
- **Context Max Chars**: Final context budget (1000-10000)

### Generation
- **Temperature** (0-1): LLM sampling randomness

### Verification
- **Accept Threshold** (0-1): Min score to accept answer

## Tips

### Getting Non-Zero Metrics

The corpus must contain relevant documents:

```bash
# Use existing corpus
poetry run streamlit run streamlit_app.py

# Or create custom corpus
mkdir -p data/my_corpus
echo "Transfer learning reuses model weights..." > data/my_corpus/ml.txt
```

### Running Evaluations

Generate telemetry data:

```bash
poetry run rag-eval \
  --dataset qa_small \
  --out runs/ \
  --corpus-dir data/corpus \
  --bm25-weight 0.7
```

Then view in the **📈 Metrics** page.

### Tuning for Better Results

1. **Higher BM25 weight** (0.7-0.8) for keyword-heavy queries
2. **Higher vector weight** (0.6-0.8) for semantic queries
3. **Lower accept threshold** (0.5-0.6) to accept more answers
4. **Higher top-k** (15-20) to retrieve more candidates
5. **Lower temperature** (0.1-0.2) for consistent answers

## Architecture

```
streamlit_app.py          # Main entrypoint
app/
  ui_theme.css            # Neon/techno styling
  state.py                # Session state management
  components.py           # Reusable widgets
  adapters/
    ollama_generator.py   # Ollama REST API client
  pages/
    1_🏗️_Corpus.py        # Upload & index
    2_🤖_Ask.py            # Query & inspect
    3_📈_Metrics.py        # Telemetry viewer
```

## Troubleshooting

### "Indexes not built"

Go to 🏗️ Corpus page and click "Build Indexes" first.

### "Ollama offline"

Start Ollama server:
```bash
ollama serve
```

Or switch to Mock generator.

### "No .txt files found"

Ensure corpus directory contains `.txt` or `.md` files:
```bash
ls data/corpus/
```

### Slow index building

Vector embeddings take time. Expected: ~30s for 10 docs, ~5min for 100 docs.

### All zeros in metrics

Use `--corpus-dir data/corpus` with real documents:
```bash
poetry run rag-eval --dataset qa_small --out runs/ --corpus-dir data/corpus
```

## Session History

Query history is stored in `st.session_state` and cleared on page reload. Export to JSONL for persistence:

1. Run queries in **🤖 Ask** page
2. Scroll to "Session History"
3. Click "Export to JSONL"
4. Find file in `runs/session_YYYYMMDD_HHMMSS/session.jsonl`

## Performance

- **Corpus loading**: ~1s for 10 docs
- **Index building**: ~30s for 10 docs (includes vector embeddings)
- **Query execution**: 10-50ms with indexes cached
- **Ollama generation**: 1-5s depending on model size

## Screenshots

### Home Page
Landing page with navigation and quick start guide.

### Corpus Page
Upload PDFs, load corpus, build indexes with progress indicators.

### Ask Page
Query interface with live config knobs, three-column results layout, execution path, step timings, and session history.

### Metrics Page
Load evaluation runs, view aggregate metrics, inspect worst/best queries.

## License

Same as parent project.
