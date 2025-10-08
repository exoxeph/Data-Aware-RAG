# RAG Papers

A Data-Aware RAG system for processing and querying PDF research papers using marker-pdf, DuckDB, and vector search.

## Overview

This project implements a **production-grade** Retrieval-Augmented Generation (RAG) system specifically designed for research papers. It processes PDFs through marker-pdf for high-quality markdown conversion, extracts and normalizes tables into DuckDB, builds vector and BM25 indices, and provides intelligent retrieval routing for text, table, and figure queries.

## 🚀 Project Status

**Current Stage:** Stage 7 - Production Hardening (78% complete)  
**Latest Milestone:** Full UI integration with async ingestion + 3-tier caching ✅

### Completed Stages

- ✅ **Stage 1-3:** Core RAG pipeline (ingestion → indexing → retrieval → generation)
- ✅ **Stage 4:** DAG orchestration with multi-path query planning
- ✅ **Stage 5:** Evaluation framework with RAGAS metrics
- ✅ **Stage 6:** Streamlit UI with techno/neon aesthetic
- ⚙️ **Stage 7:** Production hardening with async jobs + caching (7/9 tasks complete)
  - ✅ Persistence layer (content-addressable caching)
  - ✅ Async ingestion (background jobs with progress tracking)
  - ✅ 3-tier caching (embeddings → retrieval → answers)
  - ✅ API endpoints (5 new routes)
  - ✅ CLI tools (rag-ingest, rag-cache)
  - ✅ Streamlit UI integration (async ingestion + cache badges)
  - 🔴 Unit tests (pending)
  - ⚠️ Documentation polish (partial)

### Next Up

- **Stage 8:** PDF ingestion → RAG chat (minimal implementation)
- **Stage 9:** Advanced features (multi-PDF context, citations, etc.)

### Performance Highlights

- **Query speedup:** 400x faster with answer cache (~5ms vs ~2000ms)
- **Re-indexing:** 3x faster with embedding cache
- **Storage overhead:** ~5MB per 1000 queries (negligible)

See `STAGE7_COMPLETION_SUMMARY.md` for full details.

## Quick Start

1. **Install dependencies:**
   ```bash
   poetry install
   ```

2. **Run tests:**
   ```bash
   poetry run test
   ```

3. **Start development API:**
   ```bash
   poetry run dev-api
   ```

## Development

- **Lint code:** `poetry run lint`
- **Type check:** `poetry run typecheck`
- **Run tests with coverage:** `poetry run test`

## Repository Layout

```
rag_papers/
├── config/          # Settings and configuration
├── ingest/          # PDF processing and marker integration
├── index/           # Vector store and BM25 index building
├── retrieval/       # Query routing and ensemble retrieval
├── compute/         # Table operations and analytics
├── generation/      # LLM prompting and response generation
└── api/             # FastAPI web service

tests/
├── unit/            # Unit tests for individual components
├── integration/     # End-to-end workflow tests
└── data/fixtures/   # Test data and mock responses

data/
├── raw_pdfs/        # Original PDF files
├── marker/          # Marker-pdf output (markdown and JSON)
├── figures/         # Extracted figures and images
├── parquet/         # Normalized table data
└── tables.duckdb    # Table metadata and structure
```