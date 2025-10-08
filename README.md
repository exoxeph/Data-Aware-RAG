# RAG Papers

A Data-Aware RAG system for processing and querying PDF research papers using marker-pdf, DuckDB, and vector search.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system specifically designed for research papers. It processes PDFs through marker-pdf for high-quality markdown conversion, extracts and normalizes tables into DuckDB, builds vector and BM25 indices, and provides intelligent retrieval routing for text, table, and figure queries.

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