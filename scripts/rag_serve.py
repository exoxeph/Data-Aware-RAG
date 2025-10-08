"""
CLI for launching the FastAPI server.

Usage:
    poetry run rag-serve
    poetry run rag-serve --port 8080 --host 127.0.0.1
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Launch RAG Papers FastAPI server"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    args = parser.parse_args()
    
    print(f"Starting RAG Papers API server on {args.host}:{args.port}")
    print("API documentation available at: http://localhost:{args.port}/docs")
    
    uvicorn.run(
        "rag_papers.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
