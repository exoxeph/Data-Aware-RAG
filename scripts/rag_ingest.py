"""
CLI utility for async corpus ingestion.

Usage:
    poetry run rag-ingest --dir data/corpus
    poetry run rag-ingest --dir data/corpus --model all-MiniLM-L6-v2 --no-cache
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx


async def tail_progress(job_id: str, base_url: str = "http://localhost:8000"):
    """
    Poll job status and display progress until completion.
    
    Args:
        job_id: Job ID to monitor
        base_url: API base URL
    """
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(f"{base_url}/ingest/status/{job_id}")
                response.raise_for_status()
                status = response.json()
                
                state = status["state"]
                progress = status["progress"]
                message = status["message"]
                
                # Display progress
                bar_length = 40
                filled = int(bar_length * progress)
                bar = "=" * filled + "-" * (bar_length - filled)
                
                print(f"\r[{bar}] {progress*100:.1f}% | {state:10s} | {message}", end="", flush=True)
                
                # Check if done
                if state in ["succeeded", "failed"]:
                    print()  # New line after completion
                    
                    if state == "succeeded":
                        print("\n✅ Ingestion completed successfully!")
                        payload = status.get("payload", {})
                        if "meta" in payload:
                            meta = payload["meta"]
                            print(f"   Corpus ID: {meta.get('corpus_id', 'unknown')}")
                            print(f"   Documents: {meta.get('doc_count', 0)}")
                            print(f"   Dimension: {meta.get('dim', 0)}")
                            print(f"   Index dir: {payload.get('index_dir', 'unknown')}")
                        
                        if "embed_cache" in payload:
                            cache = payload["embed_cache"]
                            print(f"\n   Cache hits: {cache.get('hits', 0)}")
                            print(f"   Cache misses: {cache.get('misses', 0)}")
                            print(f"   Hit rate: {cache.get('hit_rate', 0.0)*100:.1f}%")
                        
                        return 0
                    else:
                        print("\n❌ Ingestion failed!")
                        error = status.get("payload", {}).get("error", "Unknown error")
                        print(f"   Error: {error}")
                        return 1
                
                # Wait before next poll
                await asyncio.sleep(0.5)
                
            except httpx.HTTPError as e:
                print(f"\n❌ Error polling status: {e}")
                return 1


async def start_ingestion(
    corpus_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    use_cache: bool = True,
    base_url: str = "http://localhost:8000",
):
    """
    Start ingestion job and tail progress.
    
    Args:
        corpus_dir: Path to corpus directory
        model_name: Embedding model name
        use_cache: Whether to use embedding cache
        base_url: API base URL
    """
    async with httpx.AsyncClient() as client:
        # Start job
        print(f"🚀 Starting ingestion for: {corpus_dir}")
        print(f"   Model: {model_name}")
        print(f"   Cache: {'enabled' if use_cache else 'disabled'}")
        print()
        
        try:
            response = await client.post(
                f"{base_url}/ingest/start",
                json={
                    "corpus_dir": str(corpus_dir),
                    "model_name": model_name,
                    "use_cache": use_cache,
                },
                timeout=10.0,
            )
            response.raise_for_status()
            result = response.json()
            
            job_id = result["job_id"]
            print(f"✓ Job started: {job_id}\n")
            
            # Tail progress
            return await tail_progress(job_id, base_url)
            
        except httpx.HTTPError as e:
            print(f"❌ Failed to start ingestion: {e}")
            if hasattr(e, "response") and e.response:
                print(f"   Status: {e.response.status_code}")
                try:
                    detail = e.response.json()
                    print(f"   Detail: {detail}")
                except:
                    print(f"   Response: {e.response.text[:200]}")
            return 1


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Start async corpus ingestion with progress monitoring"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Path to corpus directory containing .txt/.md files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    
    args = parser.parse_args()
    
    # Validate corpus directory
    if not args.dir.exists():
        print(f"❌ Corpus directory not found: {args.dir}")
        sys.exit(1)
    
    # Run async ingestion
    exit_code = asyncio.run(
        start_ingestion(
            corpus_dir=args.dir,
            model_name=args.model,
            use_cache=not args.no_cache,
            base_url=args.url,
        )
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
