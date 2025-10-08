"""
CLI utility for cache management.

Usage:
    poetry run rag-cache --stats
    poetry run rag-cache --purge answers,retrieval
    poetry run rag-cache --purge all
"""

import argparse
import sys
from pathlib import Path

from rag_papers.persist import KVStore


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def format_time(ts: int) -> str:
    """Format unix timestamp as human-readable string."""
    if ts == 0:
        return "never"
    
    from datetime import datetime
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def show_stats(cache_dir: Path):
    """
    Display cache statistics.
    
    Args:
        cache_dir: Directory containing cache.db
    """
    db_path = cache_dir / "cache.db"
    
    if not db_path.exists():
        print(f"❌ Cache database not found: {db_path}")
        return 1
    
    print(f"📊 Cache Statistics: {db_path}\n")
    
    with KVStore(db_path) as kv:
        tables = ["embeddings", "retrieval", "answers", "sessions"]
        
        # Print header
        print(f"{'Table':<15} {'Count':>10} {'Size':>12} {'Oldest':>20} {'Newest':>20}")
        print("=" * 80)
        
        total_count = 0
        total_bytes = 0
        
        for table in tables:
            stats = kv.stats(table)
            count = stats["count"]
            bytes_val = stats["total_bytes"]
            oldest = stats["oldest_ts"]
            newest = stats["newest_ts"]
            
            total_count += count
            total_bytes += bytes_val
            
            print(
                f"{table:<15} {count:>10,} {format_bytes(bytes_val):>12} "
                f"{format_time(oldest):>20} {format_time(newest):>20}"
            )
        
        print("=" * 80)
        print(f"{'TOTAL':<15} {total_count:>10,} {format_bytes(total_bytes):>12}")
        print()
    
    return 0


def purge_cache(cache_dir: Path, tables: list[str]):
    """
    Purge cache tables.
    
    Args:
        cache_dir: Directory containing cache.db
        tables: List of table names to purge (or ["all"])
    """
    db_path = cache_dir / "cache.db"
    
    if not db_path.exists():
        print(f"❌ Cache database not found: {db_path}")
        return 1
    
    valid_tables = ["embeddings", "retrieval", "answers", "sessions"]
    
    # Handle "all"
    if "all" in tables:
        tables = valid_tables
    
    # Validate
    invalid = set(tables) - set(valid_tables)
    if invalid:
        print(f"❌ Invalid table names: {invalid}")
        print(f"   Valid tables: {', '.join(valid_tables)}, all")
        return 1
    
    print(f"🗑️  Purging cache tables: {', '.join(tables)}\n")
    
    with KVStore(db_path) as kv:
        total_purged = 0
        
        for table in tables:
            count = kv.purge_table(table)
            total_purged += count
            print(f"   {table:<15} {count:>10,} entries purged")
        
        print(f"\n   TOTAL:          {total_purged:>10,} entries purged")
        
        # Vacuum to reclaim space
        print("\n🔧 Vacuuming database...")
        kv.vacuum()
        print("   ✓ Done")
    
    print("\n✅ Cache purge complete")
    return 0


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Manage RAG cache (view stats, purge entries)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show cache statistics",
    )
    parser.add_argument(
        "--purge",
        type=str,
        help="Purge cache tables (comma-separated: embeddings,retrieval,answers,sessions or 'all')",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Cache directory (default: data/cache)",
    )
    
    args = parser.parse_args()
    
    # Require at least one action
    if not args.stats and not args.purge:
        parser.print_help()
        print("\n❌ Error: Must specify --stats or --purge")
        sys.exit(1)
    
    # Show stats
    if args.stats:
        exit_code = show_stats(args.cache_dir)
        if exit_code != 0:
            sys.exit(exit_code)
    
    # Purge
    if args.purge:
        tables = [t.strip() for t in args.purge.split(",")]
        exit_code = purge_cache(args.cache_dir, tables)
        sys.exit(exit_code)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
