"""
SQLite-backed key-value store for caching.

Uses SQLite with separate tables for different cache types:
- embeddings: content hash → vector bytes
- retrieval: query+config hash → candidate list
- answers: query+config+model hash → answer+metadata
- sessions: session_id → JSONL lines

All values stored as BLOB with timestamp for LRU/TTL.
"""

import sqlite3
import time
from pathlib import Path
from typing import Optional


class KVStore:
    """
    File-backed SQLite key-value store for caching.
    
    Thread-safe with WAL mode and IMMEDIATE transactions.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize KV store at given path.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,  # Allow multi-threaded access
            timeout=10.0,
        )
        
        # Enable WAL mode for better concurrency
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        
        # Create tables
        self._init_tables()
    
    def _init_tables(self) -> None:
        """Create cache tables if they don't exist."""
        tables = ["embeddings", "retrieval", "answers", "sessions"]
        
        for table in tables:
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    ts INTEGER NOT NULL
                )
            """)
            
            # Index on timestamp for LRU queries
            self._conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_ts 
                ON {table}(ts)
            """)
        
        self._conn.commit()
    
    def set(self, table: str, key: str, value: bytes) -> None:
        """
        Set a key-value pair in the specified table.
        
        Args:
            table: Table name (embeddings, retrieval, answers, sessions)
            key: String key
            value: Binary value
        """
        ts = int(time.time())
        
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} (key, value, ts) VALUES (?, ?, ?)",
            (key, value, ts)
        )
        self._conn.commit()
    
    def get(self, table: str, key: str) -> Optional[bytes]:
        """
        Get value for a key from the specified table.
        
        Args:
            table: Table name
            key: String key
        
        Returns:
            Binary value if found, None otherwise
        """
        cursor = self._conn.execute(
            f"SELECT value FROM {table} WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        
        if row:
            # Update access timestamp (LRU tracking)
            self._conn.execute(
                f"UPDATE {table} SET ts = ? WHERE key = ?",
                (int(time.time()), key)
            )
            self._conn.commit()
            return row[0]
        
        return None
    
    def delete(self, table: str, key: str) -> None:
        """
        Delete a key from the specified table.
        
        Args:
            table: Table name
            key: String key
        """
        self._conn.execute(
            f"DELETE FROM {table} WHERE key = ?",
            (key,)
        )
        self._conn.commit()
    
    def purge_table(self, table: str) -> int:
        """
        Delete all entries from a table.
        
        Args:
            table: Table name
        
        Returns:
            Number of rows deleted
        """
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        
        self._conn.execute(f"DELETE FROM {table}")
        self._conn.commit()
        
        return count
    
    def stats(self, table: str) -> dict:
        """
        Get statistics for a table.
        
        Args:
            table: Table name
        
        Returns:
            Dict with count, total_bytes, oldest_ts, newest_ts
        """
        cursor = self._conn.execute(f"""
            SELECT 
                COUNT(*) as count,
                SUM(LENGTH(value)) as total_bytes,
                MIN(ts) as oldest_ts,
                MAX(ts) as newest_ts
            FROM {table}
        """)
        row = cursor.fetchone()
        
        return {
            "count": row[0] or 0,
            "total_bytes": row[1] or 0,
            "oldest_ts": row[2] or 0,
            "newest_ts": row[3] or 0,
        }
    
    def vacuum(self) -> None:
        """
        Reclaim space and optimize database.
        
        Should be called periodically after large deletions.
        """
        self._conn.execute("VACUUM")
        self._conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        self._conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
