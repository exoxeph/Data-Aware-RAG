"""
Unit tests for rag_papers/persist/sqlite_store.py

Tests thread-safe SQLite KV operations with WAL mode.
"""
import pytest
import threading
import time
from pathlib import Path
from rag_papers.persist.sqlite_store import KVStore


def test_set_get_delete_roundtrip(kv):
    """Basic set/get/delete operations."""
    table = "embeddings"
    key = "test_key_1"
    value = b"test value data"
    
    # Set value
    kv.set(table, key, value)
    
    # Get value
    retrieved = kv.get(table, key)
    assert retrieved == value
    
    # Delete value
    kv.delete(table, key)
    
    # Should return None after deletion
    assert kv.get(table, key) is None


def test_persistence_after_reopen(tmp_path):
    """Values should persist after closing and reopening store."""
    db_path = tmp_path / "persist_test.db"
    
    # First session - write data
    store1 = KVStore(db_path)
    store1.set("embeddings", "key1", b"value1")
    store1.set("retrieval", "key2", b"value2")
    store1.close()
    
    # Second session - read data
    store2 = KVStore(db_path)
    assert store2.get("embeddings", "key1") == b"value1"
    assert store2.get("retrieval", "key2") == b"value2"
    store2.close()


def test_multiple_tables(kv):
    """Different tables should store data independently."""
    tables = ["embeddings", "retrieval", "answers", "sessions"]
    
    for i, table in enumerate(tables):
        key = f"key_{i}"
        value = f"value_{i}".encode()
        kv.set(table, key, value)
    
    # Verify all values stored correctly
    for i, table in enumerate(tables):
        key = f"key_{i}"
        expected = f"value_{i}".encode()
        assert kv.get(table, key) == expected


def test_parallel_writes_no_crash(kv):
    """Parallel writes from multiple threads should not crash."""
    num_threads = 10
    writes_per_thread = 20
    errors = []
    
    def write_task(thread_id):
        try:
            for i in range(writes_per_thread):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}".encode()
                kv.set("embeddings", key, value)
                time.sleep(0.001)  # Small delay to encourage interleaving
        except Exception as e:
            errors.append(e)
    
    threads = [
        threading.Thread(target=write_task, args=(i,))
        for i in range(num_threads)
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # No errors should occur
    assert len(errors) == 0
    
    # Verify all writes succeeded
    for thread_id in range(num_threads):
        for i in range(writes_per_thread):
            key = f"thread_{thread_id}_key_{i}"
            value = kv.get("embeddings", key)
            assert value is not None


def test_vacuum_reduces_size(tmp_path):
    """Vacuum should reduce database size after deletions."""
    db_path = tmp_path / "vacuum_test.db"
    store = KVStore(db_path)
    
    # Write many entries
    for i in range(1000):
        key = f"key_{i}"
        value = b"x" * 1000  # 1KB each
        store.set("embeddings", key, value)
    
    size_before = db_path.stat().st_size
    
    # Delete all entries
    for i in range(1000):
        key = f"key_{i}"
        store.delete("embeddings", key)
    
    # Vacuum
    store.vacuum()
    
    size_after = db_path.stat().st_size
    
    # Size should be significantly reduced
    assert size_after < size_before * 0.5
    
    store.close()


def test_overwrite_existing_key(kv):
    """Overwriting a key should replace the value."""
    table = "embeddings"
    key = "test_key"
    
    # First write
    kv.set(table, key, b"original value")
    assert kv.get(table, key) == b"original value"
    
    # Overwrite
    kv.set(table, key, b"new value")
    assert kv.get(table, key) == b"new value"


def test_get_nonexistent_key(kv):
    """Getting a nonexistent key should return None."""
    result = kv.get("embeddings", "nonexistent_key")
    assert result is None


def test_delete_nonexistent_key(kv):
    """Deleting a nonexistent key should not raise an error."""
    # Should not crash
    kv.delete("embeddings", "nonexistent_key")


def test_large_value_storage(kv):
    """Should handle large binary values."""
    table = "embeddings"
    key = "large_value_key"
    # 1MB of data
    large_value = b"x" * (1024 * 1024)
    
    kv.set(table, key, large_value)
    retrieved = kv.get(table, key)
    
    assert retrieved == large_value
    assert len(retrieved) == 1024 * 1024


def test_timestamp_tracking(kv):
    """Timestamps should be recorded for LRU tracking."""
    table = "embeddings"
    key = "ts_test_key"
    value = b"test"
    
    # First write
    kv.set(table, key, value)
    time.sleep(0.1)
    
    # Read (updates timestamp)
    kv.get(table, key)
    time.sleep(0.1)
    
    # Second write (updates timestamp)
    kv.set(table, key, b"updated")
    
    # We can't directly test timestamp values, but verify no errors
    assert kv.get(table, key) == b"updated"


def test_concurrent_read_write(kv):
    """Concurrent reads and writes should work with WAL mode."""
    errors = []
    results = []
    
    def writer():
        try:
            for i in range(50):
                kv.set("embeddings", f"key_{i}", f"value_{i}".encode())
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)
    
    def reader():
        try:
            for i in range(50):
                value = kv.get("embeddings", f"key_{i}")
                results.append(value)
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)
    
    write_thread = threading.Thread(target=writer)
    read_thread = threading.Thread(target=reader)
    
    write_thread.start()
    time.sleep(0.01)  # Let writer get ahead
    read_thread.start()
    
    write_thread.join()
    read_thread.join()
    
    # No errors should occur
    assert len(errors) == 0


def test_unicode_keys(kv):
    """Should handle Unicode keys correctly."""
    table = "embeddings"
    key = "测试_key_🔥"
    value = b"unicode test value"
    
    kv.set(table, key, value)
    assert kv.get(table, key) == value


def test_empty_value(kv):
    """Should handle empty byte values."""
    table = "embeddings"
    key = "empty_key"
    value = b""
    
    kv.set(table, key, value)
    assert kv.get(table, key) == b""
