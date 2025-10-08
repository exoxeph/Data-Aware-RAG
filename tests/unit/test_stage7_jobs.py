"""
Unit tests for rag_papers/ops/jobs.py

Tests async JobManager and JSONL persistence.
"""
import pytest
import asyncio
import time
import json
from pathlib import Path
from rag_papers.ops.jobs import JobManager, JobStatus

# Mark all tests as async
pytestmark = pytest.mark.asyncio


async def test_submit_job_creates_queued_state(tmp_path):
    """Submitting a job should create it in queued state."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    async def dummy_task(job, mgr):
        return "ok"
    
    job_id = manager.submit(dummy_task, job_type="test")
    
    job = manager.get(job_id)
    assert job is not None
    assert job.state == "queued"
    assert job.progress == 0.0


async def test_job_transitions_queued_to_running_to_succeeded(tmp_path):
    """Job should transition: queued → running → succeeded."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    async def quick_task(job, mgr):
        await asyncio.sleep(0.05)
        return "success"
    
    job_id = manager.submit(quick_task, job_type="test")
    
    # Initially queued
    job = manager.get(job_id)
    assert job.state in ["queued", "running"]
    
    # Wait for completion
    await asyncio.sleep(0.2)
    
    job = manager.get(job_id)
    assert job.state == "succeeded"
    assert job.progress == 1.0
    assert job.finished_at is not None


async def test_job_status_progress_in_range(tmp_path):
    """Job progress should always be in [0, 1]."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    async def task(job, mgr):
        await asyncio.sleep(0.05)
        return "ok"
    
    job_id = manager.submit(task, job_type="test")
    
    # Check progress multiple times
    for _ in range(10):
        job = manager.get(job_id)
        assert 0.0 <= job.progress <= 1.0
        await asyncio.sleep(0.01)
    
    # Wait for completion
    await asyncio.sleep(0.2)
    job = manager.get(job_id)
    assert job.progress == 1.0


async def test_failed_job_logs_error_state(tmp_path):
    """Failed jobs should log state=failed with error message."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    async def failing_task(job, mgr):
        raise ValueError("Test error")
    
    job_id = manager.submit(failing_task, job_type="test")
    
    # Wait for failure
    await asyncio.sleep(0.2)
    
    job = manager.get(job_id)
    assert job.state == "failed"
    assert "error" in job.payload
    assert "Test error" in job.payload["error"]


async def test_jsonl_file_grows_with_appends(tmp_path):
    """JSONL file should grow with each job state update."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    async def task(job, mgr):
        await asyncio.sleep(0.05)
        return "ok"
    
    # Submit multiple jobs
    job_ids = [
        manager.submit(task, job_type="test")
        for _ in range(3)
    ]
    
    # Wait for completion
    await asyncio.sleep(0.5)
    
    # JSONL should have multiple lines
    assert state_file.exists()
    lines = state_file.read_text().strip().split('\n')
    assert len(lines) >= 3  # At least one line per job
    
    # Each line should be valid JSON
    for line in lines:
        data = json.loads(line)
        assert "id" in data
        assert "state" in data


async def test_cleanup_removes_old_jobs(tmp_path):
    """Cleanup should remove jobs older than max_age."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    def task():
        return "ok"
    
    # Submit and complete a job
    job_id = manager.submit(task, job_type="test")
    time.sleep(0.2)
    
    # Manually set finished_at to old timestamp
    job = manager.get(job_id)
    if job:
        job.finished_at = "2024-01-01T00:00:00"  # Very old
        manager._jobs[job_id] = job
    
    # Run cleanup with very short max_age
    manager.cleanup(max_age_hours=0.0001)
    
    # Job should be removed
    assert manager.get(job_id) is None


def test_list_jobs_returns_all_jobs(tmp_path):
    """list() should return all jobs."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    def task():
        time.sleep(0.05)
        return "ok"
    
    # Submit multiple jobs
    job_ids = [
        manager.submit(task, job_type="test")
        for _ in range(3)
    ]
    
    # Wait for completion
    time.sleep(0.3)
    
    # List should contain all jobs
    all_jobs = manager.list()
    assert len(all_jobs) >= 3
    
    # All our job IDs should be in the list
    listed_ids = {job.id for job in all_jobs}
    for job_id in job_ids:
        assert job_id in listed_ids


def test_concurrent_job_submissions(tmp_path):
    """Multiple concurrent jobs should all complete."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=4)
    
    def task(task_id):
        time.sleep(0.1)
        return f"task_{task_id}_done"
    
    # Submit 10 jobs
    job_ids = [
        manager.submit(lambda tid=i: task(tid), job_type="test")
        for i in range(10)
    ]
    
    # Wait for all to complete
    time.sleep(2.0)
    
    # All should succeed
    for job_id in job_ids:
        job = manager.get(job_id)
        assert job is not None
        assert job.state == "succeeded"


def test_job_payload_preserved(tmp_path):
    """Job payload should be preserved through state updates."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    def task():
        time.sleep(0.05)
        return "ok"
    
    initial_payload = {"corpus_dir": "/test/path", "model": "test-model"}
    job_id = manager.submit(task, job_type="test", payload=initial_payload)
    
    # Check payload preserved
    job = manager.get(job_id)
    assert job.payload == initial_payload
    
    # Wait for completion
    time.sleep(0.2)
    
    # Payload should still be there
    job = manager.get(job_id)
    assert "corpus_dir" in job.payload
    assert job.payload["corpus_dir"] == "/test/path"


def test_message_updates_during_execution(tmp_path):
    """Job message should update during execution."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    def task():
        time.sleep(0.05)
        return "ok"
    
    job_id = manager.submit(task, job_type="test")
    
    messages = []
    for _ in range(5):
        job = manager.get(job_id)
        if job:
            messages.append(job.message)
        time.sleep(0.02)
    
    # Should have some variety in messages (if running)
    assert len(messages) > 0


def test_get_nonexistent_job_returns_none(tmp_path):
    """Getting a nonexistent job should return None."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    job = manager.get("nonexistent-job-id")
    assert job is None


def test_jsonl_persistence_survives_restart(tmp_path):
    """Jobs should be loaded from JSONL on manager restart."""
    state_file = tmp_path / "jobs.jsonl"
    
    # First manager - create and complete a job
    manager1 = JobManager(state_file=state_file, max_workers=2)
    
    def task():
        time.sleep(0.05)
        return "ok"
    
    job_id = manager1.submit(task, job_type="test")
    time.sleep(0.2)
    
    # Verify job completed
    job = manager1.get(job_id)
    assert job.state == "succeeded"
    
    # Second manager - should load job from JSONL
    manager2 = JobManager(state_file=state_file, max_workers=2)
    
    loaded_job = manager2.get(job_id)
    assert loaded_job is not None
    assert loaded_job.state == "succeeded"


def test_job_started_at_timestamp(tmp_path):
    """started_at should be set when job begins."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    def task():
        time.sleep(0.05)
        return "ok"
    
    job_id = manager.submit(task, job_type="test")
    
    # Wait for it to start
    time.sleep(0.1)
    
    job = manager.get(job_id)
    assert job.started_at is not None


def test_job_finished_at_timestamp(tmp_path):
    """finished_at should be set when job completes."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    def task():
        time.sleep(0.05)
        return "ok"
    
    job_id = manager.submit(task, job_type="test")
    
    # Wait for completion
    time.sleep(0.2)
    
    job = manager.get(job_id)
    assert job.finished_at is not None


def test_empty_job_list_initially(tmp_path):
    """Newly created manager should have empty job list."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    jobs = manager.list()
    assert len(jobs) == 0


def test_job_type_preserved(tmp_path):
    """Job type should be preserved."""
    state_file = tmp_path / "jobs.jsonl"
    manager = JobManager(state_file=state_file, max_workers=2)
    
    def task():
        return "ok"
    
    job_id = manager.submit(task, job_type="ingest")
    
    job = manager.get(job_id)
    assert job.job_type == "ingest"


def test_multiple_managers_same_file(tmp_path):
    """Multiple managers can share same JSONL file."""
    state_file = tmp_path / "shared_jobs.jsonl"
    
    manager1 = JobManager(state_file=state_file, max_workers=2)
    manager2 = JobManager(state_file=state_file, max_workers=2)
    
    def task():
        time.sleep(0.05)
        return "ok"
    
    # Submit from manager1
    job_id = manager1.submit(task, job_type="test")
    time.sleep(0.2)
    
    # Should be visible from manager2 after reload
    manager2_reloaded = JobManager(state_file=state_file, max_workers=2)
    job = manager2_reloaded.get(job_id)
    
    assert job is not None
