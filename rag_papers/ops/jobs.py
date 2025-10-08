"""
Job management system for async operations.

Provides in-process job queue with persistent state tracking.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Callable
from concurrent.futures import ThreadPoolExecutor


@dataclass
class JobStatus:
    """Status of a background job."""
    
    id: str
    state: Literal["queued", "running", "succeeded", "failed"]
    progress: float                 # 0.0 to 1.0
    message: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    payload: dict = None            # Job-specific data
    
    def __post_init__(self):
        if self.payload is None:
            self.payload = {}
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "JobStatus":
        """Create from dict."""
        return cls(**data)


class JobManager:
    """
    In-process async job runner with persistent state.
    
    Uses thread pool for CPU-bound work and asyncio for coordination.
    State persisted to JSONL file for durability.
    """
    
    def __init__(self, state_file: Path, max_workers: int = 2):
        """
        Initialize job manager.
        
        Args:
            state_file: Path to JSONL file for job state
            max_workers: Max concurrent jobs
        """
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # In-memory job registry
        self.jobs: dict[str, JobStatus] = {}
        
        # Load existing jobs
        self._load_state()
        
        # Lock for thread-safe updates
        self._lock = asyncio.Lock()
    
    def _load_state(self) -> None:
        """Load jobs from state file."""
        if not self.state_file.exists():
            return
        
        with open(self.state_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                job = JobStatus.from_dict(data)
                self.jobs[job.id] = job
    
    def _append_state(self, job: JobStatus) -> None:
        """Append job state to file."""
        with open(self.state_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(job.to_dict()) + "\n")
    
    def submit_job(
        self,
        job_type: str,
        worker_fn: Callable,
        payload: dict = None,
    ) -> str:
        """
        Submit a generic background job.
        
        Args:
            job_type: Type identifier for job
            worker_fn: Async function to execute
            payload: Initial payload data
        
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        job = JobStatus(
            id=job_id,
            state="queued",
            progress=0.0,
            message=f"Queued {job_type}",
            payload=payload or {"type": job_type},
        )
        
        self.jobs[job_id] = job
        self._append_state(job)
        
        # Schedule execution
        asyncio.create_task(self._run_job(job_id, worker_fn))
        
        return job_id
    
    def submit(self, worker_fn: Callable, job_type: str = "generic", payload: dict = None) -> str:
        """Alias for submit_job with backwards-compatible signature."""
        return self.submit_job(job_type=job_type, worker_fn=worker_fn, payload=payload)
    
    async def _run_job(self, job_id: str, worker_fn: Callable) -> None:
        """Execute job with state tracking."""
        job = self.jobs[job_id]
        
        # Mark as running
        job.state = "running"
        job.started_at = datetime.utcnow().isoformat()
        job.message = "Starting..."
        self._append_state(job)
        
        try:
            # Execute worker (pass job for progress updates)
            await worker_fn(job, self)
            
            # Mark as succeeded
            job.state = "succeeded"
            job.progress = 1.0
            job.finished_at = datetime.utcnow().isoformat()
            job.message = "Completed successfully"
            self._append_state(job)
            
        except Exception as e:
            # Mark as failed
            job.state = "failed"
            job.finished_at = datetime.utcnow().isoformat()
            job.message = f"Failed: {str(e)}"
            job.payload["error"] = str(e)
            self._append_state(job)
    
    def update_progress(self, job_id: str, progress: float, message: str) -> None:
        """
        Update job progress (called from worker).
        
        Args:
            job_id: Job ID
            progress: Progress value (0.0 to 1.0)
            message: Status message
        """
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        job.progress = progress
        job.message = message
        self._append_state(job)
    
    def get(self, job_id: str) -> Optional[JobStatus]:
        """
        Get job status by ID.
        
        Args:
            job_id: Job ID
        
        Returns:
            JobStatus if found, None otherwise
        """
        return self.jobs.get(job_id)
    
    def list(self, state: Optional[str] = None) -> list[JobStatus]:
        """
        List all jobs, optionally filtered by state.
        
        Args:
            state: Filter by state (queued, running, succeeded, failed)
        
        Returns:
            List of JobStatus objects
        """
        jobs = list(self.jobs.values())
        
        if state:
            jobs = [j for j in jobs if j.state == state]
        
        # Sort by most recent first
        jobs.sort(key=lambda j: j.started_at or "", reverse=True)
        
        return jobs
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Remove old completed/failed jobs.
        
        Args:
            max_age_hours: Max age in hours
        
        Returns:
            Number of jobs removed
        """
        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0
        
        for job_id, job in list(self.jobs.items()):
            if job.state in ["succeeded", "failed"]:
                if job.finished_at:
                    finished_ts = datetime.fromisoformat(job.finished_at).timestamp()
                    if finished_ts < cutoff:
                        del self.jobs[job_id]
                        removed += 1
        
        return removed
    
    def shutdown(self) -> None:
        """Shutdown executor and cleanup."""
        self.executor.shutdown(wait=True)
