"""Job persistence operations."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Job


class JobRepository:
    """Repository for background job records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, job_type: str, user_id: str | None, payload: dict) -> Job:
        """Create a pending job."""
        job = Job(job_type=job_type, user_id=user_id, payload_json=payload)
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: str) -> Job | None:
        """Return a job by ID."""
        return self.session.get(Job, job_id)

    def mark_running(self, job_id: str) -> Job | None:
        """Mark a job as running."""
        job = self.get(job_id)
        if job is not None:
            job.status = "running"
        return job

    def mark_succeeded(self, job_id: str, notes: dict | None = None) -> Job | None:
        """Mark a job as succeeded."""
        job = self.get(job_id)
        if job is not None:
            job.status = "succeeded"
            job.error_message = None
            if notes:
                job.payload_json = {**job.payload_json, "result_notes": notes}
        return job

    def mark_failed(self, job_id: str, error_message: str) -> Job | None:
        """Mark a job as failed."""
        job = self.get(job_id)
        if job is not None:
            job.status = "failed"
            job.error_message = error_message
        return job
