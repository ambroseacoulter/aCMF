"""Background job read orchestration."""

from __future__ import annotations

from fastapi import HTTPException

from app.api.schemas.jobs import JobStatusResponse
from app.storage.job_repo import JobRepository


class JobEngine:
    """Read background job status."""

    def __init__(self, job_repo: JobRepository) -> None:
        self.job_repo = job_repo

    def get_job_status(self, job_id: str) -> JobStatusResponse:
        """Return one background job status payload."""
        job = self.job_repo.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        payload = job.payload_json or {}
        result_notes = payload.get("result_notes")
        return JobStatusResponse(
            job_id=job.id,
            job_type=job.job_type,
            user_id=job.user_id,
            status=job.status,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
            result_notes=result_notes if isinstance(result_notes, dict) else None,
        )
