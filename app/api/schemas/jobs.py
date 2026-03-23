"""Schemas for `/v1/jobs/{job_id}`."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class JobStatusResponse(BaseModel):
    """Read-only background job status payload."""

    job_id: str
    job_type: str
    user_id: str | None = None
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    result_notes: dict | None = None
