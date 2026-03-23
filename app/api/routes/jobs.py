"""Route for `/v1/jobs/{job_id}`."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import build_job_engine
from app.api.schemas.jobs import JobStatusResponse
from app.db.session import get_session
from app.engines.job_engine import JobEngine

router = APIRouter(tags=["jobs"])


def get_engine(session: Session = Depends(get_session)) -> JobEngine:
    """Build the job engine for the request."""
    return build_job_engine(session)


@router.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(
    job_id: str,
    engine: JobEngine = Depends(get_engine),
    session: Session = Depends(get_session),
) -> JobStatusResponse:
    """Return a background job status."""
    response = engine.get_job_status(job_id)
    session.commit()
    return response
