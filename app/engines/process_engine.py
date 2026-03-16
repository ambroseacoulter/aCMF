"""Post-turn orchestration."""

from __future__ import annotations

from app.api.schemas.process import ProcessRequest, ProcessResponse
from app.core.time import utc_now
from app.core.enums import JobType
from app.storage.container_repo import ContainerRepository
from app.storage.job_repo import JobRepository
from app.storage.user_repo import UserRepository


class ProcessEngine:
    """Handle API-time process-turn orchestration."""

    def __init__(self, user_repo: UserRepository, container_repo: ContainerRepository, job_repo: JobRepository) -> None:
        self.user_repo = user_repo
        self.container_repo = container_repo
        self.job_repo = job_repo

    def enqueue_turn(self, request: ProcessRequest) -> ProcessResponse:
        """Upsert user/container state and enqueue the background job."""
        _, created_user = self.user_repo.get_or_create(request.user_id)
        created_containers = self.container_repo.create_missing(request.user_id, [container.model_dump() for container in request.containers])
        job = self.job_repo.create(JobType.PROCESS_TURN.value, request.user_id, request.model_dump(mode="json"))
        from app.workers.process_turn_worker import process_turn_task

        process_turn_task.delay(job.id)
        return ProcessResponse(
            status="accepted",
            job_id=job.id,
            created_user=created_user,
            created_containers=created_containers,
            accepted_at=utc_now(),
        )
