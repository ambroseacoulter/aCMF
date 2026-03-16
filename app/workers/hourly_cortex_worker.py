"""Celery task for hourly cortex maintenance."""

from __future__ import annotations

import logging

from app.api.dependencies import build_worker_dependencies
from app.db.session import get_session_factory
from app.storage.user_repo import UserRepository
from app.workers.queue import celery_app
from app.workers.sync_graph_projection_worker import dispatch_graph_projection_task

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.hourly_cortex_worker.hourly_cortex_task")
def hourly_cortex_task(user_id: str | None = None) -> int:
    """Run hourly cortex maintenance for one user or all users."""
    session = get_session_factory()()
    try:
        deps = build_worker_dependencies(session)
        user_repo: UserRepository = deps["user_repo"]  # type: ignore[assignment]
        cortex_engine = deps["cortex_engine"]
        user_ids = [user_id] if user_id else user_repo.get_all_ids()
        processed = 0
        projection_event_ids: list[str] = []
        for current_user_id in user_ids:
            if current_user_id:
                projection_event_ids.extend(cortex_engine.run_hourly(current_user_id))
                processed += 1
        session.commit()
        if projection_event_ids:
            try:
                dispatch_graph_projection_task(list(dict.fromkeys(projection_event_ids)))
            except Exception:
                logger.warning(
                    "Failed to dispatch immediate graph projection sync after hourly cortex; beat will retry.",
                    exc_info=True,
                )
        return processed
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
