"""Celery task for syncing graph projection events into Neo4j."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.api.dependencies import build_worker_dependencies
from app.db.session import get_session_factory
from app.storage.graph_repo import GraphRepository
from app.workers.queue import celery_app

logger = logging.getLogger(__name__)


def dispatch_graph_projection_task(event_ids: list[str] | None = None) -> None:
    """Enqueue an immediate outbox sync for the provided events."""
    kwargs = {"event_ids": event_ids} if event_ids else {}
    sync_graph_projection_task.apply_async(kwargs=kwargs)


@celery_app.task(name="app.workers.sync_graph_projection_worker.sync_graph_projection_task")
def sync_graph_projection_task(event_ids: list[str] | None = None) -> int:
    """Sync pending outbox events into Neo4j."""
    session = get_session_factory()()
    try:
        deps = build_worker_dependencies(session)
        settings = deps["settings"]
        graph_repo: GraphRepository = deps["graph_repo"]  # type: ignore[assignment]
        graph_engine = deps["graph_engine"]
        oldest_pending = graph_repo.get_oldest_pending_outbox_created_at()
        if event_ids:
            events = graph_repo.get_outbox_events_by_ids(event_ids)
        else:
            events = graph_repo.get_pending_outbox_events(limit=settings.graph_projection_batch_size)
        result = graph_engine.sync_projection(events, max_attempts=settings.graph_projection_max_attempts)
        session.commit()
        oldest_pending_age_seconds = None
        if oldest_pending is not None:
            pending_dt = oldest_pending
            if pending_dt.tzinfo is None:
                pending_dt = pending_dt.replace(tzinfo=timezone.utc)
            else:
                pending_dt = pending_dt.astimezone(timezone.utc)
            oldest_pending_age_seconds = round(
                (datetime.now(timezone.utc) - pending_dt).total_seconds(),
                3,
            )
        logger.info(
            "Graph projection sync batch complete: events=%s processed=%s failed=%s oldest_pending_age_seconds=%s types=%s",
            [event.id for event in events],
            result.processed_count,
            result.failed_count,
            oldest_pending_age_seconds,
            [event.event_type for event in events],
        )
        return result.processed_count
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@celery_app.task(name="app.workers.sync_graph_projection_worker.rebuild_graph_projection_task")
def rebuild_graph_projection_task(user_id: str | None = None) -> int:
    """Rebuild the projected Neo4j graph from canonical Postgres state."""
    session = get_session_factory()()
    try:
        deps = build_worker_dependencies(session)
        graph_engine = deps["graph_engine"]
        rebuilt = graph_engine.rebuild_projection(user_id)
        logger.info("Graph projection rebuild complete: user_id=%s rebuilt=%s", user_id, rebuilt)
        return rebuilt
    finally:
        session.close()
