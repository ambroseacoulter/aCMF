"""Celery application configuration."""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab
from celery.signals import beat_init, worker_init

from app.core.config import get_settings
from app.db.runtime_migrations import maybe_run_startup_migrations

settings = get_settings()

celery_app = Celery(
    "acmf",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.workers.process_turn_worker",
        "app.workers.hourly_cortex_worker",
        "app.workers.sync_graph_projection_worker",
    ],
)
celery_app.conf.update(
    timezone="UTC",
    task_always_eager=settings.celery_task_always_eager,
    beat_schedule={
        "hourly-cortex": {
            "task": "app.workers.hourly_cortex_worker.hourly_cortex_task",
            "schedule": crontab(minute=0),
        },
        "sync-graph-projection": {
            "task": "app.workers.sync_graph_projection_worker.sync_graph_projection_task",
            "schedule": settings.graph_projection_retry_backoff_seconds,
        },
    },
)


@worker_init.connect
def _run_worker_startup_migrations(**_: object) -> None:
    """Apply pending migrations before the worker starts processing tasks."""
    maybe_run_startup_migrations(settings, wait_for_db=False)


@beat_init.connect
def _run_beat_startup_migrations(**_: object) -> None:
    """Apply pending migrations before beat starts scheduling tasks."""
    maybe_run_startup_migrations(settings, wait_for_db=False)
