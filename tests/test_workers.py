"""Worker orchestration tests."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field

from app.engines.graph_engine import ProjectionSyncResult
from app.workers import hourly_cortex_worker, process_turn_worker, sync_graph_projection_worker


class FakeSession:
    """Minimal session stub."""

    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


class FakeSessionFactory:
    """Callable session factory."""

    def __init__(self, session: FakeSession) -> None:
        self.session = session

    def __call__(self) -> FakeSession:
        return self.session


@dataclass
class FakeJob:
    """Job payload holder."""

    id: str = "job-1"
    payload_json: dict = field(
        default_factory=lambda: {
            "user_id": "user-1",
            "containers": [{"id": "thread-1", "type": "thread"}],
            "scope_policy": {"write_user": "auto", "write_global": "auto", "write_container": True},
            "turn": {
                "user_message": "Remember I prefer pytest",
                "assistant_response": "Noted",
                "occurred_at": None,
                "user_message_id": None,
                "assistant_message_id": None,
            },
            "metadata": {},
        }
    )


class FakeJobRepo:
    """Job repo stub."""

    def __init__(self) -> None:
        self.failed_with: str | None = None
        self.running = False
        self.succeeded = False
        self.job = FakeJob()

    def mark_running(self, job_id: str) -> None:
        self.running = True

    def get(self, job_id: str) -> FakeJob:
        return self.job

    def mark_succeeded(self, job_id: str, notes=None) -> None:  # noqa: ANN001
        self.succeeded = True

    def mark_failed(self, job_id: str, error_message: str) -> None:
        self.failed_with = error_message


class FakeUserRepo:
    """User repo stub."""

    def __init__(self) -> None:
        self.snapshot_dirty = False
        self.user_ids = ["user-1", "user-2"]

    def get_or_create(self, user_id: str):
        return object(), False

    def mark_snapshot_dirty(self, user_id: str, dirty: bool) -> None:
        self.snapshot_dirty = dirty

    def get_all_ids(self) -> list[str]:
        return self.user_ids


class FakeMemory:
    """Persisted memory stub."""

    def __init__(self, memory_id: str) -> None:
        self.id = memory_id


class FakeMemoryRepo:
    """Memory repo stub."""

    def list_recent_candidates(self, user_id: str, scope_types: list[str], container_ids: list[str], limit: int):
        return []

    def summarize_open_contradictions(self, user_id: str, limit: int = 8):
        return []

    def summarize_lineage(self, user_id: str, limit: int = 8):
        return []

    def count_user_memories(self, user_id: str) -> int:
        return 1


class FakeMemoryService:
    """Memory service stub."""

    def __init__(self) -> None:
        self.persisted = [FakeMemory("mem-1")]
        self.touched = False

    def apply_adjudication_session(self, **kwargs):  # noqa: ANN003
        return type(
            "Result",
            (),
            {
                "created_memories": self.persisted,
                "touched_memories": [],
                "contradiction_topics": [],
                "graph_entities": [],
                "graph_relations": [],
                "memory_links": [],
                "graph_edges": [],
            },
        )()

    def touch_memories(self, memories, query_similarity_map, graph_density_signal=0.0) -> None:
        self.touched = True


class FakeAdjudicator:
    """Adjudicator stub."""

    def run_with_tools(self, context, executor):  # noqa: ANN001
        executor.execute("search_memory", {"query": context.subject, "limit": 5})
        executor.execute(
            "stage_create_memory",
            {
                "memory_type": "preference",
                "content": "User prefers pytest",
                "summary": "Preference for pytest",
                "rationale": "Durable testing preference.",
                "evidence": ["Remember I prefer pytest"],
                "importance_score": 0.8,
                "confidence_score": 0.9,
                "novelty_score": 0.7,
                "initial_relevance_score": 0.75,
                "contradiction_risk": 0.1,
                "target_scopes": ["user"],
            },
        )
        executor.execute("finalize_adjudication", {"reasoning_summary": "Created one durable preference memory."})
        return type(
            "Result",
            (),
            {
                "reasoning_summary": "Created one durable preference memory.",
                "tool_call_count": 3,
                "tool_session": executor.session,
            },
        )()


class FakeGraphEngine:
    """Graph engine stub."""

    def __init__(self) -> None:
        self.persist_called = False
        self.synced_events = None

    def traverse(self, user_id: str, query: str, limit: int, seed_memory_ids=None):
        return type("GraphResult", (), {"facts": [], "graph_density_signal": 0.1})()

    def persist_canonical_graph(self, **kwargs) -> None:
        self.persist_called = True
        return type("Mutation", (), {"projection_event_ids": ["evt-1", "evt-2"]})()

    def sync_projection(self, events, *, max_attempts: int = 5):
        self.synced_events = events
        return ProjectionSyncResult(
            processed_count=len(events),
            failed_count=0,
            processed_event_ids=["evt-{0}".format(index) for index, _event in enumerate(events)],
            failed_event_ids=[],
        )

    def rebuild_projection(self, user_id: str | None = None) -> int:
        return 1


class FakeGraphRepo:
    """Graph repo stub."""

    def __init__(self) -> None:
        self.events = [
            type("Event", (), {"id": "evt-1", "event_type": "memory_nodes_upserted"})(),
            type("Event", (), {"id": "evt-2", "event_type": "graph_edges_upserted"})(),
        ]

    def get_pending_outbox_events(self, limit: int = 100):
        return self.events

    def get_outbox_events_by_ids(self, event_ids: list[str]):
        return self.events[: len(event_ids)]

    def get_oldest_pending_outbox_created_at(self):
        return datetime.utcnow()


class FakeTurnRepo:
    """Turn repo stub."""

    def __init__(self) -> None:
        self.record = type("TurnRecord", (), {"id": "turn-1"})()
        self.processed = False

    def create(self, payload: dict):
        return self.record

    def mark_processed(self, turn_id: str, notes=None) -> None:  # noqa: ANN001
        self.processed = True

    def mark_failed(self, turn_id: str, error_message: str) -> None:
        self.processed = False


class FakeCortexEngine:
    """Cortex engine stub."""

    def __init__(self) -> None:
        self.processed: list[str] = []

    def run_hourly(self, user_id: str) -> list[str]:
        self.processed.append(user_id)
        return [f"evt-{user_id}"]


class FakeVectorSearch:
    """Vector search stub."""

    def search(self, user_id: str, scope_types: list[str], bucket_ids: list[str], query: str, limit: int):  # noqa: ANN001
        return []


def test_process_turn_task_marks_snapshot_dirty(monkeypatch) -> None:
    session = FakeSession()
    job_repo = FakeJobRepo()
    user_repo = FakeUserRepo()
    memory_service = FakeMemoryService()
    graph_engine = FakeGraphEngine()
    turn_repo = FakeTurnRepo()

    dispatched: list[list[str]] = []

    monkeypatch.setattr(process_turn_worker, "get_session_factory", lambda: FakeSessionFactory(session))
    monkeypatch.setattr(
        process_turn_worker,
        "build_worker_dependencies",
        lambda _session: {
            "job_repo": job_repo,
            "turn_repo": turn_repo,
            "user_repo": user_repo,
            "memory_service": memory_service,
            "graph_engine": graph_engine,
            "vector_search": FakeVectorSearch(),
            "adjudicator": FakeAdjudicator(),
            "memory_repo": FakeMemoryRepo(),
        },
    )
    monkeypatch.setattr(process_turn_worker, "dispatch_graph_projection_task", lambda event_ids=None: dispatched.append(event_ids or []))

    process_turn_worker.process_turn_task("job-1")

    assert job_repo.running is True
    assert job_repo.succeeded is True
    assert user_repo.snapshot_dirty is True
    assert graph_engine.persist_called is True
    assert turn_repo.processed is True
    assert session.committed is True
    assert dispatched == [["evt-1", "evt-2"]]


def test_sync_graph_projection_task_processes_pending_events(monkeypatch) -> None:
    session = FakeSession()
    graph_repo = FakeGraphRepo()
    graph_engine = FakeGraphEngine()

    monkeypatch.setattr(sync_graph_projection_worker, "get_session_factory", lambda: FakeSessionFactory(session))
    monkeypatch.setattr(
        sync_graph_projection_worker,
        "build_worker_dependencies",
        lambda _session: {
            "settings": type("Settings", (), {"graph_projection_batch_size": 100, "graph_projection_max_attempts": 5})(),
            "graph_repo": graph_repo,
            "graph_engine": graph_engine,
        },
    )

    processed = sync_graph_projection_worker.sync_graph_projection_task()

    assert processed == 2
    assert graph_engine.synced_events == graph_repo.events
    assert session.committed is True


def test_sync_graph_projection_task_can_target_specific_events(monkeypatch) -> None:
    session = FakeSession()
    graph_repo = FakeGraphRepo()
    graph_engine = FakeGraphEngine()

    monkeypatch.setattr(sync_graph_projection_worker, "get_session_factory", lambda: FakeSessionFactory(session))
    monkeypatch.setattr(
        sync_graph_projection_worker,
        "build_worker_dependencies",
        lambda _session: {
            "settings": type("Settings", (), {"graph_projection_batch_size": 100, "graph_projection_max_attempts": 5})(),
            "graph_repo": graph_repo,
            "graph_engine": graph_engine,
        },
    )

    processed = sync_graph_projection_worker.sync_graph_projection_task(["evt-1"])

    assert processed == 1
    assert graph_engine.synced_events == graph_repo.events[:1]
    assert session.committed is True


def test_rebuild_graph_projection_task_runs(monkeypatch) -> None:
    session = FakeSession()
    graph_engine = FakeGraphEngine()

    monkeypatch.setattr(sync_graph_projection_worker, "get_session_factory", lambda: FakeSessionFactory(session))
    monkeypatch.setattr(
        sync_graph_projection_worker,
        "build_worker_dependencies",
        lambda _session: {"graph_engine": graph_engine},
    )

    rebuilt = sync_graph_projection_worker.rebuild_graph_projection_task("user-1")

    assert rebuilt == 1
    assert session.closed is True


def test_hourly_cortex_task_runs_for_all_users(monkeypatch) -> None:
    session = FakeSession()
    user_repo = FakeUserRepo()
    cortex_engine = FakeCortexEngine()

    dispatched: list[list[str]] = []

    monkeypatch.setattr(hourly_cortex_worker, "get_session_factory", lambda: FakeSessionFactory(session))
    monkeypatch.setattr(
        hourly_cortex_worker,
        "build_worker_dependencies",
        lambda _session: {"user_repo": user_repo, "cortex_engine": cortex_engine},
    )
    monkeypatch.setattr(hourly_cortex_worker, "dispatch_graph_projection_task", lambda event_ids=None: dispatched.append(event_ids or []))

    processed = hourly_cortex_worker.hourly_cortex_task()

    assert processed == 2
    assert cortex_engine.processed == ["user-1", "user-2"]
    assert session.committed is True
    assert dispatched == [["evt-user-1", "evt-user-2"]]
