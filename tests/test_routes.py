"""Route contract tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routes import context as context_route
from app.api.routes import deep_memory as deep_memory_route
from app.api.routes import process as process_route
from app.api.routes import snapshot as snapshot_route
from app.api.schemas.common import RetrievalDiagnostics
from app.api.schemas.context import ContextResponse
from app.api.schemas.deep_memory import DeepMemoryResponse, EvidenceItem
from app.api.schemas.process import ProcessResponse
from app.api.schemas.snapshot import SnapshotPayload, SnapshotResponse
from app.core.enums import ReadMode, ScopeLevel
from app.db.session import get_session
from app.main import app


class DummySession:
    """Minimal session stub for route tests."""

    def commit(self) -> None:
        """No-op commit."""

    def close(self) -> None:
        """No-op close."""


class FakeContextEngine:
    """Fake context engine."""

    def build_context(self, request):  # noqa: ANN001
        return ContextResponse(
            status="ok",
            has_usable_context=False,
            context_enhancement="",
            abstained_reason="No grounded memory found.",
            diagnostics=RetrievalDiagnostics(
                scope_applied=ScopeLevel.USER_GLOBAL_CONTAINER,
                read_mode=ReadMode.BALANCED,
                user_found=True,
                candidate_count=3,
                used_memory_count=0,
                missing_containers=["missing-container"],
                source_breakdown={"vector": 3},
                evidence_strength=0.21,
                warnings=["missing_containers"],
            ),
        )

    def answer_deep_memory(self, request):  # noqa: ANN001
        return DeepMemoryResponse(
            status="ok",
            answer="I do not have enough grounded memory evidence to answer that confidently.",
            abstained=True,
            abstained_reason="Evidence strength was too low.",
            used_memory_count=1,
            diagnostics=RetrievalDiagnostics(
                scope_applied=ScopeLevel.USER_GLOBAL,
                read_mode=ReadMode.DEEP,
                user_found=True,
                candidate_count=4,
                used_memory_count=1,
                missing_containers=[],
                source_breakdown={"vector": 4},
                evidence_strength=0.33,
            ),
            evidence=[
                EvidenceItem(
                    memory_id="mem-1",
                    scope_type="user",
                    bucket_id=None,
                    relevance=0.87,
                    support=0.82,
                )
            ],
        )


class FakeProcessEngine:
    """Fake process engine."""

    def enqueue_turn(self, request):  # noqa: ANN001
        return ProcessResponse(
            status="accepted",
            job_id="job-123",
            created_user=True,
            created_containers=["thread-1"],
            accepted_at="2026-03-16T00:00:00Z",
        )


class FakeSnapshotEngine:
    """Fake snapshot engine."""

    def get_latest_snapshot(self, user_id: str) -> SnapshotResponse:
        return SnapshotResponse(
            status="ok",
            user_id=user_id,
            generated_at="2026-03-16T00:00:00Z",
            snapshot=SnapshotPayload(
                summary="snapshot summary",
                memory_refs=["mem-1"],
                health_note="healthy",
            ),
        )


def override_session():
    """Yield a dummy session."""
    yield DummySession()


def test_context_route_returns_diagnostics_and_abstention() -> None:
    app.dependency_overrides[context_route.get_engine] = lambda: FakeContextEngine()
    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    response = client.post(
        "/v1/context",
        json={
            "user_id": "user-1",
            "message": "What should I remember?",
            "containers": [{"id": "thread-1", "type": "thread"}],
            "scope_level": "user_global_container",
            "read_mode": "balanced",
            "budgets": {"max_output_tokens": 400, "max_candidate_memories": 30},
            "metadata": {"app": "test"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["abstained_reason"] == "No grounded memory found."
    assert payload["diagnostics"]["missing_containers"] == ["missing-container"]
    app.dependency_overrides.clear()


def test_process_route_returns_acceptance() -> None:
    app.dependency_overrides[process_route.get_engine] = lambda: FakeProcessEngine()
    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    response = client.post(
        "/v1/process",
        json={
            "user_id": "user-1",
            "containers": [{"id": "thread-1", "type": "thread"}],
            "scope_policy": {"write_user": "auto", "write_global": "auto", "write_container": True},
            "turn": {
                "user_message": "Hi",
                "assistant_response": "Hello",
                "occurred_at": "2026-03-16T00:00:00Z",
            },
            "metadata": {"app": "test"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["created_containers"] == ["thread-1"]
    app.dependency_overrides.clear()


def test_snapshot_route_returns_latest_snapshot() -> None:
    app.dependency_overrides[snapshot_route.get_engine] = lambda: FakeSnapshotEngine()
    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    response = client.get("/v1/snapshot/user-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["snapshot"]["health_note"] == "healthy"
    app.dependency_overrides.clear()


def test_deep_memory_route_returns_grounded_shape() -> None:
    app.dependency_overrides[deep_memory_route.get_engine] = lambda: FakeContextEngine()
    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    response = client.post(
        "/v1/deep-memory",
        json={
            "user_id": "user-1",
            "query": "What do you remember?",
            "containers": [],
            "scope_level": "user_global",
            "read_mode": "deep",
            "budgets": {"max_output_tokens": 500, "max_candidate_memories": 40},
            "metadata": {"trace_id": "abc"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["abstained"] is True
    assert payload["evidence"][0]["support"] == 0.82
    app.dependency_overrides.clear()
