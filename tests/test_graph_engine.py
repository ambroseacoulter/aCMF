"""Graph engine and projection hardening tests."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from app.core.config import Settings
from app.core.enums import OutboxStatus, ProjectionEventType
from app.engines import graph_engine as graph_engine_module
from app.engines.graph_engine import GraphEngine, Neo4jGraphClient, StubGraphClient
from app.services.graph_projection_service import GraphProjectionNormalizer


class FakeRunSession:
    """Capture Neo4j session calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **kwargs) -> None:
        self.calls.append((query, kwargs))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeDriver:
    """Return a reusable fake session."""

    def __init__(self) -> None:
        self.session_instance = FakeRunSession()

    def session(self) -> FakeRunSession:
        return self.session_instance


class FakeTraverseSession:
    """Neo4j session stub for traversal tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        if "RETURN m.id AS memory_id" in query or "RETURN DISTINCT neighbor.id AS memory_id" in query:
            return [{"memory_id": "mem-1"}]
        return [
            {
                "source_name": "memory source",
                "relation_type": "RELATED_TO",
                "target_name": "neighbor entity",
                "confidence": 0.75,
                "source_memory_id": "mem-1",
                "target_memory_id": None,
            }
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeTraverseDriver:
    """Driver stub for traversal tests."""

    def __init__(self) -> None:
        self.session_instance = FakeTraverseSession()

    def session(self) -> FakeTraverseSession:
        return self.session_instance


class FakeProjectionClient:
    """Projection client stub with optional failure modes."""

    def __init__(self) -> None:
        self.cleared: list[str] = []
        self.memory_nodes: list[tuple[str, list[dict]]] = []
        self.entities: list[tuple[str, list[dict]]] = []
        self.relations: list[tuple[str, list[dict]]] = []
        self.memory_links: list[tuple[str, list[dict]]] = []
        self.graph_edges: list[tuple[str, list[dict]]] = []
        self.fail_event_type: str | None = None

    def clear_user_projection(self, user_id: str) -> None:
        self.cleared.append(user_id)

    def upsert_memory_nodes(self, user_id: str, memory_nodes: list[dict]) -> None:
        if self.fail_event_type == ProjectionEventType.MEMORY_NODES_UPSERTED.value:
            raise RuntimeError("memory projection failed")
        self.memory_nodes.append((user_id, memory_nodes))

    def upsert_entities(self, user_id: str, entities: list[dict]) -> None:
        if self.fail_event_type == ProjectionEventType.ENTITIES_UPSERTED.value:
            raise RuntimeError("entity projection failed")
        self.entities.append((user_id, entities))

    def upsert_relations(self, user_id: str, relations: list[dict]) -> None:
        if self.fail_event_type == ProjectionEventType.RELATIONS_UPSERTED.value:
            raise RuntimeError("relation projection failed")
        self.relations.append((user_id, relations))

    def upsert_memory_links(self, user_id: str, memory_links: list[dict]) -> None:
        if self.fail_event_type == ProjectionEventType.MEMORY_LINKS_UPSERTED.value:
            raise RuntimeError("memory link projection failed")
        self.memory_links.append((user_id, memory_links))

    def upsert_graph_edges(self, user_id: str, graph_edges: list[dict]) -> None:
        if self.fail_event_type == ProjectionEventType.GRAPH_EDGES_UPSERTED.value:
            raise RuntimeError("graph edge projection failed")
        self.graph_edges.append((user_id, graph_edges))


class FakeGraphRepo:
    """Canonical graph repo stub for rebuild tests."""

    def list_projection_user_ids(self) -> list[str]:
        return ["user-1"]

    def get_user_memory_projection_payloads(self, user_id: str) -> list[dict]:
        return [
            {
                "id": "mem-1",
                "user_id": user_id,
                "scope_type": "user",
                "bucket_id": None,
                "memory_type": "preference",
                "status": "active",
                "importance_score": 0.8,
                "confidence_score": 0.9,
                "novelty_score": 0.6,
                "current_relevance_score": 0.7,
                "average_relevance_score": 0.7,
                "contradiction_risk": 0.1,
                "recall_count": 2,
                "decay_score": 0.9,
                "summary": "Prefers pytest",
                "content": "User prefers pytest over unittest",
                "rationale": "Durable stated preference",
                "evidence_json": {"items": ["prefers pytest"]},
                "created_at": "2026-03-16T00:00:00Z",
                "updated_at": "2026-03-16T00:00:00Z",
            }
        ]

    def get_entity_projection_payloads(self, user_id: str) -> list[dict]:
        return [
            {
                "id": "entity-1",
                "entity_type": "tool",
                "canonical_name": "pytest",
                "aliases_json": ["pytest"],
                "attributes_json": {"category": "testing"},
            }
        ]

    def get_relation_projection_payloads(self, user_id: str) -> list[dict]:
        return [
            {
                "id": "rel-1",
                "from_entity_id": "entity-1",
                "to_entity_id": "entity-2",
                "from_entity_name": "pytest",
                "to_entity_name": "unittest",
                "relation_type": "PREFERS",
                "confidence_score": 0.9,
                "attributes_json": {"strength": "high"},
                "evidence_json": {"items": ["prefers pytest"]},
            }
        ]

    def get_memory_link_projection_payloads(self, user_id: str) -> list[dict]:
        return [
            {
                "id": "mem-1:entity-1:MENTIONS",
                "memory_id": "mem-1",
                "entity_id": "entity-1",
                "link_type": "MENTIONS",
                "entity_name": "pytest",
                "memory_summary": "Prefers pytest",
            }
        ]

    def get_graph_edge_projection_payloads(self, user_id: str) -> list[dict]:
        return [
            {
                "id": "edge-1",
                "from_node_type": "memory",
                "from_node_id": "mem-1",
                "to_node_type": "entity",
                "to_node_id": "entity-1",
                "edge_type": "MENTIONS",
                "confidence_score": 0.8,
                "attributes_json": {},
                "source_type": "memory_entity_link",
                "source_ref": "mem-1:entity-1",
            }
        ]


def test_graph_projection_normalizer_builds_searchable_fields() -> None:
    normalizer = GraphProjectionNormalizer()

    payload = normalizer.normalize_entity(
        {
            "id": "entity-1",
            "entity_type": "tool",
            "canonical_name": "pytest",
            "aliases_json": ["py.test"],
            "attributes_json": {"category": "testing", "framework": "python"},
        }
    )

    assert "pytest" in payload["search_text"]
    assert "category testing" in payload["search_text"]
    assert payload["attributes_json"] == '{"category":"testing","framework":"python"}'


def test_sync_projection_marks_failed_after_retry_budget() -> None:
    projection_client = FakeProjectionClient()
    projection_client.fail_event_type = ProjectionEventType.MEMORY_LINKS_UPSERTED.value
    engine = GraphEngine(FakeGraphRepo(), projection_client, StubGraphClient())
    event = SimpleNamespace(
        id="evt-1",
        event_type=ProjectionEventType.MEMORY_LINKS_UPSERTED.value,
        user_id="user-1",
        payload_json={"memory_links": [{"id": "link-1"}]},
        attempt_count=0,
        status=OutboxStatus.PENDING.value,
        error_message=None,
        processed_at=None,
        created_at=datetime.utcnow(),
    )

    first = engine.sync_projection([event], max_attempts=2)
    second = engine.sync_projection([event], max_attempts=2)

    assert first.failed_count == 1
    assert event.status == OutboxStatus.FAILED.value
    assert second.failed_count == 1
    assert event.attempt_count == 2
    assert event.error_message == "memory link projection failed"


def test_rebuild_projection_clears_and_replays_user_graph() -> None:
    projection_client = FakeProjectionClient()
    engine = GraphEngine(FakeGraphRepo(), projection_client, StubGraphClient())

    rebuilt = engine.rebuild_projection("user-1")

    assert rebuilt == 1
    assert projection_client.cleared == ["user-1"]
    assert projection_client.memory_nodes[0][1][0]["search_text"]
    assert projection_client.entities[0][1][0]["search_text"]
    assert projection_client.relations[0][1][0]["search_text"]
    assert projection_client.graph_edges[0][1][0]["search_text"]


def test_neo4j_memory_link_projection_uses_valid_with_clauses(monkeypatch) -> None:
    driver = FakeDriver()
    monkeypatch.setattr(graph_engine_module.GraphDatabase, "driver", lambda *args, **kwargs: driver)
    client = Neo4jGraphClient(Settings(_env_file=None, env="development"))

    client.upsert_memory_links(
        "user-1",
        [{"id": "link-1", "memory_id": "mem-1", "entity_id": "entity-1", "link_type": "MENTIONS", "search_text": "pytest"}],
    )

    query, params = driver.session_instance.calls[0]
    assert "WITH link, m" in query
    assert "WITH link, m, e" in query
    assert "MERGE (m)-[r:MENTIONS {id: link.id}]->(e)" in query
    assert params["memory_links"][0]["id"] == "link-1"


def test_neo4j_traversal_uses_search_query_parameter_without_keyword_collision(monkeypatch) -> None:
    driver = FakeTraverseDriver()
    monkeypatch.setattr(graph_engine_module.GraphDatabase, "driver", lambda *args, **kwargs: driver)
    client = Neo4jGraphClient(Settings(_env_file=None, env="development"))

    result = client.traverse_context("user-1", "pytest", 5, seed_memory_ids=["mem-1"])

    assert result.memory_ids == ["mem-1"]
    assert len(result.facts) == 1
    first_query, first_params = driver.session_instance.calls[0]
    second_query, second_params = driver.session_instance.calls[1]
    assert "$search_query" in first_query
    assert "$search_query" in second_query
    assert first_params["search_query"] == "pytest"
    assert second_params["search_query"] == "pytest"
