"""Context-engine relevance gating tests."""

from __future__ import annotations

from types import SimpleNamespace

from app.api.schemas.context import ContextRequest
from app.api.schemas.deep_memory import DeepMemoryRequest
from app.core.config import Settings
from app.core.enums import ReadMode, ScopeLevel
from app.engines.context_engine import ContextEngine
from app.llms.context_enhancer import ContextEnhancementPayload, DeepMemoryPayload
from app.retrieval.graph_search import GraphTraversalResult
from app.retrieval.reranker import Reranker
from app.retrieval.scope_resolver import ScopeResolver
from app.retrieval.vector_search import VectorSearchHit


class FakeUserRepo:
    """Minimal user repo for context-engine tests."""

    def get(self, user_id: str):  # noqa: ANN001
        return object() if user_id == "user-1" else None


class FakeContainerRepo:
    """Minimal container repo for context-engine tests."""

    def get_existing_ids(self, user_id: str, containers: list[str]) -> list[str]:  # noqa: ARG002
        return containers


class FakeSnapshotRepo:
    """Minimal snapshot repo for context-engine tests."""

    def __init__(self, memory_refs: list[str] | None = None) -> None:
        self.memory_refs = memory_refs or []

    def get_latest(self, user_id: str):  # noqa: ARG002
        if not self.memory_refs:
            return None
        return SimpleNamespace(memory_refs=self.memory_refs)


class FakeMemoryRepo:
    """Minimal memory repo for context-engine tests."""

    def __init__(self, memories: dict[str, object], metadata_ids: list[str] | None = None) -> None:
        self.memories = memories
        self.metadata_ids = metadata_ids or []

    def get_by_ids(self, memory_ids: list[str]):
        return [self.memories[memory_id] for memory_id in memory_ids if memory_id in self.memories]

    def list_high_signal_candidates(self, user_id: str, scope_types: list[str], bucket_ids: list[str], limit: int):  # noqa: ARG002
        return self.get_by_ids(self.metadata_ids[:limit])

    def summarize_open_contradictions(self, user_id: str, limit: int = 8):  # noqa: ARG002
        return []

    def summarize_lineage(self, user_id: str, limit: int = 8):  # noqa: ARG002
        return []


class FakeVectorSearch:
    """Vector search stub with score support."""

    def __init__(self, hits: list[VectorSearchHit]) -> None:
        self.hits = hits

    def search_with_scores(self, *, user_id: str, scope_types: list[str], bucket_ids: list[str], query: str, limit: int):  # noqa: ARG002
        return self.hits[:limit]


class FakeMemoryService:
    """Track memory touch calls."""

    def __init__(self) -> None:
        self.touch_calls = 0

    def touch_memories(self, memories, query_similarity_map, graph_density_signal=0.0) -> None:  # noqa: ANN001, ARG002
        self.touch_calls += 1


class FakeGraphEngine:
    """Graph traversal stub."""

    def __init__(self, result: GraphTraversalResult | None = None) -> None:
        self.result = result or GraphTraversalResult()
        self.calls: list[tuple[str, str, int, list[str] | None]] = []

    def traverse(self, user_id: str, query: str, limit: int, seed_memory_ids=None):  # noqa: ANN001
        self.calls.append((user_id, query, limit, seed_memory_ids))
        return self.result


class FakeEnhancer:
    """Enhancer stub with call counters."""

    def __init__(self) -> None:
        self.context_calls = 0
        self.deep_calls = 0

    def synthesize_context(self, scope_level: ScopeLevel, context):  # noqa: ANN001, ARG002
        self.context_calls += 1
        memory_id = context.memories[0].id if context.memories else None
        return ContextEnhancementPayload(
            has_usable_context=True,
            summary="Relevant preference found.",
            active_context="The user prefers pytest.",
            used_memory_ids=[memory_id] if memory_id else [],
        )

    def answer_deep_memory(self, context):  # noqa: ANN001
        self.deep_calls += 1
        memory_id = context.memories[0].id if context.memories else None
        return DeepMemoryPayload(
            answer="The user prefers pytest.",
            used_memory_ids=[memory_id] if memory_id else [],
            abstained=False,
        )


def build_memory(memory_id: str, summary: str, content: str) -> object:
    """Return a memory-shaped test object."""
    return SimpleNamespace(
        id=memory_id,
        scope_type="user",
        bucket_id=None,
        memory_type="preference",
        status="active",
        importance_score=0.8,
        confidence_score=0.9,
        current_relevance_score=0.5,
        average_relevance_score=0.5,
        contradiction_risk=0.0,
        superseded_by_memory_id=None,
        summary=summary,
        content=content,
        rationale=None,
    )


def build_engine(
    *,
    memories: dict[str, object],
    vector_hits: list[VectorSearchHit],
    metadata_ids: list[str] | None = None,
    snapshot_ids: list[str] | None = None,
    graph_result: GraphTraversalResult | None = None,
):
    """Construct a context engine with test doubles."""
    graph_engine = FakeGraphEngine(graph_result)
    enhancer = FakeEnhancer()
    memory_service = FakeMemoryService()
    engine = ContextEngine(
        settings=Settings(_env_file=None, env="test"),
        user_repo=FakeUserRepo(),
        container_repo=FakeContainerRepo(),
        memory_repo=FakeMemoryRepo(memories, metadata_ids=metadata_ids),
        snapshot_repo=FakeSnapshotRepo(snapshot_ids),
        vector_search=FakeVectorSearch(vector_hits),
        scope_resolver=ScopeResolver(),
        reranker=Reranker(),
        memory_service=memory_service,
        graph_engine=graph_engine,
        enhancer=enhancer,
    )
    return engine, enhancer, graph_engine, memory_service


def test_context_unrelated_query_returns_zero_relevant_candidates() -> None:
    pytest_memory = build_memory("mem-1", "Preference for pytest", "The user prefers pytest over unittest.")
    project_memory = build_memory("mem-2", "Neptune migration task", "Migrate Project Neptune to the new stack.")
    engine, enhancer, graph_engine, memory_service = build_engine(
        memories={"mem-1": pytest_memory, "mem-2": project_memory},
        vector_hits=[
            VectorSearchHit(memory=pytest_memory, similarity=0.41),
            VectorSearchHit(memory=project_memory, similarity=0.33),
        ],
        metadata_ids=["mem-1", "mem-2"],
    )

    response = engine.build_context(
        ContextRequest(
            user_id="user-1",
            message="Hows the weather",
            containers=[{"id": "thread-1", "type": "thread"}],
            scope_level=ScopeLevel.USER_GLOBAL_CONTAINER,
            read_mode=ReadMode.BALANCED,
        )
    )

    assert response.status == "ok"
    assert response.has_usable_context is False
    assert response.context_enhancement == ""
    assert response.abstained_reason == "No relevant grounded memory was found for this query."
    assert response.diagnostics.candidate_count == 0
    assert response.diagnostics.used_memory_count == 0
    assert response.diagnostics.source_breakdown == {}
    assert enhancer.context_calls == 0
    assert graph_engine.calls == []
    assert memory_service.touch_calls == 0


def test_context_relevant_query_calls_enhancer_and_reports_relevant_counts() -> None:
    pytest_memory = build_memory("mem-1", "Preference for pytest", "The user prefers pytest over unittest.")
    graph_result = GraphTraversalResult(facts=[], memory_ids=["mem-1"], graph_density_signal=0.2)
    engine, enhancer, graph_engine, memory_service = build_engine(
        memories={"mem-1": pytest_memory},
        vector_hits=[VectorSearchHit(memory=pytest_memory, similarity=0.88)],
        metadata_ids=["mem-1"],
        graph_result=graph_result,
    )

    response = engine.build_context(
        ContextRequest(
            user_id="user-1",
            message="What do I know about pytest?",
            containers=[],
            scope_level=ScopeLevel.USER_GLOBAL,
            read_mode=ReadMode.BALANCED,
        )
    )

    assert response.status == "ok"
    assert response.has_usable_context is True
    assert "<contextenhancement>" in response.context_enhancement
    assert response.diagnostics.candidate_count == 1
    assert response.diagnostics.used_memory_count == 1
    assert response.diagnostics.source_breakdown == {"vector": 1, "metadata": 1, "graph_memories": 1}
    assert enhancer.context_calls == 1
    assert graph_engine.calls[0][3] == ["mem-1"]
    assert memory_service.touch_calls == 1


def test_deep_memory_unrelated_query_abstains_without_llm_call() -> None:
    pytest_memory = build_memory("mem-1", "Preference for pytest", "The user prefers pytest over unittest.")
    engine, enhancer, graph_engine, memory_service = build_engine(
        memories={"mem-1": pytest_memory},
        vector_hits=[VectorSearchHit(memory=pytest_memory, similarity=0.4)],
        metadata_ids=["mem-1"],
    )

    response = engine.answer_deep_memory(
        DeepMemoryRequest(
            user_id="user-1",
            query="What is the weather",
            containers=[],
            scope_level=ScopeLevel.USER_GLOBAL,
            read_mode=ReadMode.DEEP,
        )
    )

    assert response.status == "ok"
    assert response.abstained is True
    assert response.abstained_reason == "No relevant grounded memory was found for this query."
    assert response.used_memory_count == 0
    assert response.diagnostics.candidate_count == 0
    assert response.diagnostics.source_breakdown == {}
    assert response.evidence == []
    assert enhancer.deep_calls == 0
    assert graph_engine.calls == []
    assert memory_service.touch_calls == 0
