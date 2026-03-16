"""Unit tests for pure service/config/prompt logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.core.config import Settings
from app.core.enums import ProviderKind, ScopeLevel
from app.db import runtime_migrations
from app.engines.graph_engine import Neo4jGraphClient
from app.llms.context_enhancer import ContextEnhancementPayload, DeepMemoryPayload
from app.llms.client import load_prompt
from app.retrieval.scope_resolver import ScopeResolver
from app.services.relevance_service import RelevanceService
from app.services.scoring_service import ScoringService
from app.services.tool_session_service import StagedGraphEntity


def test_scope_resolver_handles_all_levels() -> None:
    resolver = ScopeResolver()

    user_scope = resolver.resolve(ScopeLevel.USER, ["c1"], ["c1"])
    assert user_scope.scope_types == ["user"]

    user_global_scope = resolver.resolve(ScopeLevel.USER_GLOBAL, ["c1"], ["c1"])
    assert user_global_scope.scope_types == ["user", "global"]

    full_scope = resolver.resolve(ScopeLevel.USER_GLOBAL_CONTAINER, ["c1", "c2"], ["c1"])
    assert full_scope.scope_types == ["user", "global", "container"]
    assert full_scope.container_ids == ["c1"]
    assert full_scope.missing_containers == ["c2"]


def test_relevance_touch_updates_counts_and_scores() -> None:
    memory = SimpleNamespace(
        recall_count=2,
        updated_at=datetime.utcnow() - timedelta(days=2),
        importance_score=0.8,
        confidence_score=0.9,
        current_relevance_score=0.0,
        average_relevance_score=0.4,
        contradiction_risk=0.0,
        superseded_by_memory_id=None,
        last_recalled_at=None,
    )
    service = RelevanceService()

    updated = service.touch_memory(memory, query_similarity=0.9, graph_density_signal=0.2)

    assert updated.recall_count == 3
    assert updated.current_relevance_score > 0
    assert updated.average_relevance_score > 0.4
    assert updated.last_recalled_at is not None


def test_relevance_touch_handles_naive_updated_at_with_aware_now() -> None:
    memory = SimpleNamespace(
        recall_count=0,
        updated_at=datetime.utcnow(),
        importance_score=0.8,
        confidence_score=0.9,
        current_relevance_score=0.0,
        average_relevance_score=0.0,
        contradiction_risk=0.0,
        superseded_by_memory_id=None,
        last_recalled_at=None,
    )
    service = RelevanceService()

    updated = service.touch_memory(memory, query_similarity=0.8, now=datetime.now(timezone.utc))

    assert updated.recall_count == 1
    assert updated.last_recalled_at.tzinfo is not None


def test_scoring_service_maps_decay_thresholds() -> None:
    service = ScoringService()

    assert service.status_from_decay(0.8) == "active"
    assert service.status_from_decay(0.5) == "warm"
    assert service.status_from_decay(0.3) == "stale"
    assert service.status_from_decay(0.1) == "archived"


def test_provider_matrix_resolves_role_overrides() -> None:
    settings = Settings(
        _env_file=None,
        env="test",
        openai_default_base_url="https://default.example/v1",
        openai_default_api_key="default-key",
        adjudicator_provider=ProviderKind.OPENAI_COMPATIBLE,
        adjudicator_base_url="https://judge.example/v1",
        adjudicator_api_key="judge-key",
        adjudicator_model="judge-model",
        context_enhancer_provider=ProviderKind.STUB,
    )

    matrix = settings.resolve_provider_matrix()

    assert matrix.adjudicator.base_url == "https://judge.example/v1"
    assert matrix.adjudicator.api_key == "judge-key"
    assert matrix.adjudicator.model == "judge-model"
    assert matrix.context_enhancer.provider == ProviderKind.STUB
    assert matrix.embedding.base_url == "https://default.example/v1"


def test_prompt_loader_inlines_shared_sections() -> None:
    prompt = load_prompt("adjudicator")

    assert "Memory policy:" in prompt
    assert "Graph policy:" in prompt
    assert "Few-shot guidance:" in prompt
    assert "Search first, then stage writes." in prompt
    assert "finalize_adjudication" in prompt


def test_cortex_review_prompt_mentions_staged_review_contract() -> None:
    prompt = load_prompt("cortex_review")

    assert "Approve, skip, or adjust proposal items using staged review tools." in prompt
    assert "stage_status_update" in prompt
    assert "finalize_cortex_review" in prompt


def test_staged_graph_entity_accepts_common_llm_aliases() -> None:
    payload = StagedGraphEntity.model_validate(
        {
            "canonical_name": "pytest",
            "type": "testing_framework",
            "linked_memories": ["adjudication_1"],
            "attributes": {"category": "tool"},
        }
    )

    assert payload.entity_type == "testing_framework"
    assert payload.linked_memory_refs == ["adjudication_1"]


def test_staged_graph_entity_accepts_type_nested_in_attributes() -> None:
    payload = StagedGraphEntity.model_validate(
        {
            "canonical_name": "Project Cedar",
            "attributes": {"type": "project", "priority": "high"},
        }
    )

    assert payload.entity_type == "project"
    assert payload.attributes["priority"] == "high"


def test_graph_projection_serializes_map_fields_for_neo4j() -> None:
    serialized = Neo4jGraphClient._serialize_projection_maps(
        {
            "id": "entity-1",
            "attributes_json": {"preference": "pytest"},
            "aliases_json": ["pytest"],
        },
        ["attributes_json"],
    )

    assert serialized["attributes_json"] == '{"preference":"pytest"}'
    assert serialized["aliases_json"] == ["pytest"]


def test_context_enhancement_payload_accepts_null_string_fields() -> None:
    payload = ContextEnhancementPayload.model_validate(
        {
            "has_usable_context": False,
            "summary": None,
            "active_context": None,
            "confidence_note": None,
            "used_memory_ids": [],
        }
    )

    assert payload.summary == ""
    assert payload.active_context == ""
    assert payload.confidence_note == "Use as supportive context, not unquestionable fact."


def test_deep_memory_payload_accepts_null_answer() -> None:
    payload = DeepMemoryPayload.model_validate(
        {
            "answer": None,
            "used_memory_ids": [],
            "abstained": True,
        }
    )

    assert payload.answer == ""


def test_runtime_migration_normalizes_psycopg_urls() -> None:
    normalized = runtime_migrations.normalize_database_url("postgresql+psycopg://user:pass@db:5432/acmf")

    assert normalized == "postgresql://user:pass@db:5432/acmf"


def test_runtime_migration_skips_when_disabled(monkeypatch) -> None:
    settings = Settings(_env_file=None, env="development", auto_migrate_on_startup=False)
    called = {"run": 0}

    monkeypatch.setattr(runtime_migrations, "wait_for_database", lambda *args, **kwargs: called.__setitem__("wait", 1))
    monkeypatch.setattr(runtime_migrations, "run_migrations", lambda *args, **kwargs: called.__setitem__("run", 1))
    monkeypatch.setattr(runtime_migrations, "_migration_completed", False)

    runtime_migrations.maybe_run_startup_migrations(settings, wait_for_db=True)

    assert called["run"] == 0


def test_runtime_migration_runs_once(monkeypatch) -> None:
    settings = Settings(_env_file=None, env="development", auto_migrate_on_startup=True, db_startup_timeout_seconds=5.0)
    called = {"wait": 0, "run": 0}

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(runtime_migrations, "wait_for_database", lambda *args, **kwargs: called.__setitem__("wait", called["wait"] + 1))
    monkeypatch.setattr(runtime_migrations, "run_migrations", lambda *args, **kwargs: called.__setitem__("run", called["run"] + 1))
    monkeypatch.setattr(runtime_migrations, "_migration_completed", False)

    runtime_migrations.maybe_run_startup_migrations(settings, wait_for_db=True)
    runtime_migrations.maybe_run_startup_migrations(settings, wait_for_db=True)

    assert called["wait"] == 1
    assert called["run"] == 1
