"""Dependency wiring for API routes and workers."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.engines.context_engine import ContextEngine
from app.engines.cortex_engine import CortexEngine
from app.engines.graph_engine import GraphEngine, Neo4jGraphClient, StubGraphClient
from app.engines.process_engine import ProcessEngine
from app.engines.snapshot_engine import SnapshotEngine
from app.llms.adjudicator import AdjudicatorLLM
from app.llms.client import resolve_embedding_client, resolve_llm_client
from app.llms.context_enhancer import ContextEnhancerLLM
from app.llms.cortex import CortexLLM
from app.retrieval.reranker import Reranker
from app.retrieval.scope_resolver import ScopeResolver
from app.retrieval.vector_search import VectorSearch
from app.services.embedding_service import EmbeddingService
from app.services.maintenance_service import MaintenanceService
from app.services.memory_service import MemoryService
from app.services.relevance_service import RelevanceService
from app.services.scoring_service import ScoringService
from app.storage.container_repo import ContainerRepository
from app.storage.graph_repo import GraphRepository
from app.storage.job_repo import JobRepository
from app.storage.memory_repo import MemoryRepository
from app.storage.snapshot_repo import SnapshotRepository
from app.storage.turn_repo import TurnRepository
from app.storage.user_repo import UserRepository


def resolve_graph_client():
    """Create the configured graph client."""
    settings = get_settings()
    if settings.env == "test":
        return StubGraphClient()
    return Neo4jGraphClient(settings)


def build_embedding_service() -> EmbeddingService:
    """Create the embedding service."""
    settings = get_settings()
    matrix = settings.resolve_provider_matrix()
    return EmbeddingService(resolve_embedding_client(matrix.embedding), matrix.embedding.model)


def build_process_engine(session: Session) -> ProcessEngine:
    """Construct the process engine for a session."""
    return ProcessEngine(
        user_repo=UserRepository(session),
        container_repo=ContainerRepository(session),
        job_repo=JobRepository(session),
    )


def build_snapshot_engine(session: Session) -> SnapshotEngine:
    """Construct the snapshot engine for a session."""
    return SnapshotEngine(user_repo=UserRepository(session), snapshot_repo=SnapshotRepository(session))


def build_context_engine(session: Session) -> ContextEngine:
    """Construct the context engine for a session."""
    settings = get_settings()
    matrix = settings.resolve_provider_matrix()
    graph_client = resolve_graph_client()
    user_repo = UserRepository(session)
    container_repo = ContainerRepository(session)
    memory_repo = MemoryRepository(session)
    snapshot_repo = SnapshotRepository(session)
    embedding_service = build_embedding_service()
    graph_repo = GraphRepository(session)
    relevance_service = RelevanceService()
    memory_service = MemoryService(memory_repo, embedding_service, relevance_service, matrix.embedding.model)
    graph_engine = GraphEngine(graph_repo, graph_client, graph_client)
    return ContextEngine(
        settings=settings,
        user_repo=user_repo,
        container_repo=container_repo,
        memory_repo=memory_repo,
        snapshot_repo=snapshot_repo,
        vector_search=VectorSearch(memory_repo, embedding_service),
        scope_resolver=ScopeResolver(),
        reranker=Reranker(),
        memory_service=memory_service,
        graph_engine=graph_engine,
        enhancer=ContextEnhancerLLM(resolve_llm_client("context_enhancer", matrix.context_enhancer)),
    )


def build_cortex_engine(session: Session) -> CortexEngine:
    """Construct the cortex engine for a session."""
    settings = get_settings()
    matrix = settings.resolve_provider_matrix()
    graph_client = resolve_graph_client()
    memory_repo = MemoryRepository(session)
    embedding_service = build_embedding_service()
    return CortexEngine(
        settings=settings,
        user_repo=UserRepository(session),
        memory_repo=memory_repo,
        snapshot_repo=SnapshotRepository(session),
        scoring_service=ScoringService(),
        maintenance_service=MaintenanceService(ScoringService()),
        memory_service=MemoryService(memory_repo, embedding_service, RelevanceService(), matrix.embedding.model),
        graph_engine=GraphEngine(GraphRepository(session), graph_client, graph_client),
        cortex_llm=CortexLLM(resolve_llm_client("cortex", matrix.cortex)),
    )


def build_worker_dependencies(session: Session) -> dict[str, object]:
    """Construct shared worker dependencies for a session."""
    settings = get_settings()
    matrix = settings.resolve_provider_matrix()
    graph_client = resolve_graph_client()
    graph_repo = GraphRepository(session)
    memory_repo = MemoryRepository(session)
    embedding_service = build_embedding_service()
    return {
        "settings": settings,
        "user_repo": UserRepository(session),
        "container_repo": ContainerRepository(session),
        "memory_repo": memory_repo,
        "snapshot_repo": SnapshotRepository(session),
        "job_repo": JobRepository(session),
        "turn_repo": TurnRepository(session),
        "graph_repo": graph_repo,
        "graph_engine": GraphEngine(graph_repo, graph_client, graph_client),
        "vector_search": VectorSearch(memory_repo, embedding_service),
        "adjudicator": AdjudicatorLLM(resolve_llm_client("adjudicator", matrix.adjudicator)),
        "enhancer": ContextEnhancerLLM(resolve_llm_client("context_enhancer", matrix.context_enhancer)),
        "memory_service": MemoryService(memory_repo, embedding_service, RelevanceService(), matrix.embedding.model),
        "maintenance_service": MaintenanceService(ScoringService()),
        "cortex_engine": build_cortex_engine(session),
    }
