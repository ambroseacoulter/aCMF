"""Application configuration for aCMF v1."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.enums import ProviderKind


class OpenAICompatibleRoleConfig(BaseModel):
    """Resolved provider configuration for a chat-completion role."""

    provider: ProviderKind
    base_url: str | None = None
    api_key: str | None = None
    model: str
    timeout_seconds: float = 30.0


class EmbeddingRoleConfig(BaseModel):
    """Resolved provider configuration for embeddings."""

    provider: ProviderKind
    base_url: str | None = None
    api_key: str | None = None
    model: str
    dimensions: int
    timeout_seconds: float = 30.0


class ResolvedProviderMatrix(BaseModel):
    """Resolved provider matrix for all role clients."""

    adjudicator: OpenAICompatibleRoleConfig
    context_enhancer: OpenAICompatibleRoleConfig
    cortex: OpenAICompatibleRoleConfig
    embedding: EmbeddingRoleConfig


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ACMF_", extra="ignore")

    env: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    database_url: str = "postgresql+psycopg://acmf:acmf@localhost:5432/acmf"
    redis_url: str = "redis://localhost:6379/0"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "acmfpassword"
    auto_migrate_on_startup: bool = True
    db_startup_timeout_seconds: float = 60.0

    openai_default_base_url: str | None = "https://api.openai.com/v1"
    openai_default_api_key: str | None = None
    openai_default_timeout_seconds: float = 30.0

    adjudicator_provider: ProviderKind | str = ProviderKind.OPENAI_COMPATIBLE
    adjudicator_base_url: str | None = None
    adjudicator_api_key: str | None = None
    adjudicator_model: str = "gpt-4.1-mini"
    adjudicator_timeout_seconds: float | None = None

    context_enhancer_provider: ProviderKind | str = ProviderKind.OPENAI_COMPATIBLE
    context_enhancer_base_url: str | None = None
    context_enhancer_api_key: str | None = None
    context_enhancer_model: str = "gpt-4.1-mini"
    context_enhancer_timeout_seconds: float | None = None

    cortex_provider: ProviderKind | str = ProviderKind.OPENAI_COMPATIBLE
    cortex_base_url: str | None = None
    cortex_api_key: str | None = None
    cortex_model: str = "gpt-4.1-mini"
    cortex_timeout_seconds: float | None = None

    embedding_provider: ProviderKind | str = ProviderKind.OPENAI_COMPATIBLE
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_timeout_seconds: float | None = None

    context_max_output_tokens: int = 400
    deep_memory_max_output_tokens: int = 500
    context_default_candidate_limit: int = 30
    deep_memory_default_candidate_limit: int = 40
    simple_candidate_limit: int = 10
    balanced_candidate_limit: int = 30
    deep_candidate_limit: int = 50
    simple_graph_limit: int = 5
    balanced_graph_limit: int = 10
    deep_graph_limit: int = 20
    query_text_relevance_threshold: float = 0.25
    query_vector_similarity_threshold: float = 0.78
    graph_query_relevance_threshold: float = 0.2
    evidence_abstain_threshold: float = 0.45
    celery_task_always_eager: bool = Field(default=False)
    graph_projection_batch_size: int = 100
    graph_projection_max_attempts: int = 5
    graph_projection_retry_backoff_seconds: float = 300.0

    @field_validator(
        "adjudicator_timeout_seconds",
        "context_enhancer_timeout_seconds",
        "cortex_timeout_seconds",
        "embedding_timeout_seconds",
        mode="before",
    )
    @classmethod
    def _blank_timeout_to_none(cls, value: object) -> object:
        """Treat blank timeout env vars as unset."""
        if value == "":
            return None
        return value

    def _resolve_provider(self, value: ProviderKind | str) -> ProviderKind:
        """Normalize provider strings."""
        if isinstance(value, ProviderKind):
            return value
        return ProviderKind(value)

    def _default_provider(self) -> ProviderKind:
        """Return the runtime default provider."""
        if self.env == "test":
            return ProviderKind.STUB
        return ProviderKind.OPENAI_COMPATIBLE

    def _resolve_chat_role(self, prefix: str, fallback_model: str) -> OpenAICompatibleRoleConfig:
        """Resolve one chat role configuration."""
        provider = self._resolve_provider(getattr(self, "{0}_provider".format(prefix)) or self._default_provider())
        return OpenAICompatibleRoleConfig(
            provider=provider,
            base_url=getattr(self, "{0}_base_url".format(prefix)) or self.openai_default_base_url,
            api_key=getattr(self, "{0}_api_key".format(prefix)) or self.openai_default_api_key,
            model=getattr(self, "{0}_model".format(prefix)) or fallback_model,
            timeout_seconds=getattr(self, "{0}_timeout_seconds".format(prefix)) or self.openai_default_timeout_seconds,
        )

    def resolve_provider_matrix(self) -> ResolvedProviderMatrix:
        """Resolve the provider matrix used by dependency assembly."""
        embedding_provider = self._resolve_provider(self.embedding_provider or self._default_provider())
        return ResolvedProviderMatrix(
            adjudicator=self._resolve_chat_role("adjudicator", self.adjudicator_model),
            context_enhancer=self._resolve_chat_role("context_enhancer", self.context_enhancer_model),
            cortex=self._resolve_chat_role("cortex", self.cortex_model),
            embedding=EmbeddingRoleConfig(
                provider=embedding_provider,
                base_url=self.embedding_base_url or self.openai_default_base_url,
                api_key=self.embedding_api_key or self.openai_default_api_key,
                model=self.embedding_model,
                dimensions=self.embedding_dimensions,
                timeout_seconds=self.embedding_timeout_seconds or self.openai_default_timeout_seconds,
            ),
        )

    def read_profile(self) -> Dict[str, Dict[str, int]]:
        """Return retrieval profile configuration keyed by read mode."""
        return {
            "simple": {"candidate_limit": self.simple_candidate_limit, "graph_limit": self.simple_graph_limit},
            "balanced": {"candidate_limit": self.balanced_candidate_limit, "graph_limit": self.balanced_graph_limit},
            "deep": {"candidate_limit": self.deep_candidate_limit, "graph_limit": self.deep_graph_limit},
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object."""
    return Settings()
