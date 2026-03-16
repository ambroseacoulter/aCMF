"""Schemas for `/v1/process`."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.api.schemas.common import ContainerHint, RequestMetadata, TurnEnvelope


class ScopePolicy(BaseModel):
    """Write policy controls for post-turn processing."""

    write_user: str = Field(default="auto", pattern="^(auto|true|false)$")
    write_global: str = Field(default="auto", pattern="^(auto|true|false)$")
    write_container: bool = True


class ProcessRequest(BaseModel):
    """Post-turn processing request."""

    user_id: str
    containers: list[ContainerHint] = Field(default_factory=list)
    scope_policy: ScopePolicy = Field(default_factory=ScopePolicy)
    turn: TurnEnvelope
    metadata: RequestMetadata = Field(default_factory=RequestMetadata)


class ProcessResponse(BaseModel):
    """Async processing acknowledgement."""

    status: str
    job_id: str
    created_user: bool
    created_containers: list[str]
    accepted_at: datetime
