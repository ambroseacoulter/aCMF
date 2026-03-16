"""Scope resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.enums import ScopeLevel, ScopeType


@dataclass
class ResolvedScope:
    """Resolved read scope definition."""

    scope_types: list[str]
    container_ids: list[str]
    missing_containers: list[str]
    applied_scope: ScopeLevel


class ScopeResolver:
    """Resolve requested scope level against existing containers."""

    def resolve(self, scope_level: ScopeLevel, requested_container_ids: list[str], existing_container_ids: list[str]) -> ResolvedScope:
        """Resolve effective scopes and missing containers."""
        missing = [container_id for container_id in requested_container_ids if container_id not in existing_container_ids]
        if scope_level == ScopeLevel.USER:
            return ResolvedScope([ScopeType.USER.value], [], missing, ScopeLevel.USER)
        if scope_level == ScopeLevel.USER_GLOBAL:
            return ResolvedScope([ScopeType.USER.value, ScopeType.GLOBAL.value], [], missing, ScopeLevel.USER_GLOBAL)
        return ResolvedScope(
            [ScopeType.USER.value, ScopeType.GLOBAL.value, ScopeType.CONTAINER.value],
            existing_container_ids,
            missing,
            ScopeLevel.USER_GLOBAL_CONTAINER,
        )
