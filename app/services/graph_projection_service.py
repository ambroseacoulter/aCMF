"""Helpers for preparing canonical graph data for Neo4j projection."""

from __future__ import annotations

import json
from typing import Any


class GraphProjectionNormalizer:
    """Normalize canonical payloads into Neo4j-safe projection records."""

    def normalize_memory_node(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Prepare one memory-node payload for projection."""
        normalized = dict(payload)
        normalized["search_text"] = self._compact_text(
            [
                payload.get("memory_type"),
                payload.get("summary"),
                payload.get("content"),
                payload.get("rationale"),
                self._flatten_structured_value(payload.get("evidence_json")),
            ]
        )
        normalized["evidence_json"] = self._serialize_structured_field(payload.get("evidence_json"))
        return normalized

    def normalize_entity(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Prepare one entity payload for projection."""
        normalized = dict(payload)
        normalized["search_text"] = self._compact_text(
            [
                payload.get("canonical_name"),
                payload.get("entity_type"),
                " ".join(payload.get("aliases_json", []) or []),
                self._selected_attribute_text(payload.get("attributes_json")),
            ]
        )
        normalized["attributes_json"] = self._serialize_structured_field(payload.get("attributes_json"))
        return normalized

    def normalize_relation(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Prepare one entity-relation payload for projection."""
        normalized = dict(payload)
        normalized["search_text"] = self._compact_text(
            [
                payload.get("relation_type"),
                payload.get("from_entity_name"),
                payload.get("to_entity_name"),
                self._flatten_structured_value(payload.get("evidence_json")),
                self._selected_attribute_text(payload.get("attributes_json")),
            ]
        )
        normalized["attributes_json"] = self._serialize_structured_field(payload.get("attributes_json"))
        normalized["evidence_json"] = self._serialize_structured_field(payload.get("evidence_json"))
        return normalized

    def normalize_memory_link(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Prepare one memory-entity link payload for projection."""
        normalized = dict(payload)
        normalized["id"] = payload.get("id") or "{0}:{1}:{2}".format(
            payload["memory_id"],
            payload["entity_id"],
            payload.get("link_type", "MENTIONS"),
        )
        normalized["search_text"] = self._compact_text(
            [
                payload.get("link_type"),
                payload.get("entity_name"),
                payload.get("memory_summary"),
            ]
        )
        return normalized

    def normalize_graph_edge(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Prepare one general graph edge payload for projection."""
        normalized = dict(payload)
        normalized["search_text"] = self._compact_text(
            [
                payload.get("edge_type"),
                payload.get("source_type"),
                payload.get("source_ref"),
                self._selected_attribute_text(payload.get("attributes_json")),
            ]
        )
        normalized["attributes_json"] = self._serialize_structured_field(payload.get("attributes_json"))
        return normalized

    @staticmethod
    def _serialize_structured_field(value: Any) -> str | None:
        """Serialize a structured field into a deterministic JSON string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def _selected_attribute_text(self, value: Any) -> str:
        """Return a searchable string from selected structured attributes."""
        if not isinstance(value, dict):
            return self._flatten_structured_value(value)
        parts: list[str] = []
        for key in sorted(value):
            flattened = self._flatten_structured_value(value[key])
            if flattened:
                parts.append("{0} {1}".format(key, flattened))
        return " ".join(parts)

    def _flatten_structured_value(self, value: Any) -> str:
        """Flatten structured values into searchable text."""
        if value is None:
            return ""
        if isinstance(value, dict):
            parts: list[str] = []
            for key in sorted(value):
                flattened = self._flatten_structured_value(value[key])
                if flattened:
                    parts.append("{0} {1}".format(key, flattened))
            return " ".join(parts)
        if isinstance(value, list):
            return " ".join(self._flatten_structured_value(item) for item in value if self._flatten_structured_value(item))
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return ""

    @staticmethod
    def _compact_text(parts: list[str | None]) -> str:
        """Join non-empty text parts into a normalized search string."""
        normalized = [" ".join(str(part).split()) for part in parts if part]
        return " ".join(part for part in normalized if part)
