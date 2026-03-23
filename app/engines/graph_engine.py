"""Graph orchestration and Neo4j integration."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from neo4j import GraphDatabase

from app.core.config import Settings
from app.core.enums import GraphNodeType, OutboxStatus, ProjectionEventType
from app.core.time import utc_now
from app.retrieval.graph_search import GraphFact, GraphTraversalResult
from app.services.graph_projection_service import GraphProjectionNormalizer
from app.storage.graph_repo import GraphRepository

logger = logging.getLogger(__name__)


class GraphProjectionClient(Protocol):
    """Write-side graph projection interface."""

    def clear_user_projection(self, user_id: str) -> None:
        """Delete all projected graph data for one user."""

    def upsert_memory_nodes(self, user_id: str, memory_nodes: list[dict]) -> None:
        """Project memory nodes into Neo4j."""

    def upsert_entities(self, user_id: str, entities: list[dict]) -> None:
        """Project entities into Neo4j."""

    def upsert_relations(self, user_id: str, relations: list[dict]) -> None:
        """Project relations into Neo4j."""

    def upsert_memory_links(self, user_id: str, memory_links: list[dict]) -> None:
        """Project memory links into Neo4j."""

    def upsert_graph_edges(self, user_id: str, graph_edges: list[dict]) -> None:
        """Project general graph edges into Neo4j."""


class GraphQueryClient(Protocol):
    """Read-side graph traversal interface."""

    def traverse_context(self, user_id: str, query: str, limit: int, seed_memory_ids: list[str] | None = None) -> GraphTraversalResult:
        """Return graph facts related to the query."""


class StubGraphClient(GraphProjectionClient, GraphQueryClient):
    """No-op graph client used for tests and local fallback."""

    def clear_user_projection(self, user_id: str) -> None:
        """No-op projection clear."""

    def upsert_memory_nodes(self, user_id: str, memory_nodes: list[dict]) -> None:
        """No-op projection."""

    def upsert_entities(self, user_id: str, entities: list[dict]) -> None:
        """No-op projection."""

    def upsert_relations(self, user_id: str, relations: list[dict]) -> None:
        """No-op projection."""

    def upsert_memory_links(self, user_id: str, memory_links: list[dict]) -> None:
        """No-op projection."""

    def upsert_graph_edges(self, user_id: str, graph_edges: list[dict]) -> None:
        """No-op projection."""

    def traverse_context(self, user_id: str, query: str, limit: int, seed_memory_ids: list[str] | None = None) -> GraphTraversalResult:
        """Return an empty traversal result."""
        return GraphTraversalResult()


class Neo4jGraphClient(GraphProjectionClient, GraphQueryClient):
    """Neo4j-backed graph projection and traversal client."""

    def __init__(self, settings: Settings) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

    def clear_user_projection(self, user_id: str) -> None:
        """Delete all projected graph data for one user."""
        query = """
        MATCH (n)
        WHERE n.user_id = $user_id
        DETACH DELETE n
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id)

    def upsert_memory_nodes(self, user_id: str, memory_nodes: list[dict]) -> None:
        """Project memory nodes into Neo4j."""
        if not memory_nodes:
            return
        query = """
        UNWIND $memory_nodes AS memory
        MERGE (m:Memory {id: memory.id})
        SET m.user_id = $user_id,
            m.scope_type = memory.scope_type,
            m.bucket_id = memory.bucket_id,
            m.memory_type = memory.memory_type,
            m.status = memory.status,
            m.importance_score = memory.importance_score,
            m.confidence_score = memory.confidence_score,
            m.novelty_score = memory.novelty_score,
            m.current_relevance_score = memory.current_relevance_score,
            m.average_relevance_score = memory.average_relevance_score,
            m.contradiction_risk = memory.contradiction_risk,
            m.recall_count = memory.recall_count,
            m.decay_score = memory.decay_score,
            m.summary = memory.summary,
            m.content = memory.content,
            m.rationale = memory.rationale,
            m.evidence_json = memory.evidence_json,
            m.search_text = memory.search_text,
            m.created_at = memory.created_at,
            m.updated_at = memory.updated_at
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id, memory_nodes=memory_nodes)

    def upsert_entities(self, user_id: str, entities: list[dict]) -> None:
        """Project entities into Neo4j."""
        if not entities:
            return
        serialized_entities = [self._serialize_projection_maps(entity, ["attributes_json"]) for entity in entities]
        query = """
        UNWIND $entities AS entity
        MERGE (n:Entity {id: entity.id})
        SET n.user_id = $user_id,
            n.entity_type = entity.entity_type,
            n.canonical_name = entity.canonical_name,
            n.aliases = entity.aliases_json,
            n.attributes_json = entity.attributes_json,
            n.search_text = entity.search_text
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id, entities=serialized_entities)

    def upsert_relations(self, user_id: str, relations: list[dict]) -> None:
        """Project relations into Neo4j."""
        if not relations:
            return
        serialized_relations = [
            self._serialize_projection_maps(relation, ["attributes_json", "evidence_json"])
            for relation in relations
        ]
        query = """
        UNWIND $relations AS relation
        MATCH (source:Entity {id: relation.from_entity_id})
        MATCH (target:Entity {id: relation.to_entity_id})
        MERGE (source)-[r:RELATED {id: relation.id}]->(target)
        SET r.user_id = $user_id,
            r.relation_type = relation.relation_type,
            r.confidence_score = relation.confidence_score,
            r.attributes_json = relation.attributes_json,
            r.evidence_json = relation.evidence_json,
            r.search_text = relation.search_text
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id, relations=serialized_relations)

    def upsert_memory_links(self, user_id: str, memory_links: list[dict]) -> None:
        """Project memory-to-entity links into Neo4j."""
        if not memory_links:
            return
        query = """
        UNWIND $memory_links AS link
        MERGE (m:Memory {id: link.memory_id})
        SET m.user_id = $user_id
        WITH link, m
        MATCH (e:Entity {id: link.entity_id})
        WITH link, m, e
        MERGE (m)-[r:MENTIONS {id: link.id}]->(e)
        SET r.user_id = $user_id,
            r.link_type = link.link_type,
            r.search_text = link.search_text
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id, memory_links=memory_links)

    def upsert_graph_edges(self, user_id: str, graph_edges: list[dict]) -> None:
        """Project general graph edges into Neo4j."""
        if not graph_edges:
            return
        grouped: dict[tuple[str, str], list[dict]] = {}
        for edge in graph_edges:
            key = (self._label_for(edge["from_node_type"]), self._label_for(edge["to_node_type"]))
            grouped.setdefault(key, []).append(self._serialize_projection_maps(edge, ["attributes_json"]))
        with self.driver.session() as session:
            for (from_label, to_label), edges in grouped.items():
                query = """
                UNWIND $edges AS edge
                MERGE (source:{from_label} {{id: edge.from_node_id}})
                SET source.user_id = $user_id
                WITH edge, source
                MERGE (target:{to_label} {{id: edge.to_node_id}})
                SET target.user_id = $user_id
                WITH edge, source, target
                MERGE (source)-[r:GRAPH_EDGE {{id: edge.id}}]->(target)
                SET r.user_id = $user_id,
                    r.edge_type = edge.edge_type,
                    r.confidence_score = edge.confidence_score,
                    r.attributes_json = edge.attributes_json,
                    r.search_text = edge.search_text,
                    r.source_type = edge.source_type,
                    r.source_ref = edge.source_ref
                """.format(from_label=from_label, to_label=to_label)
                session.run(query, user_id=user_id, edges=edges)

    def traverse_context(self, user_id: str, query: str, limit: int, seed_memory_ids: list[str] | None = None) -> GraphTraversalResult:
        """Traverse Neo4j for graph facts connected to relevant memory seeds."""
        if seed_memory_ids is not None and not seed_memory_ids:
            return GraphTraversalResult()
        if seed_memory_ids is None:
            edge_cypher = """
            MATCH (a)-[r:GRAPH_EDGE]->(b)
            WHERE a.user_id = $user_id
              AND b.user_id = $user_id
              AND r.user_id = $user_id
              AND (
                $search_query = '' OR
                toLower(coalesce(a.search_text, a.canonical_name, a.summary, a.content, '')) CONTAINS toLower($search_query) OR
                toLower(coalesce(r.search_text, r.edge_type, '')) CONTAINS toLower($search_query) OR
                toLower(coalesce(b.search_text, b.canonical_name, b.summary, b.content, '')) CONTAINS toLower($search_query)
              )
            RETURN CASE WHEN a:Entity THEN a.canonical_name ELSE coalesce(a.summary, substring(a.content, 0, 120)) END AS source_name,
                   r.edge_type AS relation_type,
                   CASE WHEN b:Entity THEN b.canonical_name ELSE coalesce(b.summary, substring(b.content, 0, 120)) END AS target_name,
                   r.confidence_score AS confidence,
                   CASE WHEN a:Memory THEN a.id ELSE NULL END AS source_memory_id,
                   CASE WHEN b:Memory THEN b.id ELSE NULL END AS target_memory_id
            LIMIT $limit
            """
            memory_cypher = """
            MATCH (m:Memory)
            WHERE m.user_id = $user_id
              AND (
                $search_query = '' OR
                toLower(coalesce(m.search_text, m.summary, m.content, '')) CONTAINS toLower($search_query)
              )
            RETURN m.id AS memory_id
            LIMIT $limit
            """
        else:
            edge_cypher = """
            MATCH (seed:Memory)
            WHERE seed.user_id = $user_id
              AND seed.id IN $seed_memory_ids
            MATCH (seed)-[r]-(neighbor)
            WHERE neighbor.user_id = $user_id
              AND (
                $search_query = '' OR
                toLower(coalesce(seed.search_text, seed.summary, seed.content, '')) CONTAINS toLower($search_query) OR
                toLower(coalesce(r.search_text, r.edge_type, r.relation_type, r.link_type, '')) CONTAINS toLower($search_query) OR
                toLower(coalesce(neighbor.search_text, neighbor.canonical_name, neighbor.summary, neighbor.content, '')) CONTAINS toLower($search_query)
              )
            RETURN coalesce(seed.summary, substring(seed.content, 0, 120)) AS source_name,
                   coalesce(r.edge_type, r.relation_type, r.link_type, type(r)) AS relation_type,
                   CASE WHEN neighbor:Entity THEN neighbor.canonical_name ELSE coalesce(neighbor.summary, substring(neighbor.content, 0, 120)) END AS target_name,
                   coalesce(r.confidence_score, 0.5) AS confidence,
                   seed.id AS source_memory_id,
                   CASE WHEN neighbor:Memory THEN neighbor.id ELSE NULL END AS target_memory_id
            LIMIT $limit
            """
            memory_cypher = """
            MATCH (seed:Memory)
            WHERE seed.user_id = $user_id
              AND seed.id IN $seed_memory_ids
            MATCH (seed)-[r]-(neighbor:Memory)
            WHERE neighbor.user_id = $user_id
              AND (
                $search_query = '' OR
                toLower(coalesce(neighbor.search_text, neighbor.summary, neighbor.content, '')) CONTAINS toLower($search_query) OR
                toLower(coalesce(r.search_text, r.edge_type, r.relation_type, r.link_type, '')) CONTAINS toLower($search_query)
              )
            RETURN DISTINCT neighbor.id AS memory_id
            LIMIT $limit
            """
        facts: list[GraphFact] = []
        memory_ids: list[str] = []
        with self.driver.session() as session:
            result = session.run(
                edge_cypher,
                user_id=user_id,
                search_query=query,
                limit=limit,
                seed_memory_ids=seed_memory_ids or [],
            )
            for row in result:
                facts.append(
                    GraphFact(
                        entity_name=row["source_name"],
                        relation_type=row["relation_type"],
                        related_entity_name=row["target_name"],
                        confidence_score=row["confidence"] or 0.5,
                    )
                )
                if row["source_memory_id"]:
                    memory_ids.append(row["source_memory_id"])
                if row["target_memory_id"]:
                    memory_ids.append(row["target_memory_id"])
            memory_result = session.run(
                memory_cypher,
                user_id=user_id,
                search_query=query,
                limit=limit,
                seed_memory_ids=seed_memory_ids or [],
            )
            for row in memory_result:
                memory_ids.append(row["memory_id"])
        unique_memory_ids = list(dict.fromkeys(memory_ids))
        density = min((len(facts) + len(unique_memory_ids)) / max(limit, 1), 1.0)
        return GraphTraversalResult(facts=facts, memory_ids=unique_memory_ids, graph_density_signal=density)

    @staticmethod
    def _label_for(node_type: str) -> str:
        """Map graph node types to Neo4j labels."""
        return "Memory" if node_type == GraphNodeType.MEMORY.value else "Entity"

    @staticmethod
    def _serialize_projection_maps(payload: dict, field_names: list[str]) -> dict:
        """Compatibility wrapper for tests and callers using the projection normalizer contract."""
        normalizer = GraphProjectionNormalizer()
        serialized = dict(payload)
        for field_name in field_names:
            serialized[field_name] = normalizer._serialize_structured_field(serialized.get(field_name))
        return serialized


@dataclass
class CanonicalGraphMutation:
    """Canonical graph updates produced during process/cortex flows."""

    memory_nodes: list[dict]
    entities: list[dict]
    relations: list[dict]
    memory_entity_links: list[dict]
    graph_edges: list[dict]
    projection_event_ids: list[str]


@dataclass
class ProjectionSyncResult:
    """Outcome of one projection sync batch."""

    processed_count: int
    failed_count: int
    processed_event_ids: list[str]
    failed_event_ids: list[str]


class GraphEngine:
    """Coordinate canonical graph writes and Neo4j projection reads."""

    def __init__(self, graph_repo: GraphRepository, projection_client: GraphProjectionClient, query_client: GraphQueryClient) -> None:
        self.graph_repo = graph_repo
        self.projection_client = projection_client
        self.query_client = query_client
        self.normalizer = GraphProjectionNormalizer()

    def traverse(self, user_id: str, query: str, limit: int, seed_memory_ids: list[str] | None = None) -> GraphTraversalResult:
        """Return graph traversal results, soft-failing on connectivity issues."""
        try:
            return self.query_client.traverse_context(user_id, query, limit, seed_memory_ids=seed_memory_ids)
        except Exception:
            logger.warning("Graph traversal failed for user_id=%s", user_id, exc_info=True)
            return GraphTraversalResult()

    def persist_canonical_graph(
        self,
        *,
        user_id: str,
        memory_nodes: Iterable[str] | None = None,
        entities: Iterable[dict],
        relations: Iterable[dict],
        memory_links: Iterable[tuple[str, str, str]],
        graph_edges: Iterable[dict] | None = None,
    ) -> CanonicalGraphMutation:
        """Persist canonical graph rows and enqueue projection events."""
        persisted_memory_nodes = [
            self.normalizer.normalize_memory_node(payload)
            for payload in self.graph_repo.get_memory_projection_payloads(list(dict.fromkeys(memory_nodes or [])))
        ]
        persisted_entities: list[dict] = []
        name_to_id: dict[str, str] = {}
        for entity in entities:
            record = self.graph_repo.upsert_entity(
                user_id=user_id,
                entity_type=entity["entity_type"],
                canonical_name=entity["canonical_name"],
                aliases=entity.get("aliases", []),
                attributes=entity.get("attributes", {}),
            )
            persisted_entities.append(
                self.normalizer.normalize_entity(
                    {
                    "id": record.id,
                    "entity_type": record.entity_type,
                    "canonical_name": record.canonical_name,
                    "aliases_json": record.aliases_json,
                    "attributes_json": record.attributes_json,
                    }
                )
            )
            name_to_id[record.canonical_name] = record.id
        persisted_relations: list[dict] = []
        persisted_graph_edges: list[dict] = []
        for relation in relations:
            from_entity_id = name_to_id.get(relation["from_entity_name"])
            to_entity_id = name_to_id.get(relation["to_entity_name"])
            if not from_entity_id:
                existing_from = self.graph_repo.find_entity(user_id, relation["from_entity_name"])
                from_entity_id = existing_from.id if existing_from else None
            if not to_entity_id:
                existing_to = self.graph_repo.find_entity(user_id, relation["to_entity_name"])
                to_entity_id = existing_to.id if existing_to else None
            if not from_entity_id or not to_entity_id:
                continue
            record = self.graph_repo.upsert_relation(
                user_id=user_id,
                from_entity_id=from_entity_id,
                to_entity_id=to_entity_id,
                relation_type=relation["relation_type"],
                confidence_score=relation.get("confidence_score", 0.5),
                evidence={"items": relation.get("evidence", [])},
                attributes=relation.get("attributes", {}),
            )
            persisted_relations.append(
                self.normalizer.normalize_relation(
                    {
                    "id": record.id,
                    "from_entity_id": record.from_entity_id,
                    "to_entity_id": record.to_entity_id,
                    "from_entity_name": relation["from_entity_name"],
                    "to_entity_name": relation["to_entity_name"],
                    "relation_type": record.relation_type,
                    "confidence_score": record.confidence_score,
                    "attributes_json": record.attributes_json,
                    "evidence_json": record.evidence_json,
                    }
                )
            )
            edge = self.graph_repo.upsert_graph_edge(
                user_id=user_id,
                from_node_type=GraphNodeType.ENTITY.value,
                from_node_id=record.from_entity_id,
                to_node_type=GraphNodeType.ENTITY.value,
                to_node_id=record.to_entity_id,
                edge_type=record.relation_type,
                confidence_score=record.confidence_score,
                attributes=record.attributes_json,
                source_type="relation",
                source_ref=record.id,
            )
            persisted_graph_edges.append(self.normalizer.normalize_graph_edge(self._serialize_graph_edge(edge)))
        persisted_memory_links: list[dict] = []
        for memory_id, entity_name, link_type in memory_links:
            entity_id = name_to_id.get(entity_name)
            if not entity_id:
                existing_entity = self.graph_repo.find_entity(user_id, entity_name)
                entity_id = existing_entity.id if existing_entity else None
            if not entity_id:
                continue
            link = self.graph_repo.link_memory_to_entity(memory_id, entity_id, link_type)
            persisted_memory_links.append(
                self.normalizer.normalize_memory_link(
                    {
                    "id": "{0}:{1}:{2}".format(link.memory_id, link.entity_id, link.link_type),
                    "memory_id": link.memory_id,
                    "entity_id": link.entity_id,
                    "link_type": link.link_type,
                    "entity_name": entity_name,
                    }
                )
            )
            edge = self.graph_repo.upsert_graph_edge(
                user_id=user_id,
                from_node_type=GraphNodeType.MEMORY.value,
                from_node_id=link.memory_id,
                to_node_type=GraphNodeType.ENTITY.value,
                to_node_id=link.entity_id,
                edge_type=link.link_type,
                confidence_score=0.8,
                attributes={},
                source_type="memory_entity_link",
                source_ref="{0}:{1}".format(link.memory_id, link.entity_id),
            )
            persisted_graph_edges.append(self.normalizer.normalize_graph_edge(self._serialize_graph_edge(edge)))
        for edge_payload in graph_edges or []:
            from_node_type = edge_payload["from_node_type"]
            to_node_type = edge_payload["to_node_type"]
            from_ref = edge_payload["from_node_id"]
            to_ref = edge_payload["to_node_id"]
            if from_node_type == GraphNodeType.ENTITY.value:
                from_ref = name_to_id.get(from_ref, from_ref)
                if from_ref == edge_payload["from_node_id"]:
                    existing_entity = self.graph_repo.find_entity(user_id, from_ref)
                    from_ref = existing_entity.id if existing_entity else from_ref
            if to_node_type == GraphNodeType.ENTITY.value:
                to_ref = name_to_id.get(to_ref, to_ref)
                if to_ref == edge_payload["to_node_id"]:
                    existing_entity = self.graph_repo.find_entity(user_id, to_ref)
                    to_ref = existing_entity.id if existing_entity else to_ref
            edge = self.graph_repo.upsert_graph_edge(
                user_id=user_id,
                from_node_type=from_node_type,
                from_node_id=from_ref,
                to_node_type=to_node_type,
                to_node_id=to_ref,
                edge_type=edge_payload["edge_type"],
                confidence_score=edge_payload.get("confidence_score", 0.5),
                attributes=edge_payload.get("attributes", {}),
                source_type=edge_payload.get("source_type", "system"),
                source_ref=edge_payload.get("source_ref"),
            )
            persisted_graph_edges.append(self.normalizer.normalize_graph_edge(self._serialize_graph_edge(edge)))
        projection_event_ids: list[str] = []
        if persisted_memory_nodes:
            event = self.graph_repo.enqueue_projection_event(
                ProjectionEventType.MEMORY_NODES_UPSERTED.value,
                user_id,
                {"memory_nodes": persisted_memory_nodes},
            )
            projection_event_ids.append(event.id)
        if persisted_entities:
            event = self.graph_repo.enqueue_projection_event(
                ProjectionEventType.ENTITIES_UPSERTED.value,
                user_id,
                {"entities": persisted_entities},
            )
            projection_event_ids.append(event.id)
        if persisted_relations:
            event = self.graph_repo.enqueue_projection_event(
                ProjectionEventType.RELATIONS_UPSERTED.value,
                user_id,
                {"relations": persisted_relations},
            )
            projection_event_ids.append(event.id)
        if persisted_memory_links:
            event = self.graph_repo.enqueue_projection_event(
                ProjectionEventType.MEMORY_LINKS_UPSERTED.value,
                user_id,
                {"memory_links": persisted_memory_links},
            )
            projection_event_ids.append(event.id)
        if persisted_graph_edges:
            deduped_edges = list({edge["id"]: edge for edge in persisted_graph_edges}.values())
            event = self.graph_repo.enqueue_projection_event(
                ProjectionEventType.GRAPH_EDGES_UPSERTED.value,
                user_id,
                {"graph_edges": deduped_edges},
            )
            projection_event_ids.append(event.id)
            persisted_graph_edges = deduped_edges
        return CanonicalGraphMutation(
            persisted_memory_nodes,
            persisted_entities,
            persisted_relations,
            persisted_memory_links,
            persisted_graph_edges,
            projection_event_ids,
        )

    def sync_projection(self, events: list, *, max_attempts: int = 5) -> ProjectionSyncResult:
        """Push pending outbox events into Neo4j."""
        processed = 0
        failed = 0
        processed_event_ids: list[str] = []
        failed_event_ids: list[str] = []
        for event in events:
            payload = event.payload_json
            event.attempt_count = int(event.attempt_count or 0) + 1
            try:
                if event.event_type == ProjectionEventType.MEMORY_NODES_UPSERTED.value:
                    self.projection_client.upsert_memory_nodes(event.user_id, payload.get("memory_nodes", []))
                elif event.event_type == ProjectionEventType.ENTITIES_UPSERTED.value:
                    self.projection_client.upsert_entities(event.user_id, payload.get("entities", []))
                elif event.event_type == ProjectionEventType.RELATIONS_UPSERTED.value:
                    self.projection_client.upsert_relations(event.user_id, payload.get("relations", []))
                elif event.event_type == ProjectionEventType.MEMORY_LINKS_UPSERTED.value:
                    self.projection_client.upsert_memory_links(event.user_id, payload.get("memory_links", []))
                elif event.event_type == ProjectionEventType.GRAPH_EDGES_UPSERTED.value:
                    self.projection_client.upsert_graph_edges(event.user_id, payload.get("graph_edges", []))
                event.status = OutboxStatus.PROCESSED.value
                event.error_message = None
                event.processed_at = utc_now()
                processed += 1
                processed_event_ids.append(event.id)
            except Exception as exc:
                failed += 1
                failed_event_ids.append(event.id)
                event.error_message = str(exc)
                event.processed_at = None
                event.status = (
                    OutboxStatus.FAILED.value if event.attempt_count >= max_attempts else OutboxStatus.PENDING.value
                )
                logger.warning(
                    "Graph projection failed for outbox event %s (%s) on attempt %s/%s.",
                    event.id,
                    event.event_type,
                    event.attempt_count,
                    max_attempts,
                    exc_info=True,
                )
        return ProjectionSyncResult(processed, failed, processed_event_ids, failed_event_ids)

    def rebuild_projection(self, user_id: str | None = None) -> int:
        """Rebuild Neo4j projection directly from canonical Postgres state."""
        user_ids = [user_id] if user_id else self.graph_repo.list_projection_user_ids()
        rebuilt = 0
        for current_user_id in user_ids:
            if not current_user_id:
                continue
            self.projection_client.clear_user_projection(current_user_id)
            memory_nodes = [
                self.normalizer.normalize_memory_node(payload)
                for payload in self.graph_repo.get_user_memory_projection_payloads(current_user_id)
            ]
            entities = [
                self.normalizer.normalize_entity(payload)
                for payload in self.graph_repo.get_entity_projection_payloads(current_user_id)
            ]
            relations = [
                self.normalizer.normalize_relation(payload)
                for payload in self.graph_repo.get_relation_projection_payloads(current_user_id)
            ]
            memory_links = [
                self.normalizer.normalize_memory_link(payload)
                for payload in self.graph_repo.get_memory_link_projection_payloads(current_user_id)
            ]
            graph_edges = [
                self.normalizer.normalize_graph_edge(payload)
                for payload in self.graph_repo.get_graph_edge_projection_payloads(current_user_id)
            ]
            self.projection_client.upsert_memory_nodes(current_user_id, memory_nodes)
            self.projection_client.upsert_entities(current_user_id, entities)
            self.projection_client.upsert_relations(current_user_id, relations)
            self.projection_client.upsert_memory_links(current_user_id, memory_links)
            self.projection_client.upsert_graph_edges(current_user_id, graph_edges)
            rebuilt += 1
        return rebuilt

    @staticmethod
    def _serialize_graph_edge(edge) -> dict:
        """Serialize a canonical graph edge for projection."""
        return {
            "id": edge.id,
            "from_node_type": edge.from_node_type,
            "from_node_id": edge.from_node_id,
            "to_node_type": edge.to_node_type,
            "to_node_id": edge.to_node_id,
            "edge_type": edge.edge_type,
            "confidence_score": edge.confidence_score,
            "attributes_json": edge.attributes_json,
            "source_type": edge.source_type,
            "source_ref": edge.source_ref,
        }
