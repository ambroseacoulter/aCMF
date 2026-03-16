"""Celery task for post-turn memory processing."""

from __future__ import annotations

import logging

from app.api.dependencies import build_worker_dependencies
from app.api.schemas.process import ProcessRequest
from app.db.session import get_session_factory
from app.llms.adjudicator import AdjudicationToolExecutor
from app.llms.prompting import PromptRenderContext
from app.storage.job_repo import JobRepository
from app.storage.turn_repo import TurnRepository
from app.storage.user_repo import UserRepository
from app.workers.queue import celery_app
from app.workers.sync_graph_projection_worker import dispatch_graph_projection_task

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.process_turn_worker.process_turn_task")
def process_turn_task(job_id: str) -> None:
    """Execute async post-turn adjudication and persistence."""
    session = get_session_factory()()
    deps = build_worker_dependencies(session)
    job_repo: JobRepository = deps["job_repo"]  # type: ignore[assignment]
    turn_repo: TurnRepository = deps["turn_repo"]  # type: ignore[assignment]
    user_repo: UserRepository = deps["user_repo"]  # type: ignore[assignment]
    memory_service = deps["memory_service"]
    graph_engine = deps["graph_engine"]
    vector_search = deps["vector_search"]
    adjudicator = deps["adjudicator"]
    memory_repo = deps["memory_repo"]
    turn_record_id: str | None = None
    projection_event_ids: list[str] = []
    try:
        job_repo.mark_running(job_id)
        job = job_repo.get(job_id)
        if job is None:
            session.commit()
            return
        request = ProcessRequest.model_validate(job.payload_json)
        user_repo.get_or_create(request.user_id)
        turn_record = turn_repo.create(
            {
                "user_id": request.user_id,
                "job_id": job.id,
                "user_message": request.turn.user_message,
                "assistant_response": request.turn.assistant_response,
                "occurred_at": request.turn.occurred_at,
                "user_message_id": request.turn.user_message_id,
                "assistant_message_id": request.turn.assistant_message_id,
                "metadata_json": request.metadata.model_dump(),
                "container_ids": [container.id for container in request.containers],
                "processing_status": "pending",
                "processing_notes": {},
            }
        )
        turn_record_id = turn_record.id
        container_ids = [container.id for container in request.containers]
        nearby_memories = memory_repo.list_recent_candidates(
            request.user_id,
            ["user", "global", "container"],
            container_ids,
            limit=12,
        )
        graph_result = graph_engine.traverse(request.user_id, request.turn.user_message, limit=10)
        contradiction_summaries = memory_repo.summarize_open_contradictions(request.user_id, limit=8)
        lineage_summaries = memory_repo.summarize_lineage(request.user_id, limit=8)
        context = PromptRenderContext(
            user_id=request.user_id,
            subject="USER: {0}\nASSISTANT: {1}".format(request.turn.user_message, request.turn.assistant_response),
            memories=nearby_memories,
            graph_facts=graph_result.facts,
            contradiction_summaries=contradiction_summaries,
            lineage_summaries=lineage_summaries,
            budgets={"max_candidates": 12},
            extras={
                "scope_policy": request.scope_policy.model_dump(),
                "metadata": request.metadata.model_dump(),
                "containers": [container.model_dump() for container in request.containers],
            },
        )
        executor = AdjudicationToolExecutor(
            user_id=request.user_id,
            container_ids=container_ids,
            memory_repo=memory_repo,
            vector_search=vector_search,
            graph_engine=graph_engine,
        )
        result = adjudicator.run_with_tools(context, executor)
        allowed_scopes: list[str] = []
        if request.scope_policy.write_user != "false":
            allowed_scopes.append("user")
        if request.scope_policy.write_global != "false":
            allowed_scopes.append("global")
        if request.scope_policy.write_container and container_ids:
            allowed_scopes.append("container")
        validation_errors = executor.validate(allowed_scopes)
        if validation_errors:
            raise ValueError("Invalid adjudication session: {0}".format("; ".join(validation_errors)))
        commit_result = memory_service.apply_adjudication_session(
            user_id=request.user_id,
            turn_record_id=turn_record.id,
            tool_session=result.tool_session,
            default_container_id=container_ids[0] if container_ids else None,
        )
        graph_mutation = graph_engine.persist_canonical_graph(
            user_id=request.user_id,
            memory_nodes=[memory.id for memory in commit_result.created_memories + commit_result.touched_memories],
            entities=commit_result.graph_entities,
            relations=commit_result.graph_relations,
            memory_links=commit_result.memory_links,
            graph_edges=commit_result.graph_edges,
        )
        projection_event_ids = graph_mutation.projection_event_ids
        created_or_touched = []
        for memory in commit_result.created_memories + commit_result.touched_memories:
            if memory.id not in {item.id for item in created_or_touched}:
                created_or_touched.append(memory)
        memory_service.touch_memories(
            created_or_touched,
            query_similarity_map={memory.id: 0.85 for memory in created_or_touched},
            graph_density_signal=graph_result.graph_density_signal,
        )
        user_repo.mark_snapshot_dirty(request.user_id, True)
        turn_repo.mark_processed(
            turn_record.id,
            {
                "reasoning_summary": result.reasoning_summary,
                "tool_call_count": result.tool_call_count,
                "staged_operation_count": len(result.tool_session.operations),
                "contradiction_topics": commit_result.contradiction_topics,
            },
        )
        job_repo.mark_succeeded(
            job_id,
            {
                "turn_record_id": turn_record.id,
                "tool_call_count": result.tool_call_count,
                "staged_operation_count": len(result.tool_session.operations),
                "memory_count_after": memory_repo.count_user_memories(request.user_id),
            },
        )
        session.commit()
        if projection_event_ids:
            try:
                dispatch_graph_projection_task(projection_event_ids)
            except Exception:
                logger.warning(
                    "Failed to dispatch immediate graph projection sync for job %s; beat will retry.",
                    job_id,
                    exc_info=True,
                )
    except Exception as exc:
        session.rollback()
        recovery_session = get_session_factory()()
        try:
            recovery_deps = build_worker_dependencies(recovery_session)
            recovery_job_repo: JobRepository = recovery_deps["job_repo"]  # type: ignore[assignment]
            recovery_turn_repo: TurnRepository = recovery_deps["turn_repo"]  # type: ignore[assignment]
            if turn_record_id:
                recovery_turn_repo.mark_failed(turn_record_id, str(exc))
            recovery_job_repo.mark_failed(job_id, str(exc))
            recovery_session.commit()
        finally:
            recovery_session.close()
        raise
    finally:
        session.close()
