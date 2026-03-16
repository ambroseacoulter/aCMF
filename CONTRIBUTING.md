# Contributing to aCMF

Thanks for contributing to aCMF.

aCMF is a long-term memory sidecar for LLM applications. It exposes read and write APIs, stores canonical state in PostgreSQL, uses pgvector for similarity search, projects graph state into Neo4j, and runs background work through Celery.

This guide explains how to set up the project, what parts matter architecturally, and what quality bar contributions are expected to meet.

## What We Want Help With

High-value contributions usually fall into one of these buckets:

1. Bug fixes
2. Retrieval quality and grounding improvements
3. Write-path correctness and memory lifecycle fixes
4. Graph projection and graph-assisted retrieval hardening
5. Test coverage
6. Documentation
7. Operability and developer experience

If a change affects memory correctness, retrieval grounding, graph consistency, or schema/API behavior, treat it as high-risk and test it accordingly.

## Development Setup

### Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.9+ | The project currently targets `>=3.9` |
| Docker + Docker Compose | Recommended for local development |
| Git | Standard workflow |

### Local Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

### Environment configuration

```bash
cp .env.example .env
```

The checked-in `.env.example` uses `stub` providers by default, which is the easiest way to run locally without external model credentials.

Minimum runtime services:

- PostgreSQL
- Redis
- Neo4j

The minimum environment variables needed when configuring manually are documented in the README and the full environment reference is in the docs site. [DOCS](https://ecmf.mintlify.app)

## Running aCMF

### Recommended: Docker Compose

```bash
docker compose up --build
```

This starts:

- `api`
- `worker`
- `beat`
- `postgres`
- `redis`
- `neo4j`

Container startup runs Alembic automatically before the service process starts.

### Run outside Docker

You still need reachable Postgres, Redis, and Neo4j instances.

```bash
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000
celery -A app.workers.queue.celery_app worker --loglevel=INFO
celery -A app.workers.queue.celery_app beat --loglevel=INFO
```

## Test and Verification

Run the full suite:

```bash
python3 -m pytest
```

Useful focused checks:

```bash
python3 -m pytest tests/test_routes.py
python3 -m pytest tests/test_workers.py
python3 -m pytest tests/test_context_engine.py
python3 -c 'from app.main import app; print(app.title, app.version)'
```

For a quick API smoke run against a live server:

```bash
python3 scripts/process_smoke_test.py --base-url http://localhost:8000 --pretty
```

If your change affects:

- schema or migrations: test on a fresh database
- workers or graph projection: verify a real process path, not just route tests
- retrieval behavior: add or update explicit read-path tests
- docs: make sure examples still match the real request and response shapes

## Project Structure

Key directories:

- `app/api`
  - FastAPI routes and request/response schemas
- `app/core`
  - settings, enums, time helpers, shared runtime config
- `app/db`
  - SQLAlchemy models and Alembic migrations
- `app/storage`
  - repositories for canonical persistence
- `app/services`
  - memory application logic, scoring, relevance, tool session validation, graph payload normalization
- `app/retrieval`
  - scope resolution, vector search, reranking, query relevance, graph retrieval models
- `app/llms`
  - role clients, prompt rendering, prompt templates
- `app/engines`
  - orchestration for read paths, graph sync, process flow, snapshots, and cortex
- `app/workers`
  - Celery tasks and schedules
- `tests`
  - route, service, graph, context, and worker tests
- `scripts`
  - helper scripts including container startup and process smoke testing

## Architecture Notes for Contributors

### Canonical storage

PostgreSQL is the source of truth.

That includes:

- users
- containers
- turns
- memories
- embeddings
- contradiction and lineage state
- jobs
- snapshots
- canonical graph tables
- graph projection outbox

Neo4j is projection-only. Do not treat it as canonical state.

### Read path

`/v1/context` and `/v1/deep-memory` share the same retrieval pipeline:

1. resolve scope
2. gather raw candidates from snapshot refs, vector search, metadata, and graph
3. apply strict query-relevance gating
4. expand graph only from relevant seeds
5. rerank relevant candidates
6. either call the LLM or abstain directly

Important implication:

- unrelated memories must not leak into public diagnostics or abstention reasons

### Write path

`/v1/process` returns quickly after enqueuing work.

The worker then:

1. stores a normalized turn record
2. runs the adjudicator tool loop
3. validates staged operations
4. commits canonical writes atomically
5. refreshes embeddings
6. updates graph canonical tables and outbox events
7. dispatches immediate graph projection
8. marks the user snapshot dirty

Important implication:

- LLM roles do not write canonical state directly
- canonical writes must remain application-controlled and transactional

### Graph subsystem

- canonical graph writes happen in Postgres first
- Neo4j projection is immediate async after commit
- Celery beat remains the backlog drain and retry path
- every persisted memory should be usable without requiring entity extraction

If you change graph behavior, test both canonical persistence and projection behavior.

## Contribution Rules

### Keep changes aligned with the current design

Do not introduce:

- direct Neo4j write-path ownership
- uncontrolled LLM writes to canonical tables
- retrieval behavior that exposes unrelated memories publicly
- undocumented request or response shape changes

### Add tests with behavior changes

Any change to one of these areas should come with tests:

- request or response schemas
- retrieval ranking or abstention behavior
- adjudicator staging and validation
- graph projection or outbox handling
- hourly cortex maintenance behavior

### Keep docs in sync

If you change:

- env vars
- endpoint shapes
- retrieval behavior
- worker/runtime behavior

update the relevant docs in the same change:

- `README.md`
- Mintlify pages under `reference/`, `guides/`, or `architecture/`

### Prefer small, focused PRs

Good PRs are:

- one logical behavior change
- one refactor with no behavior change
- one docs-only clarification

Avoid mixing schema redesign, retrieval tuning, and docs cleanup in one PR unless they are inseparable.

## Coding Expectations

### Style

- Follow the existing code style in the touched area
- Keep comments brief and only for non-obvious intent
- Prefer explicit, unsurprising logic over clever abstractions
- Keep interfaces narrow and typed where practical

### Error handling

- Fail safely on read paths
- Do not silently relax correctness on write paths
- Soft-fail graph reads when Neo4j is unavailable
- Avoid hiding canonical write failures

### Configuration

- New settings must live in `app/core/config.py`
- Add them to `.env.example` when relevant
- Document them if they affect contributors or operators

### Migrations

If you change the schema:

- update the SQLAlchemy models
- update the Alembic migration state appropriately
- verify the app still boots against a fresh database

## Pull Requests

### Before opening a PR

At minimum:

1. run `python3 -m pytest`
2. verify the changed behavior manually when practical
3. update docs if your change affects public behavior or contributor workflow

If relevant, include:

- example request/response before vs after
- migration notes
- env var additions
- operational impact

### PR description

A useful PR description should state:

- what changed
- why it changed
- how it was verified
- any follow-up risk or known limitation

### Suggested branch names

Examples:

- `fix/context-relevance-gating`
- `feat/graph-replay-task`
- `docs/update-contributing-guide`
- `refactor/context-engine-retrieval`

## Security and Data Correctness

Please call out security-sensitive or correctness-sensitive changes explicitly.

That includes changes involving:

- prompt or tool execution boundaries
- provider credentials or config resolution
- SQL queries or migrations
- graph projection retries and replay
- memory deletion, merge, contradiction, or supersession logic

## Documentation

Project docs live in two places:

- repo root docs such as `README.md` and `CONTRIBUTING.md`
- Mintlify docs under:
  - `index.mdx`
  - `quickstart.mdx`
  - `architecture/`
  - `guides/`
  - `reference/`

If you are unsure where something belongs:

- contributor workflow: `CONTRIBUTING.md`
- operator/setup guidance: `README.md` or `guides/`
- endpoint contracts: `reference/`
- system behavior and design: `architecture/`

## Questions and Discussion

If you are planning a larger change, open the discussion early and frame it in terms of:

- canonical correctness
- grounding quality
- operational safety
- migration impact
- testability

That makes review faster and reduces redesign churn later.
