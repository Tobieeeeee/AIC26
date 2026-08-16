# 05 — Reuse FastAPI Serving Patterns and Complete Tests

## Goal

Reuse the FastAPI Lab and Lab 6 serving patterns to expose the retrieval baseline through a clean API.

Do not invent endpoints for features that do not exist.

## Required preflight

Read:

```text
AGENTS.md
codex/00_MASTER_PLAN.md
codex/05_API_AND_TESTS.md
docs/LAB_REUSE_MAP.md
docs/CODEX_PROGRESS.md
```

Confirm Phase 04 is complete or clearly identify which multimodal components are available.

## Source material

Inspect the actual FastAPI Lab and the FastAPI section of Lab 6 under:

```text
D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder
```

Reuse patterns for:

- app structure
- Pydantic schemas
- request validation
- response validation
- API tests
- CI

## Target files

Create or complete:

```text
src/mvr/api/app.py
src/mvr/api/schemas.py
```

## Minimum endpoints

Always provide if semantic baseline exists:

```text
GET /health

POST /search/semantic
```

Only expose the following if the underlying implementations are actually working:

```text
POST /search/text

POST /search/object

POST /search/hybrid
```

Do not create fake placeholder success responses.

## Health endpoint

`/health` should report meaningful runtime status without pretending unavailable services are healthy.

For example, it may report:

- API ready
- semantic index loaded/not loaded
- Elasticsearch configured/not configured

Do not leak secrets.

## API startup

Target command:

```powershell
uvicorn mvr.api.app:app --reload
```

Only document it as working after verification.

## Tests

Reuse/adapt FastAPI Lab tests for:

- success response
- empty results
- invalid query
- invalid `top_k`
- response schema
- health endpoint
- sorting/order behavior where applicable

Keep unit tests lightweight.

Separate heavy integration/model tests where needed.

## CI

Inspect existing:

```text
.github/workflows/ci.yml
```

Reuse the working FastAPI Lab CI pattern where helpful.

Do not make CI download huge models or datasets.

Default CI should run lightweight tests.

## Git

Before commit:

```bash
git status -sb
git diff --stat
git diff
pytest -q
```

Suggested commit:

```text
feat: reuse FastAPI serving baseline
```

If substantial test-only work is separate:

```text
test: verify reused MVR baseline
```

## README

Update Quick Start with exact verified commands for:

- environment setup
- tests
- ingestion
- index building
- semantic search
- API startup

Do not claim unverified commands work.

## Finish

Update:

```text
docs/LAB_REUSE_MAP.md
docs/CODEX_PROGRESS.md
README.md
```

Stop after Phase 05.

Do not push or create the PR unless Phase 06 is explicitly requested.
