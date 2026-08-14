# AGENTS.md — MVR Competition Repository Rules

## Project purpose

This repository is the shared base for a Multimedia Video Retrieval competition system.

The current priority is to build the first runnable baseline by **reusing existing lab implementations before rewriting or improving them**.

## Source labs

The existing labs are located at:

```text
D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder
```

Treat that directory as **READ ONLY**.

Do not rename, delete, move, or modify source lab files.

## Reuse-first policy

During the first baseline phase:

1. Reuse existing lab implementations before writing replacements.
2. Preserve original algorithms, model choices, thresholds, parameters, normalization logic, similarity calculations, index types, and default behavior whenever practical.
3. Only make minimal compatibility or packaging changes required to make reused code importable and runnable.
4. Any unavoidable behavior change must be documented in `docs/LAB_REUSE_MAP.md`.
5. Do not silently replace an implementation because a newer model or library exists.

## Repository organization

Production code belongs under:

```text
src/mvr/
```

Use functional organization:

```text
src/mvr/
├── ingestion/
├── enrichment/
├── indexing/
├── retrieval/
├── evaluation/
└── api/
```

Do not create production folders such as:

```text
src/lab3/
src/lab4/
src/lab6/
src/shot_lab/
```

Reference notebooks belong under:

```text
notebooks/reference/
```

Notebook code must not remain the only implementation of a production feature.

## Files that must not be committed

Never commit:

- `.venv/`, `venv/`, `env/`
- `__pycache__/`, `.pytest_cache/`
- `.env`, credentials, API keys, secrets
- datasets
- raw videos
- generated keyframes
- generated embeddings
- FAISS indexes
- Elasticsearch runtime data
- model weights
- temporary outputs
- large generated benchmark artifacts

Do not delete local files just because they are ignored by Git.

## Git rules

- Feature work must not be performed directly on `main`.
- Use the branch `feat/reuse-lab-baseline` for this migration unless the active task explicitly specifies otherwise.
- Run relevant tests before committing.
- Use logical commits.
- Never force-push.
- Do not rewrite unrelated Git history.
- Do not overwrite an unrelated `origin` remote.

## Scope restrictions

Do not implement future features unless the active task explicitly requests them.

Out of scope for the first reuse baseline:

- Query Understanding
- Temporal Retrieval
- Sequence Retrieval
- VLM caption generation
- new embedding-model experiments
- LLM query decomposition
- advanced reranking
- advanced ANN indexes
- feedback learning
- competition UI
- DRES integration
- submission adapter
- Docker deployment
- model tuning
- hyperparameter search

## Working style

For each phase:

1. Read this file.
2. Read `codex/00_MASTER_PLAN.md`.
3. Read the active phase file.
4. Read `docs/CODEX_PROGRESS.md`.
5. Execute only the active phase.
6. Run tests/verification required by that phase.
7. Update `docs/CODEX_PROGRESS.md`.
8. Commit logical changes.
9. Stop and report.

Do not automatically start the next phase.
