# Codex Progress

This file is the persistent checkpoint for the lab-reuse migration.

Codex must update it at the end of every phase.

## Phase 01 — Discover and Map

Status: COMPLETE

Commit: `docs: map reusable lab implementations` (this phase commit; hash reported after commit)

Completed:
- Installed the Codex prompt pack at the required repository-root paths.
- Located the Shot Boundary Lab, Lab 3, Lab 4, Lab 6, and FastAPI Lab in the read-only source root.
- Inspected the source implementations for every component required by Phase 01.
- Added 47 provenance-backed component mappings to `docs/LAB_REUSE_MAP.md`.
- Verified that all three repository reference notebooks are byte-identical to their source originals.
- Documented behavior contracts, compatibility concerns, deferred experiments, and target destinations.
- Kept production code unchanged and did not begin migration.

Remaining:
- None for Phase 01.

Tests:
- `python -m pytest -q`: 4 passed.
- Reuse-map validation: 47 component rows, all `MAPPED`, no `TBD`/`TODO` placeholders.
- Production diff check (`src/`, `tests/`, `configs/`, `pyproject.toml`, `.github/`): no changes.
- Read-only source integrity: unchanged at 11,177 files / 1,221,999,740 bytes / metadata SHA-256 `07b34aa47d766429a2667a682d504df837a304b707c9b999aacc737ef057bfbd`.

Blockers:
- None.

---

## Phase 02 — Reuse Ingestion

Status: NOT STARTED

Commit: N/A

Completed:
- None

Remaining:
- Reuse Shot Boundary Lab.
- Verify ingestion CLI/pipeline.

Tests:
- N/A

Blockers:
- None recorded.

---

## Phase 03 — Semantic End-to-End Baseline

Status: NOT STARTED

Commit: N/A

Completed:
- None

Remaining:
- Reuse OpenCLIP from Lab 3.
- Reuse FAISS from Lab 4.
- Connect ingestion → embedding → index → text search.
- Verify first E2E semantic baseline.

Tests:
- N/A

Blockers:
- None recorded.

---

## Phase 04 — Multimodal Reuse

Status: NOT STARTED

Commit: N/A

Completed:
- None

Remaining:
- EasyOCR
- PaddleOCR
- YOLOv8n
- Elasticsearch
- BM25
- object search
- hybrid retrieval

Tests:
- N/A

Blockers:
- None recorded.

---

## Phase 05 — API and Tests

Status: NOT STARTED

Commit: N/A

Completed:
- None

Remaining:
- Reuse FastAPI patterns.
- Add verified endpoints.
- Complete lightweight tests/CI.

Tests:
- N/A

Blockers:
- None recorded.

---

## Phase 06 — Verify and Publish

Status: NOT STARTED

Commit: N/A

Completed:
- None

Remaining:
- Final audit.
- Final tests.
- Push feature branch.
- Create PR to `main`.

Tests:
- N/A

Git:
- Branch: TBD
- Remote: TBD
- Push: NOT STARTED
- PR: NOT CREATED

Blockers:
- None recorded.

---

# Current active phase

Phase 01 — Discover and Map: COMPLETE

Next phase: Phase 02 — Reuse Ingestion (NOT STARTED).

Stop gate: do not begin Phase 02 without an explicit instruction.

# Resume instruction

If work is interrupted, read:

```text
AGENTS.md
codex/00_MASTER_PLAN.md
docs/CODEX_PROGRESS.md
```

Then execute only the currently active phase file.
