# 02 — Reuse Shot Boundary Ingestion

## Goal

Reuse the existing Shot Boundary Lab implementation as the production ingestion baseline.

Do not rewrite the algorithms.

## Required preflight

Read:

```text
AGENTS.md
codex/00_MASTER_PLAN.md
codex/02_REUSE_INGESTION.md
docs/LAB_REUSE_MAP.md
docs/CODEX_PROGRESS.md
```

Confirm Phase 01 is complete.

## Source of truth

Use the actual Shot Boundary Lab located under:

```text
D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder
```

Expected reusable files may include equivalents of:

```text
probe.py
models.py
keyframes.py
extract.py
metadata.py
pipeline.py
detectors/pyscenedetect.py
detectors/transnetv2_pytorch.py
```

Verify actual paths first.

## Target structure

Place reused production code under approximately:

```text
src/mvr/ingestion/
├── __init__.py
├── models.py
├── probe.py
├── keyframes.py
├── extract.py
├── metadata.py
├── pipeline.py
└── shot_detection/
    ├── __init__.py
    ├── pyscenedetect.py
    └── transnetv2.py
```

Do not create empty files without purpose beyond required package `__init__.py`.

## Reuse policy

Preserve original:

- shot boundary behavior
- PySceneDetect logic
- TransNetV2 logic
- keyframe midpoint logic
- long-shot sampling logic
- frame/timestamp semantics
- FFmpeg extraction strategy
- metadata semantics
- thresholds and defaults

Allowed changes:

- import path fixes
- namespace changes
- converting notebook/global assumptions into function parameters
- path handling
- CLI wrapper
- tests
- small compatibility fixes

Any behavior change must be documented in `docs/LAB_REUSE_MAP.md`.

## Runnable requirement

Provide a CLI or script that performs:

```text
video
  ↓
probe
  ↓
shot detection
  ↓
keyframe selection
  ↓
frame extraction
  ↓
metadata mapping
```

Prefer a command similar to:

```powershell
python -m mvr.cli ingest "<video-or-directory>"
```

If another command is cleaner, document the exact verified command.

## Tests

Reuse or adapt existing Shot Boundary Lab tests when practical.

Add lightweight tests for:

- frame/timestamp mapping
- metadata models
- keyframe selection logic
- detector interface behavior that does not require large model downloads

Do not download huge models merely for unit testing.

## Git

Before commit:

```bash
git status -sb
git diff --stat
git diff
pytest -q
```

If some tests require unavailable heavy dependencies, report them honestly.

Suggested commit:

```text
feat: reuse shot boundary ingestion pipeline
```

## Finish

Update:

```text
docs/LAB_REUSE_MAP.md
docs/CODEX_PROGRESS.md
README.md
```

README should document only verified ingestion commands.

Stop after Phase 02.

Do not begin OpenCLIP/FAISS work.
