# AGENTS.md — MVR Competition Repository Rules

## 0. Mandatory First Step — Read AGENTS.md

Before doing anything in this repository, including analysis, planning, editing, coding, testing, Git operations, or executing commands:

1. Read `AGENTS.md` completely.
2. Treat `AGENTS.md` as the repository-level source of truth for development rules.
3. Do not modify files or execute implementation work before reading and understanding these instructions.
4. If the active task conflicts with `AGENTS.md`, stop and report the conflict instead of silently choosing one.
5. If `AGENTS.md` cannot be found or read, stop and report the blocker.
6. After reading `AGENTS.md`, read any additional files required by the active task, such as:
   - `README.md`
   - `docs/ARCHITECTURE.md`
   - `docs/ROADMAP.md`
   - `docs/GIT_WORKFLOW.md`
   - `codex/00_MASTER_PLAN.md`
   - the active phase file
   - `docs/CODEX_PROGRESS.md`

Do not skip this step even if the task appears simple or similar to a previous task.

**Rule: Read `AGENTS.md` first, then act.**
## Project Purpose

This repository is the shared base for a **Multimedia Video Retrieval (MVR)** competition system.

The current priority is to build the first runnable baseline by **reusing existing lab implementations before rewriting or improving them**.

All coding agents, including Codex, must follow the rules in this file.

---

## 1. Source Labs

The existing labs are located at:

`D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder`

Treat that directory as **READ ONLY**.

Do not:

- rename source lab files;
- delete source lab files;
- move source lab files;
- modify source lab files in place.

If lab code must be reused, copy or adapt the necessary logic into the repository while preserving the original source files.

---

## 2. Reuse-First Policy

During the first baseline phase:

1. Reuse existing lab implementations before writing replacements.
2. Preserve original algorithms, model choices, thresholds, parameters, normalization logic, similarity calculations, index types, and default behavior whenever practical.
3. Only make minimal compatibility or packaging changes required to make reused code importable and runnable.
4. Any unavoidable behavior change must be documented in:
   - `docs/LAB_REUSE_MAP.md`
5. Do not silently replace an implementation because a newer model or library exists.
6. Do not optimize, tune, or redesign reused components unless the active task explicitly requests it.

The first objective is:

**Reuse first → make it runnable → verify behavior → improve later.**

---

## 3. Required Reading Before Coding

Before implementing any feature or migration task, read:

- `AGENTS.md`
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
- `docs/GIT_WORKFLOW.md`

When working on the reuse baseline, also read:

- `codex/00_MASTER_PLAN.md`
- the active phase file
- `docs/CODEX_PROGRESS.md`
- `docs/LAB_REUSE_MAP.md` if it already exists

Also inspect the relevant existing source modules and tests before making changes.

Do not start implementation before understanding:

- the active task;
- the current branch;
- the existing architecture;
- the source lab implementation being reused.

---

## 4. Repository Organization

Production code belongs under:

`src/mvr/`

Use functional organization:

    src/mvr/
    ├── ingestion/
    ├── enrichment/
    │   ├── ocr/
    │   ├── objects/
    │   ├── embeddings/
    │   └── captions/
    ├── indexing/
    ├── retrieval/
    ├── query/
    ├── evaluation/
    ├── api/
    └── submission/

Do not create production folders such as:

    src/lab3/
    src/lab4/
    src/lab6/
    src/shot_lab/

The repository is organized by **system responsibility**, not by the lab where the code originally came from.

---

## 5. Production Code Policy

All code used by the actual MVR system must be implemented as importable Python modules under:

`src/mvr/`

Production code includes:

- video ingestion
- video probing
- shot boundary detection
- keyframe extraction
- OCR
- object detection
- image/text embeddings
- metadata construction
- FAISS indexing
- Elasticsearch indexing
- semantic retrieval
- OCR retrieval
- object retrieval
- hybrid retrieval
- query understanding
- temporal retrieval
- reranking
- evaluation logic
- API serving
- submission logic

Production functionality must not exist only inside a Jupyter notebook.

Production modules should be:

- reusable;
- importable;
- testable;
- configurable where appropriate.

Example:

    src/mvr/
    ├── ingestion/
    │   ├── video_probe.py
    │   ├── shot_detector.py
    │   └── keyframe_extractor.py
    │
    ├── enrichment/
    │   ├── embeddings/
    │   │   └── clip_encoder.py
    │   ├── ocr/
    │   │   └── ocr_engine.py
    │   └── objects/
    │       └── object_detector.py
    │
    ├── indexing/
    │   └── faiss_index.py
    │
    └── retrieval/
        └── semantic.py

---

## 6. Jupyter Notebook Policy

Reference notebooks belong under:

`notebooks/reference/`

New experimental notebooks belong under:

`notebooks/experiments/`

Jupyter notebooks are allowed for:

- exploratory experiments;
- model comparison;
- benchmarking;
- parameter tuning;
- visualization;
- ablation studies;
- demonstrations.

Notebook code must not remain the only implementation of a production feature.

The production system must be able to run without Jupyter.

Do not implement the production pipeline as one large notebook.

---

## 7. Experiment Promotion Rule

When experimentation is necessary, follow this workflow:

    .ipynb experiment
            ↓
    evaluate result
            ↓
    select approach
            ↓
    extract reusable logic
            ↓
    .py module under src/mvr/
            ↓
    unit/integration tests
            ↓
    feature integration

If an experiment is successful and is required by the real system, move the reusable implementation into `src/mvr/`.

Do not make production modules import from notebooks.

Whenever practical, notebooks should import production modules instead of duplicating their logic.

---

## 8. File Selection Rule

When implementing a feature:

- use `.py` by default;
- use `.ipynb` only when experimentation, comparison, visualization, or tuning is required;
- reusable logic belongs in `.py`;
- production runtime must not depend on notebook execution state.

Correct:

    notebooks/experiments/compare_clip_siglip.ipynb
    src/mvr/enrichment/embeddings/clip_encoder.py
    src/mvr/enrichment/embeddings/siglip_encoder.py

Incorrect:

    notebooks/final_system.ipynb

containing the complete OCR + YOLO + CLIP + FAISS + API implementation.

---

## 9. Architecture Boundaries

Keep responsibilities separated.

### Ingestion

    Video
      ↓
    Shot Detection
      ↓
    Keyframe Extraction

Target area:

`src/mvr/ingestion/`

### Enrichment

Includes:

- OCR
- object detection
- embeddings
- optional captions

Target area:

`src/mvr/enrichment/`

### Indexing

Includes:

- metadata indexes
- FAISS
- Elasticsearch mappings
- persistence/loading logic

Target area:

`src/mvr/indexing/`

### Retrieval

Includes:

- semantic retrieval
- OCR retrieval
- object retrieval
- hybrid retrieval
- temporal retrieval
- reranking

Target area:

`src/mvr/retrieval/`

### Query Understanding

Target area:

`src/mvr/query/`

### Evaluation

Target area:

`src/mvr/evaluation/`

### API

Target area:

`src/mvr/api/`

### Submission

Target area:

`src/mvr/submission/`

Do not put unrelated responsibilities into one large module.

Do not move production logic back into notebooks.

---

## 10. Target End-to-End System Direction

The intended long-term architecture is:

### Offline Pipeline

    Video
      ↓
    Shot Detection
      ↓
    Keyframe Extraction
      ↓
    OCR + Object Detection + Embeddings + optional VLM Caption
      ↓
    Metadata
      ↓
    FAISS / Elasticsearch

### Online Pipeline

    User Query
      ↓
    Query Understanding
      ↓
    Semantic / OCR / Object / Temporal Retrieval
      ↓
    Hybrid Fusion
      ↓
    Reranking
      ↓
    Results
      ↓
    API
      ↓
    Competition UI
      ↓
    Submission

This is the target architecture, not permission to implement every stage immediately.

Active task scope always takes priority.

---

## 11. Files That Must Not Be Committed

Never commit:

- `.venv/`
- `venv/`
- `env/`
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `.env`
- credentials
- API keys
- secrets
- datasets
- raw videos
- generated keyframes
- generated embeddings
- FAISS indexes
- Elasticsearch runtime data
- model weights
- temporary outputs
- large generated benchmark artifacts

Generated/runtime data should normally live under:

- `data/`
- `artifacts/`

Do not delete local files merely because they are ignored by Git.

If a generated or large file is accidentally tracked, remove it from Git tracking without deleting the user's local copy unless explicitly instructed otherwise.

---

## 12. Git Rules

Feature work must not normally be performed directly on `main`.

For the lab reuse migration baseline, use:

`feat/reuse-lab-baseline`

unless the active task explicitly specifies another branch.

Other feature branches may include:

- `feat/e2e-baseline`
- `feat/shot-keyframe`
- `feat/semantic-retrieval`
- `feat/ocr-object`
- `feat/indexing`
- `feat/hybrid-search`
- `feat/query-understanding`
- `feat/temporal-retrieval`
- `feat/reranking`
- `feat/vlm-caption`
- `feat/evaluation`
- `feat/api`
- `feat/ui`
- `feat/submission`
- `feat/deployment`

Before coding, run:

    git status -sb
    git branch --show-current

Confirm that the current branch matches the active task.

Do not:

- force-push;
- rewrite unrelated Git history;
- silently discard user changes;
- commit unrelated files;
- overwrite an unrelated `origin` remote;
- merge into `main` without explicit instruction.

Use logical commit messages such as:

- `feat: reuse shot detection pipeline from lab`
- `feat: add CLIP embedding encoder`
- `fix: preserve original similarity normalization`
- `test: add semantic retrieval regression tests`
- `docs: document lab reuse mapping`

---

## 13. Scope Restrictions for the First Reuse Baseline

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

The first reuse baseline should focus on integrating existing lab functionality with minimal behavioral change.

---

## 14. Feature Scope Rule

Only modify files required for the active task.

If another module is required but outside the approved scope:

1. define a clean interface if necessary;
2. document the dependency;
3. do not implement unrelated features without approval.

Avoid large unrelated refactors.

Do not automatically "improve" reused code during migration.

---

## 15. Testing Rules

Reusable production code should have tests.

Use:

- `tests/unit/` for isolated modules;
- `tests/integration/` for cross-module pipelines.

A feature is not complete merely because a notebook cell runs successfully.

Where practical, verify using:

    pytest

For reused lab behavior, add regression tests where practical to confirm that packaging/refactoring has not changed expected behavior.

Do not download huge models solely to satisfy lightweight CI tests.

Use:

- mocks;
- small fixtures;
- dependency injection;
- synthetic test data;

when appropriate.

---

## 16. Dependency Rule

Avoid adding dependencies unnecessarily.

Before adding a library:

1. inspect existing dependencies;
2. check whether the source lab already depends on a suitable library;
3. reuse existing libraries when reasonable;
4. add only what the active feature requires;
5. document important new dependencies.

Do not silently replace an existing dependency with a newer alternative during the reuse baseline.

Do not download large ML model weights during repository setup or lightweight tests unless explicitly requested.

---

## 17. Configuration Rule

Do not hard-code important system or experiment parameters deep inside modules when they should be configurable.

Prefer configuration under:

`configs/`

Examples:

- model name
- device
- batch size
- keyframe interval
- shot detection parameters
- OCR engine
- FAISS settings
- retrieval top-k
- hybrid weights
- temporal window

However, during the reuse-first phase:

**preserve original lab defaults first.**

If a parameter is moved into config, the configured default should preserve the original lab behavior unless the active task explicitly approves a behavior change.

---

## 18. Lab Reuse Documentation Rule

Maintain:

`docs/LAB_REUSE_MAP.md`

for migrated/reused functionality.

For each reused component, record:

- source lab/file;
- original component/function;
- target production module;
- original behavior preserved;
- compatibility changes made;
- behavior changes, if unavoidable;
- tests or verification performed.

Do not hide behavior changes inside refactoring.

---

## 19. Working Style for Phased Codex Execution

For each phase:

1. Read `AGENTS.md`.
2. Read `codex/00_MASTER_PLAN.md`.
3. Read the active phase file.
4. Read `docs/CODEX_PROGRESS.md`.
5. Inspect the relevant source lab files.
6. Inspect the current production modules and tests.
7. Confirm the active Git branch.
8. Execute only the active phase.
9. Run the tests/verification required by that phase.
10. Update `docs/LAB_REUSE_MAP.md` when lab code is reused or behavior is changed.
11. Update `docs/CODEX_PROGRESS.md`.
12. Inspect the final diff.
13. Commit logical changes if the phase requires a commit.
14. Stop and report.

Do not automatically start the next phase.

---

## 20. Completion Rule

Before reporting a feature or phase as complete:

1. inspect the final diff;
2. run relevant tests;
3. verify imports;
4. ensure production functionality does not depend on manually executed notebook state;
5. confirm reused lab behavior is preserved unless documented otherwise;
6. verify no generated artifacts, datasets, model weights, or secrets are staged;
7. update required progress/reuse documentation;
8. summarize exactly what was implemented;
9. summarize what remains unfinished.

Do not claim an end-to-end feature works unless it has actually been executed or validated end-to-end.

Do not claim a reused implementation is behavior-equivalent unless that has been verified.

---

## 21. Core Principle

For the current stage of this project, follow this priority:

    Reuse existing lab implementation
            ↓
    preserve behavior
            ↓
    package into src/mvr/
            ↓
    test and verify
            ↓
    integrate into runnable baseline
            ↓
    improve only in later approved phases

The goal is not to rewrite everything from scratch.

The goal is to turn the existing lab work into a clean, testable, modular, runnable Multimedia Video Retrieval system.
