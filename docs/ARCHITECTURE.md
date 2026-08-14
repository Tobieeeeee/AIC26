# MVR Competition System Architecture

## 1. Purpose

This document defines the architecture of the **Multimedia Video Retrieval (MVR)** competition system.

The repository is designed as a **modular end-to-end retrieval system**, not as a single end-to-end neural network.

The long-term target is:

    Video corpus
        ↓
    Offline preprocessing and indexing
        ↓
    Searchable multimodal representation
        ↓
    User query
        ↓
    Multimodal retrieval
        ↓
    Ranked video/frame results
        ↓
    Competition UI
        ↓
    Submission

The current implementation priority is more limited:

> Reuse existing lab implementations first, package them into production modules, preserve their behavior, verify them, and build the first runnable baseline before introducing advanced improvements.

---

## 2. Architecture Principles

The system follows these principles:

1. **Reuse first**
   - Existing lab implementations are the first source for the baseline.
   - Do not replace working lab logic merely because newer models or libraries exist.

2. **Modular production code**
   - Runtime code belongs under `src/mvr/`.
   - Each module should have a clear responsibility.

3. **Notebooks are not production runtime**
   - Existing lab notebooks are reference material.
   - New notebooks are for experiments, benchmarking, tuning, and visualization.
   - The real system must run without executing Jupyter notebooks.

4. **Offline and online pipelines are separated**
   - Expensive preprocessing happens offline.
   - User-facing search must use precomputed metadata/indexes whenever practical.

5. **Evaluation before improvement**
   - Model or algorithm changes should be supported by benchmark evidence.

6. **Configuration over hard-coding**
   - Important system parameters should be configurable.
   - During the reuse baseline, configuration defaults must preserve original lab behavior.

---

## 3. High-Level System

The target system has two major stages.

### 3.1 Offline pipeline

    Raw videos
        ↓
    Video probing
        ↓
    Shot boundary detection
        ↓
    Keyframe selection
        ↓
    Exact frame extraction
        ↓
    ┌───────────────────────────────────────────────┐
    │ Keyframe enrichment                           │
    │                                               │
    │  OCR        Object Detection       Embedding  │
    │   │               │                   │       │
    │   └───────────────┴───────────────────┘       │
    └─────────────────────────┬─────────────────────┘
                              ↓
                       Metadata builder
                              ↓
                    Search/index preparation
                              ↓
                 FAISS / Elasticsearch / metadata

The offline pipeline converts videos into a searchable representation.

It should not be rerun for every user query.

---

### 3.2 Online pipeline

    User query
        ↓
    Query processing
        ↓
    ┌─────────────────────────────────────────────┐
    │ Retrieval branches                          │
    │                                             │
    │ Semantic      OCR/Text      Object/Metadata │
    │ Retrieval     Retrieval     Retrieval        │
    └───────────────┬─────────────────────────────┘
                    ↓
                Fusion
                    ↓
              Ranked results
                    ↓
                  API
                    ↓
             Competition UI
                    ↓
               Submission

Advanced components may later extend this pipeline:

    Query Understanding
    Temporal Retrieval
    Sequence Retrieval
    VLM Caption Retrieval
    Reranking
    Diversification
    Feedback-based Retrieval

These are **target architecture components**, not automatically part of the first reuse baseline.

---

## 4. First Runnable Reuse Baseline

The first baseline must prioritize integration of existing lab functionality.

A reasonable minimum path is:

    Video
      ↓
    Shot / Keyframe extraction
      ↓
    Visual embedding
      ↓
    Vector index
      ↓
    Text query
      ↓
    Semantic retrieval
      ↓
    Top-K frame results

Then existing lab functionality can be integrated incrementally:

    Semantic baseline
        ↓
    + OCR
        ↓
    + Object metadata
        ↓
    + Elasticsearch / lexical search
        ↓
    + Hybrid fusion
        ↓
    + API

The first baseline should focus on **making existing functionality runnable and modular**.

Do not mix baseline migration with advanced model replacement or tuning.

---

## 5. Source Lab Reuse

Existing lab implementations are stored outside this repository at:

`D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder`

That directory is **READ ONLY**.

The production repository should reuse relevant logic from the labs while reorganizing it by system responsibility.

Typical mapping:

| Existing source | Main reusable role | Production destination |
|---|---|---|
| Shot boundary lab | video probing, shot detection, keyframe extraction, timestamp mapping | `src/mvr/ingestion/` |
| Lab 3 | image/text embeddings and semantic similarity experiments | `src/mvr/enrichment/embeddings/`, `src/mvr/retrieval/` |
| Lab 4 | FAISS retrieval, evaluation, reranking references | `src/mvr/indexing/`, `src/mvr/retrieval/`, `src/mvr/evaluation/` |
| Lab 6 | OCR, object detection, Elasticsearch, hybrid retrieval references | `src/mvr/enrichment/`, `src/mvr/indexing/`, `src/mvr/retrieval/` |
| FastAPI lab | service structure, request/response patterns, tests, CI references | `src/mvr/api/` |

The exact migration must be documented in:

`docs/LAB_REUSE_MAP.md`

Do not organize production code by lab number.

Incorrect:

    src/lab3/
    src/lab4/
    src/lab6/

Correct:

    src/mvr/ingestion/
    src/mvr/enrichment/
    src/mvr/indexing/
    src/mvr/retrieval/

---

## 6. Repository Architecture

The intended production structure is:

    src/mvr/
    ├── ingestion/
    │   ├── video probing
    │   ├── shot detection
    │   ├── keyframe selection
    │   └── frame extraction
    │
    ├── enrichment/
    │   ├── ocr/
    │   ├── objects/
    │   ├── embeddings/
    │   └── captions/
    │
    ├── indexing/
    │   ├── metadata preparation
    │   ├── FAISS
    │   └── Elasticsearch
    │
    ├── retrieval/
    │   ├── semantic
    │   ├── lexical/OCR
    │   ├── object
    │   ├── hybrid
    │   ├── temporal
    │   └── reranking
    │
    ├── query/
    │   └── query understanding and decomposition
    │
    ├── pipeline/
    │   ├── offline orchestration
    │   └── online/search orchestration
    │
    ├── evaluation/
    │   ├── metrics
    │   ├── qrels
    │   ├── benchmarks
    │   └── ablations
    │
    ├── api/
    │   └── serving layer
    │
    ├── submission/
    │   └── competition submission adapter
    │
    └── cli.py

Not every folder must contain a complete implementation during the first baseline.

The structure describes **ownership boundaries**.

---

## 7. Module Responsibilities

### 7.1 `src/mvr/ingestion/`

Responsible for converting videos into stable frame/keyframe references.

Possible responsibilities:

- video metadata probing;
- FPS;
- frame count;
- duration;
- resolution;
- shot boundary detection;
- shot representation;
- keyframe selection;
- frame extraction;
- frame-index to timestamp mapping;
- video/shot/keyframe identifiers.

Expected output:

    video_id
    shot_id
    frame_id
    frame_idx
    timestamp
    image_path

Ingestion must not contain retrieval logic.

---

### 7.2 `src/mvr/enrichment/ocr/`

Responsible for text information visible inside keyframes.

Possible output:

    frame_id
    text
    confidence
    bounding_box

OCR engine selection must initially preserve the reused lab behavior unless an active task explicitly changes it.

---

### 7.3 `src/mvr/enrichment/objects/`

Responsible for structured object information.

Possible output:

    frame_id
    object_label
    count
    confidence
    bounding_boxes

This supports object-aware or metadata-filtered retrieval.

---

### 7.4 `src/mvr/enrichment/embeddings/`

Responsible for image/text embedding models.

Examples may include reused CLIP-family logic from the source labs.

Responsibilities:

- image encoding;
- text encoding;
- normalization;
- batching;
- model/device configuration.

This module should not own FAISS search logic.

---

### 7.5 `src/mvr/enrichment/captions/`

Reserved for future VLM-generated captions.

This is **not required for the first reuse baseline** unless explicitly requested.

---

### 7.6 `src/mvr/indexing/`

Responsible for creating and loading searchable indexes.

Possible implementations:

- FAISS vector index;
- Elasticsearch mapping/index;
- metadata manifests.

Indexing owns persistence and loading of search indexes.

It must not contain query business logic, retrieval ranking logic, or API route logic.

Index construction/loading and query-time retrieval are separate responsibilities.

---

### 7.7 `src/mvr/retrieval/`

Responsible for turning a query into ranked candidates.

Retrieval strategies may include:

- semantic retrieval;
- OCR/text retrieval;
- object retrieval;
- hybrid fusion;
- temporal retrieval;
- reranking.

Each retrieval strategy should remain independently testable.

Hybrid retrieval should consume ranked candidates and/or scores produced by retrieval branches.

Hybrid retrieval must not directly own:

- model loading;
- model inference for unrelated branches;
- index construction;
- index persistence.

Its responsibility is candidate/score fusion and ranking.

---

### 7.8 `src/mvr/query/`

Reserved for future query understanding.

Possible responsibilities:

- query normalization;
- semantic intent;
- OCR intent;
- object constraints;
- temporal constraints;
- query decomposition;
- language normalization/translation.

This is **out of scope for the first reuse baseline** unless explicitly approved.

---

### 7.9 `src/mvr/evaluation/`

Responsible for measuring system quality.

Expected metrics may include:

- Recall@K;
- MRR;
- mAP where appropriate;
- latency;
- indexing time;
- per-query-type performance.

Evaluation should support evidence-based model changes.

---

### 7.10 `src/mvr/api/`

Responsible only for exposing the production system through a service interface.

Possible endpoints:

    GET  /health
    POST /search

Later:

    POST /search/semantic
    POST /search/ocr
    POST /search/object
    POST /search/hybrid

API routes should call reusable services.

Do not build indexes or load heavy models inside request handlers for every request.

---

### 7.11 `src/mvr/submission/`

Reserved for competition-specific submission logic.

Possible responsibilities:

- result-to-submission conversion;
- video/frame/timestamp mapping;
- competition server payload;
- DRES integration.

This is **out of scope for the first reuse baseline** unless explicitly approved.

---
### 7.12 `src/mvr/pipeline/`

Responsible for orchestrating multiple production modules into runnable system flows.

This layer connects existing components without taking ownership of their internal algorithms.

Typical responsibilities include:

- offline preprocessing orchestration;
- ingestion → enrichment → metadata → indexing;
- online search orchestration;
- query → retrieval → fusion → result construction;
- lifecycle coordination for reusable services;
- passing typed data between modules.

Possible future modules:

    src/mvr/pipeline/
    ├── offline.py
    └── search.py

Conceptual offline flow:

    video
      ↓
    ingestion
      ↓
    enrichment
      ↓
    indexing

Conceptual online flow:

    query
      ↓
    retrieval
      ↓
    fusion/reranking
      ↓
    SearchResult

The pipeline layer must not duplicate the algorithms implemented inside:

- `ingestion/`;
- `enrichment/`;
- `indexing/`;
- `retrieval/`.

It should coordinate them.

API routes and CLI commands should call pipeline/services rather than reimplementing orchestration logic themselves.

---

### 7.13 `src/mvr/cli.py`

The CLI is a command-line entry point for running production workflows without requiring Jupyter or the web UI.

Possible commands may later include:

    python -m mvr.cli ingest ...
    python -m mvr.cli build-index ...
    python -m mvr.cli search "query"

The CLI should call reusable pipeline/service modules.

It must not contain the core implementation of ingestion, embedding, indexing, or retrieval algorithms.

The exact command set may evolve with the implementation phases.

## 8. Data and Artifact Architecture

The repository separates source data from generated artifacts.

### `data/raw/`

Original videos or source datasets.

Do not commit large datasets or videos.

---

### `data/interim/`

Intermediate metadata that can be regenerated.

Examples:

- probe results;
- shot boundaries;
- preprocessing manifests.

---

### `data/processed/`

Final metadata/documents ready for indexing.

For large competition corpora, these should normally remain outside Git.

---

### `artifacts/keyframes/`

Generated keyframe images.

Do not commit.

---

### `artifacts/embeddings/`

Generated image/text vectors.

Possible formats:

- `.npy`
- `.npz`
- `.parquet`

Do not commit large generated embedding files.

---

### `artifacts/indexes/`

Generated FAISS or other index artifacts.

Do not commit large runtime indexes.

---
### Artifact Compatibility Metadata

Generated embeddings and indexes must carry enough metadata to verify that they are compatible with the runtime configuration that loads them.

At minimum, relevant artifacts should record information such as:

    schema_version
    embedding_model
    embedding_dimension
    normalization
    index_type
    model/config identifier

Where useful, also record:

    created_at
    source dataset/corpus identifier
    code or configuration version

Conceptual example:

    {
      "schema_version": "1",
      "embedding_model": "openclip/...",
      "embedding_dimension": 512,
      "normalization": "l2",
      "index_type": "IndexFlatIP"
    }

The system must avoid silently using incompatible combinations such as:

    embeddings produced by model A
        +
    query encoder from model B
        +
    index built for model A

Artifact compatibility should be checked when practical before search begins.

Changing a model, embedding dimension, normalization strategy, or index configuration may require rebuilding dependent artifacts.
---

### `artifacts/metrics/`

Benchmark and evaluation outputs.

This may include:

- retrieval metrics;
- latency measurements;
- ablation results;
- local benchmark summaries.

Small curated reports may be committed when useful.

Large generated benchmark artifacts should stay outside Git.

---

## 9. Canonical Metadata Model

A keyframe should eventually be representable using a stable metadata structure similar to:

    frame_id
    video_id
    shot_id
    frame_idx
    timestamp
    image_path

    ocr
    objects
    caption
    embedding reference

    shot_start
    shot_end

Example conceptual document:

    {
      "frame_id": "L01_V003_S010_K01",
      "video_id": "L01_V003",
      "shot_id": "S010",
      "frame_idx": 3520,
      "timestamp": 117.33,
      "image_path": "...",
      "ocr": [...],
      "objects": {...},
      "caption": null,
      "shot_start": 113.0,
      "shot_end": 120.5
    }

The exact schema may evolve, but identifiers and timestamp mapping must remain consistent across modules.

---
### Module Interface Contracts

Modules should communicate through stable, explicit data contracts whenever practical.

Avoid passing undocumented arbitrary dictionaries between production modules.

Conceptual domain objects may include:

    VideoInfo
    Shot
    Keyframe
    OCRResult
    ObjectDetection
    EmbeddingRecord
    SearchCandidate
    SearchResult

For example:

    VideoProbe
        ↓
    VideoInfo

    ShotDetector
        ↓
    List[Shot]

    KeyframeExtractor
        ↓
    List[Keyframe]

    OCREngine
        ↓
    List[OCRResult]

    EmbeddingEncoder
        ↓
    EmbeddingRecord / embedding batch

    Retriever
        ↓
    List[SearchCandidate]

    Pipeline/API
        ↓
    List[SearchResult]

The exact implementation may use:

- Python dataclasses;
- typed structures;
- Pydantic models;
- other explicit typed contracts.

The architecture does not require one specific library.

The important requirement is that module boundaries remain understandable, testable, and stable enough for parallel team development.

Shared identifiers such as:

    video_id
    shot_id
    frame_id
    timestamp

must preserve the same meaning throughout the pipeline.

---

## 10. Production Code vs Notebook Boundary

The repository has two notebook categories.

    notebooks/
    ├── reference/
    │   └── original lab/reference notebooks
    │
    └── experiments/
        └── new experiments, benchmarks, tuning, visualization

Production runtime code belongs under:

`src/mvr/`

Development flow:

    notebook experiment
        ↓
    evaluate
        ↓
    select approach
        ↓
    implement reusable .py module
        ↓
    test
        ↓
    integrate

The production system must never require hidden notebook state.

---

## 11. Offline/Online Dependency Rule

The online system should reuse offline-generated artifacts whenever practical.

Avoid:

    query
      ↓
    reload all models
      ↓
    regenerate all embeddings
      ↓
    rebuild index
      ↓
    search

Prefer:

    startup
      ↓
    load required models/indexes once
      ↓
    query
      ↓
    encode query
      ↓
    search existing index
      ↓
    return results

This separation is important for competition latency.

---

## 12. Retrieval Architecture

The target retrieval design is multi-branch.

### Semantic branch

    text query
       ↓
    text embedding
       ↓
    vector search
       ↓
    semantic candidates

### OCR/Text branch

    text query
       ↓
    lexical/OCR search
       ↓
    OCR candidates

### Object branch

    object constraints
       ↓
    metadata search/filter
       ↓
    object candidates

### Hybrid branch

    semantic candidates
          +
    OCR candidates
          +
    object candidates
          ↓
        fusion
          ↓
    ranked candidates

Existing RRF/weighted fusion utilities belong here.

Advanced reranking is a later layer, not part of the indexing module.

---

## 13. Temporal Architecture

Video frames are not independent images.

The long-term system should support:

- neighboring frame expansion;
- previous/current/next shot navigation;
- time-window retrieval;
- sequence retrieval;
- same-video temporal constraints.

Conceptual example:

    Event A candidates
          +
    Event B candidates
          ↓
    same video
    timestamp(B) > timestamp(A)
    delta < threshold
          ↓
    sequence match

Temporal retrieval is a future capability and should not be introduced during the first reuse baseline unless explicitly requested.

---

## 14. API and UI Boundary

The API is a serving layer.

The UI is a competition interaction layer.

The API should return structured result objects such as:

    video_id
    frame_id
    timestamp
    score
    image path / thumbnail reference
    optional retrieval metadata

The UI may later provide:

- result grids;
- neighboring frames;
- video playback;
- filters;
- query history;
- keyboard shortcuts;
- result selection;
- submission;
- in-session relevance feedback based on previously inspected or selected results.

Relevance feedback is a future competition feature and must not be introduced into the first reuse baseline unless explicitly requested.

Do not place UI-specific logic inside core retrieval modules.

---

## 15. Evaluation Architecture

Every significant retrieval improvement should eventually be evaluated.

Evaluation datasets should contain:

    query
      ↓
    relevant frame/video set

Metrics can include:

- Recall@1;
- Recall@5;
- Recall@10;
- Recall@50;
- MRR;
- mAP where appropriate;
- query latency.

Queries should eventually be categorized, for example:

- semantic;
- OCR;
- object;
- temporal;
- mixed.

Do not conclude that a newer model is better solely because it is newer.

### Competition Data Validation

Results obtained from educational, synthetic, toy, or small demonstration datasets must not be treated as evidence of final competition performance.

Datasets or samples such as Flickr8k, synthetic clips, or lab-sized test sets are useful for:

- smoke testing;
- debugging;
- validating pipeline correctness;
- quick model comparisons during development.

Before making final model, retrieval, or system-design decisions, benchmark on:

1. representative competition data; or
2. the closest available validation set that matches the competition domain.

When representative competition data is unavailable, clearly label results as development-only evidence.

---

## 16. Runtime Performance, Caching, and Monitoring

Interactive video retrieval is sensitive to latency.

The system should eventually measure and monitor:

- end-to-end query latency;
- embedding inference latency;
- OCR/object inference latency where relevant;
- FAISS/Elasticsearch search latency;
- reranking latency;
- API failures;
- model loading failures;
- index loading/search failures;
- unavailable external dependencies;
- frontend/result rendering latency where measurable.

### Caching

Repeated or near-duplicate queries may use caching where appropriate to reduce interactive search latency.

Possible cache targets include:

- text embeddings;
- retrieval results;
- metadata lookups;
- thumbnails;
- neighboring-frame metadata.

Caching must not:

- change retrieval correctness;
- hide stale index state;
- silently mix results from incompatible model/index versions.

Cache keys should include relevant configuration/model/index identity when necessary.

### Runtime Health

The serving layer should eventually expose health information for critical dependencies such as:

- API availability;
- loaded retrieval index;
- required embedding model;
- Elasticsearch connection when used.

Operational monitoring belongs to runtime/service infrastructure, while retrieval-quality metrics belong to `src/mvr/evaluation/`.

These are separate concerns.

---

## 17. Configuration Architecture

Important parameters should be represented in configuration rather than scattered hard-coded values.

Possible configuration groups:

    ingestion
    embedding
    ocr
    object_detection
    indexing
    retrieval
    hybrid
    api

During the reuse baseline:

> Configurable defaults must preserve the original lab behavior.

Any behavior change must be documented in `docs/LAB_REUSE_MAP.md`.

---

## 18. Components Explicitly Out of Scope for the First Reuse Baseline

Unless an active phase explicitly requests them, do not implement:

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
- in-session relevance feedback
- runtime query caching
- DRES integration
- submission adapter
- Docker deployment
- model tuning
- hyperparameter search

These belong to the long-term architecture but not the first migration baseline.

---

## 19. Architecture Evolution

Development should proceed approximately as:

    Phase A
    Reuse and package existing lab components

        ↓

    Phase B
    Build first runnable semantic retrieval baseline

        ↓

    Phase C
    Integrate OCR/object/indexing/hybrid functionality

        ↓

    Phase D
    Add API and system-level integration

        ↓

    Phase E
    Establish reliable evaluation

        ↓

    Phase F
    Introduce advanced retrieval features when justified

This document defines architecture boundaries.

Detailed execution order belongs in:

- `docs/ROADMAP.md`
- `codex/00_MASTER_PLAN.md`
- active phase files

---

## 20. Architectural Completion Criteria

A module should be considered production-integrated only when:

1. reusable logic exists under `src/mvr/`;
2. it does not depend on manually executed notebook state;
3. relevant configuration is explicit;
4. relevant tests or verification exist;
5. lab reuse/behavior changes are documented where required;
6. generated data is kept outside Git;
7. the module fits the ownership boundaries in this document.

An end-to-end baseline should not be claimed as working unless the complete intended path has actually been executed and verified.

---

## 21. Core Architecture Principle

For the current project stage:

    Existing lab work
          ↓
    preserve behavior
          ↓
    modularize into src/mvr/
          ↓
    test and verify
          ↓
    integrate into runnable baseline
          ↓
    benchmark
          ↓
    improve later

The repository architecture exists to turn reusable lab work into a clean, testable, maintainable, competition-oriented Multimedia Video Retrieval system.
