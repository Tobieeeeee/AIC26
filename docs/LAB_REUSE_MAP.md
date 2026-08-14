# Lab Reuse Map

This file records source provenance and verification for the first MVR baseline. Phase 01 mapped the labs, Phase 02 migrated ingestion, Phase 03 migrated the OpenCLIP/FAISS engineering baseline, and Phase 04 migrated Lab 6 multimodal adapters and retrieval request logic.

## Source inventory

Read-only source root:

```text
D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder
```

| Expected lab | Verified source location | Form | Notes |
|---|---|---|---|
| Shot Boundary Lab | `shot-boundary-lab-student/student/` | Python package | Contains probe, models, detectors, keyframe selection/extraction, metadata, pipeline, CLI, and tests. |
| Lab 3 | `SkillPixel_Lab3_Homework_Completed.ipynb` | Notebook | OpenCLIP baseline plus Vietnamese, SigLIP/SigLIP2, optional GVE, and concatenation experiments. |
| Lab 4 | `HCM_AIC_Lab4_semifinal.ipynb` | Notebook | OpenCLIP, exact FAISS search, retrieval metrics/latency, dense candidates, and CrossEncoder reranking. |
| Lab 6 | `HCM_AIC_Lab6.ipynb` | Notebook | OCR, objects, CLIP, Elasticsearch, retrieval/fusion, evaluation, and FastAPI prototype. |
| FastAPI Lab | `lab1-fastapi-student/` | Python app | FastAPI/Pydantic structure, query validation, tests, and GitHub Actions workflow. |

The three notebook copies already under `notebooks/reference/` have the same SHA-256 hashes as their source originals. Source datasets, videos, virtual environments, generated outputs, model weights, Elasticsearch distributions, and `New folder.rar` are evidence/artifacts, not migration inputs.

## Component mapping

`MAPPED` means the source implementation and intended destination were verified. `MIGRATED` means code is present but its runtime was not fully exercised. `VERIFIED` means the target behavior was covered by lightweight tests or an explicitly documented real-runtime smoke run.

Notebook cell numbers below are zero-based JSON cell indices.

| Source Lab | Original Source | Reused Component | Destination | Expected/Actual Changes | Status |
|---|---|---|---|---|---|
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/models.py` | `VideoInfo`, `Shot`, and `Keyframe` data contracts | `src/mvr/ingestion/models.py` | Package rename only. Frozen dataclasses, zero-based frames, inclusive `Shot.end_frame`, computed time fields, and keyframe fields are preserved. | VERIFIED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/probe.py` | `ffprobe` video probing | `src/mvr/ingestion/probe.py` | Package-relative import only. FPS fraction parsing, duration fallback, and frame-count fallback are preserved. | VERIFIED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/detectors/__init__.py` | Detector protocol and factory | `src/mvr/ingestion/shot_detection/__init__.py` | Namespace changed from `detectors` to `shot_detection`; lazy imports and the detector contract are preserved. The TransNet module import follows its destination filename. | VERIFIED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/detectors/pyscenedetect.py` | PySceneDetect `ContentDetector` | `src/mvr/ingestion/shot_detection/pyscenedetect.py` | Package import only. Threshold `27.0`, `start_in_scene=True`, exclusive-to-inclusive conversion, clamping, and full-video fallback are preserved. | VERIFIED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/detectors/transnetv2_pytorch.py` | TransNetV2 PyTorch detector | `src/mvr/ingestion/shot_detection/transnetv2.py` | Filename/import adaptation only. Threshold `0.5`, device selection, one model load, package weights with `weights_only=True`, clamping, and fallback are preserved. Heavy runtime dependencies remain lazy/optional and were not installed in Phase 02. | MIGRATED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/keyframes.py` | Keyframe frame selection | `src/mvr/ingestion/keyframes.py` | Package import only. Floor midpoint, `duration >= 12s`, `8s` sampling, sorted uniqueness, validation, and bounds are preserved. | VERIFIED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/keyframes.py` | Keyframe IDs and metadata construction | `src/mvr/ingestion/keyframes.py` | IDs, `frame_idx / fps` timestamps, ordinals, and POSIX image paths are preserved. | VERIFIED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/extract.py` | Exact FFmpeg frame extraction | `src/mvr/ingestion/extract.py` | Verbatim source with only EOF newline normalization. One `rgb24` decoder scan, early stop, duplicate-index fan-out, FFmpeg JPEG quality, and missing-frame checks are preserved. | VERIFIED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/metadata.py` | Shot/keyframe metadata writers | `src/mvr/ingestion/metadata.py` | Package import only. `shots.json` schema `1.0`, indexing declarations, CSV columns, benchmark files, and summary calculations are preserved. | VERIFIED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/pipeline.py` | Video discovery and pipeline orchestration | `src/mvr/ingestion/pipeline.py` | Only the detector package import points to `shot_detection`. Extensions, thresholds, detector reuse, primary selection, per-video extraction, timing, diagnostics, and output sequence are preserved. | VERIFIED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/src/shotlab/cli.py` | Ingestion CLI defaults | `src/mvr/cli.py` | Wrapped as the `ingest` subcommand; `--input` became the positional `input`. Required `--output`, methods, primary, device choices, thresholds `27.0`/`0.5`, and sampling defaults `12s`/`8s` are preserved. Postponed annotations support the declared Python 3.10+ range. | VERIFIED |
| Shot Boundary Lab | `shot-boundary-lab-student/student/tests/test_models.py`, `test_keyframes.py`, `test_pipeline_progress.py` | Ingestion behavior tests | `tests/unit/ingestion/`, `tests/unit/test_cli.py` | Imports/temporary paths adapted; inclusive-end, midpoint/long-shot, metadata, probing, detector, extraction no-op, orchestration, error propagation, and CLI defaults are covered. | VERIFIED |
| Lab 3 | `SkillPixel_Lab3_Homework_Completed.ipynb`, cells 2-3 | OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` loading and preprocessing | `src/mvr/enrichment/embeddings/openclip.py` | Extracted into a lazy reusable adapter. Model/pretrained IDs, `eval()`, RGB conversion, OpenCLIP preprocessing, and image batch size `32` are preserved. Added explicit device selection and actionable optional-dependency errors. | VERIFIED |
| Lab 3 | Same notebook, cells 3 and 13 | Batched image encoding | `src/mvr/enrichment/embeddings/openclip.py` | Notebook globals became instance fields; `torch.no_grad`, stacking, `encode_image`, dimension-wise L2 normalization, CPU NumPy output, and contiguous `float32` output are preserved. | VERIFIED |
| Lab 3 | Same notebook, cells 5 and 13 | Batched text encoding | `src/mvr/enrichment/embeddings/openclip.py` | Packaged without duplicating model logic; tokenizer use, `encode_text`, `torch.no_grad`, dimension-wise L2 normalization, and text batch size `64` are preserved. | VERIFIED |
| Lab 3 | Same notebook, cells 4-5 | Embedding and metadata persistence | `src/mvr/indexing/semantic_builder.py`, ignored `artifacts/indexes/` | The `.npy` embedding matrix is preserved. The notebook's Parquet mapping is compatibility-adapted to read Phase 02 `keyframe_mapping.csv` and persist a lightweight JSON row map/manifest beside FAISS, avoiding pandas/pyarrow while preserving one metadata row per embedding. Relative Phase 02 image paths become resolved absolute paths so standalone search results locate the source JPEGs. | VERIFIED |
| Lab 3 | Same notebook, cells 5-6 | Dot-product/cosine retrieval and Recall@K/MRR | `src/mvr/retrieval/semantic.py`, `src/mvr/evaluation/metrics.py` | Normalized dot product, descending rank, 1-based reciprocal rank, Recall@K, and MRR are preserved. The Lab 4 multi-relevant-ID contract is also exposed. | VERIFIED |
| Lab 3 | Same notebook, cells 15-16 | English-to-Vietnamese NLLB query experiment | `notebooks/reference/` and future evaluation utilities | Already preserved as reference. Defer production migration; translation/model experiments are outside the first baseline. | MAPPED |
| Lab 3 | Same notebook, cells 18-20 | SigLIP and SigLIP2 evaluation adapters | `notebooks/reference/` and future embedding adapters | Already preserved as reference. Do not replace the selected OpenCLIP baseline during reuse phases. | MAPPED |
| Lab 3 | Same notebook, cell 22 | Optional GVE/Qwen2.5-VL embeddings | `notebooks/reference/` | Keep deferred (`RUN_QWEN25_GVE=False`); VLM embedding experiments are explicitly out of scope. | MAPPED |
| Lab 3 | Same notebook, cell 24 | OpenCLIP + SigLIP2 concatenation fusion | `notebooks/reference/` and future evaluation | Preserve normalization, square-root alpha weighting, concatenation, and final normalization if revisited. Not part of the first baseline. | MAPPED |
| Lab 4 | `HCM_AIC_Lab4_semifinal.ipynb`, cells 13 and 15 | Contiguous normalized `float32` vectors | `src/mvr/enrichment/embeddings/openclip.py`, `src/mvr/indexing/faiss_index.py` | L2 normalization before inner-product indexing and contiguous `float32` conversion at the FAISS boundary are preserved. The wrapper rejects non-finite or non-unit vectors/queries rather than silently changing cosine semantics. | VERIFIED |
| Lab 4 | Same notebook, cell 17 | Exact FAISS `IndexFlatIP` construction | `src/mvr/indexing/faiss_index.py` | Packaged index creation/add without changing exact inner-product behavior. `write_index`/`read_index`, JSON row metadata, and a manifest were added for required persistence; load validates index type, shape/count, normalization declaration, and required metadata fields. The semantic CLI separately requires model/pretrained IDs. No IVF/HNSW/PQ substitution was made. | VERIFIED |
| Lab 4 | Same notebook, cell 19 | Text-to-image top-K search | `src/mvr/retrieval/semantic.py`, `src/mvr/cli.py` | Plotting was separated from retrieval; non-empty query validation, normalized OpenCLIP text vectors, `IndexFlatIP.search`, score order, top-K clamping, and row metadata lookup are preserved. CLI JSON adds stable result fields. | VERIFIED |
| Lab 4 | Same notebook, cells 7 and 21 | Recall@K, Hit@K, MRR, and FAISS latency | `src/mvr/evaluation/metrics.py` | Consolidated single/multiple-ground-truth helpers while preserving relevant-set Recall, Hit, 1-based reciprocal rank, and explicit top-K behavior. Latency reporting was not migrated because it is not required by the Phase 03 vertical slice. | VERIFIED |
| Lab 4 | Same notebook, cells 29, 31, and 33 | Dense candidate retrieval with FAISS | `src/mvr/retrieval/semantic.py` or a future text-retrieval module | Reusable pattern, but MiniLM/SciFact is not the Phase 03 video baseline. Preserve normalized vectors and candidate limits if later adopted. | MAPPED |
| Lab 4 | Same notebook, cells 28 and 35 | CrossEncoder candidate reranking | Future `src/mvr/retrieval/reranking.py` | Keep as a deferred reference; advanced reranking is out of the first reuse baseline. Preserve two-stage candidate-then-rerank semantics if implemented later. | MAPPED |
| Lab 4 | Same notebook, cells 37 and 39 | End-to-end retrieval evaluation | `src/mvr/evaluation/metrics.py` | Preserve query-level timing and aggregation across Recall, Hit, and reciprocal rank; parameterize datasets and search callables. | MAPPED |
| Lab 6 | `HCM_AIC_Lab6.ipynb`, cell 6 | EasyOCR (`vi`, `en`) | `src/mvr/enrichment/ocr/easyocr_engine.py` | Extracted a lazy adapter preserving confidence threshold `0.3`, automatic CUDA GPU choice, joined accepted text, and one reader initialization. EasyOCR 1.7.2 ran on five real keyframes; frame 1 produced `00:00:00` (`0.982596`) and `480` (`0.999869`). No quality conclusion was made. | VERIFIED |
| Lab 6 | Same notebook, cells 7-8 | PaddleOCR Vietnamese pipeline | `src/mvr/enrichment/ocr/paddleocr_engine.py` | Preserved 3.x-first/2.x-fallback arguments, `predict`/`ocr` compatibility, nested result traversal, missing-score behavior, threshold `0.3`, deduplication, and joined final PaddleOCR text. PaddleOCR 3.7.0/PaddlePaddle 3.3.1 ran on all five keyframes and supplied final indexed OCR text. | VERIFIED |
| Lab 6 | Same notebook, cells 6 and 8 | OCR `word_recall` comparison | `src/mvr/evaluation/ocr_metrics.py` | Preserved lowercase whitespace-token membership recall and hit threshold `0.60`; lightweight tests use the original definition. The extracted utility raises a clear error for empty ground truth, which the notebook excluded via its stamped-frame filter. No OCR quality comparison was run. | VERIFIED |
| Lab 6 | Same notebook, cell 10 | YOLOv8n object counts | `src/mvr/enrichment/objects/yolo.py` | Preserved `yolov8n.pt`, confidence `0.4`, `verbose=False`, COCO name lookup, first result, and per-frame `Counter`. Ultralytics 8.4.120 and the real weights ran on five keyframes; all counts were empty because the tiny synthetic corpus contained no detectable COCO objects. | VERIFIED |
| Lab 6 | Same notebook, cell 11 | OpenCLIP RN50 image embeddings | `src/mvr/enrichment/embeddings/openclip.py`, `src/mvr/retrieval/semantic.py`, `configs/default.yaml` | Reused the existing configurable OpenCLIP adapter behind a Lab 6 wrapper that applies `RN50`/`openai` and default image/text batch `64`. Real CPU inference produced five `float32` 1024-dimensional vectors with unit L2 norms. This remains separate from Phase 03's ViT-B-32 defaults. | VERIFIED |
| Lab 6 | Same notebook, cell 12 | Search document schema | `src/mvr/indexing/documents.py` | Added a typed `frame_id`, `video_id`, `ts`, `caption`, `ocr_text`, `objects`, and `clip_vector` representation plus a Phase 02 mapping adapter. The Lab 6 `ts` field is preserved while accepting Phase 02 `timestamp`; missing/blank frame IDs are rejected before they can collide as Elasticsearch document IDs. | VERIFIED |
| Lab 6 | Same notebook, cells 14-15 | Elasticsearch lifecycle, analyzer, mapping, and bulk indexing | `src/mvr/indexing/elasticsearch_index.py` | Preserved ES 8.x client semantics, `vi_folded` lowercase+asciifolding analyzer, keyword/time/text/object fields, indexed cosine dense vector, destructive explicit recreate, frame-ID document IDs, bulk chunk `50`, timeout `120`, refresh, and count. The official Elasticsearch 8.11.4 Windows archive was checksum-verified, started locally with Lab 6 settings, and used to recreate and populate the five-document `keyframes` index. Runtime files remain ignored. | VERIFIED |
| Lab 6 | Same notebook, cells 17 and 28 | BM25 OCR/caption retrieval | `src/mvr/retrieval/lexical.py` | Preserved `ocr_text^2` emphasis, caption fallback, empty-query `match_all`, optional person/car `range gte` filters, OCR highlighting, and positive top-K validation. Request construction/execution boundary is verified; BM25 quality was not benchmarked. | VERIFIED |
| Lab 6 | Same notebook, cells 17 and 28 | Flexible object query parsing/search | `src/mvr/retrieval/object_search.py` | Preserved all 80 COCO labels, `class:non-negative-count`, comma parsing, whitespace, `rsplit`, unknown/duplicate rejection, and Elasticsearch dynamic `range gte` filters. Production errors are `ValueError`; HTTP translation remains Phase 05. | VERIFIED |
| Lab 6 | Same notebook, cells 18-19 | CLIP text vector and Elasticsearch semantic kNN | `src/mvr/retrieval/semantic.py` | Added an injected Elasticsearch retriever preserving normalized OpenCLIP text output, `clip_vector`, `size=k`, `k`, and notebook `num_candidates=100`. Real RN50 text encoding and Elasticsearch kNN returned all five smoke keyframes for `color test pattern`. The API variant's `200` remains deferred; no quality conclusion was made. | VERIFIED |
| Lab 6 | Same notebook, cells 19 and 28 | Reciprocal Rank Fusion | `src/mvr/retrieval/hybrid.py` | Verified the existing richer `(id, score)` result preserves 1-based reciprocal ranks, default `rrf_k=60`, symmetric branch treatment, deterministic ID ties, validation, and top-K truncation. | VERIFIED |
| Lab 6 | Same notebook, cell 25 | Weighted hybrid fusion | `src/mvr/retrieval/hybrid.py` | Preserved independent per-query min-max normalization, equal-score-to-`1.0`, union candidates, endpoint active-branch tie behavior, deterministic IDs, and configured alpha `0.25`; no alpha study or tuning was performed. | VERIFIED |
| Lab 6 | Same notebook, cells 6-19 and 25 | Real multimodal runtime flow | `tests/integration/test_lab6_runtime.py` | Added an opt-in five-keyframe integration test that runs real EasyOCR, PaddleOCR, YOLOv8n, RN50, `SearchDocument`, Elasticsearch indexing, BM25, object filtering, semantic kNN, RRF, and weighted fusion. It prints representative metadata-rich Top-K results and remains skipped in lightweight CI. | VERIFIED |
| Lab 6 | Same notebook, cells 19, 22, and 25 | Query-group evaluation and fusion studies | `src/mvr/evaluation/metrics.py` | Preserve OCR-text versus semantic grouping, R@1/R@5/MRR, latency, RRF-k study, and alpha study as evaluation code; tuning remains out of the reuse baseline. | MAPPED |
| Lab 6 | Same notebook, cell 28 | FastAPI retrieval prototype | `src/mvr/api/app.py` | Extract app lifecycle and real endpoints (`/health`, `/search/text`, `/search/object`, `/search/hybrid`); preserve lazy CLIP load, validation bounds, stable hit payload, and static keyframe serving. | MAPPED |
| Lab 6 | Same notebook, cell 29 | API smoke checks | `tests/integration/test_api.py` | Convert process/network smoke checks into deterministic tests with dependency injection where practical; preserve success, malformed-object `400`, static image, and OpenAPI assertions. | MAPPED |
| FastAPI Lab | `lab1-fastapi-student/app/main.py` | FastAPI app/route structure and query validation | `src/mvr/api/app.py` | Reuse app organization, typed query bounds, async-compatible route shape, and response models; replace mock search with real services only in Phase 05. | MAPPED |
| FastAPI Lab | `lab1-fastapi-student/app/models.py` | Pydantic response schemas | `src/mvr/api/schemas.py` | Adapt fields to MVR hit/search contracts while preserving explicit response models and generated OpenAPI documentation. | MAPPED |
| FastAPI Lab | `lab1-fastapi-student/tests/test_app.py` | API contract tests | `tests/unit/api/` | Preserve TestClient patterns for success, case-insensitive search, empty results, required query, limit bounds, limit behavior, and score ordering. | MAPPED |
| FastAPI Lab | `lab1-fastapi-student/.github/workflows/ci.yml` | GitHub Actions pytest workflow | `.github/workflows/ci.yml` | Target CI already has the same checkout/setup/install/test pattern. Reconcile action versions and dependencies deliberately; do not copy generated environments. | MAPPED |

## Phase 02 ingestion verification

- Added the Shot Boundary Lab's exact lightweight dependency ranges to `pyproject.toml`: PySceneDetect `>=0.6.4,<0.8`, OpenCV `>=4.8,<6`, and tqdm `>=4.66,<5`.
- Verified `py -3.12 -m mvr.cli ingest "<video>" --output "artifacts\keyframes\phase02-smoke"` against source sample `L01_V001.mp4`.
- Generated 5 inclusive-end shot records, 5 keyframe records, 5 JPEGs, `shots.json`, `keyframe_mapping.csv`, `benchmark.json`, `benchmark.csv`, and `benchmark_summary.csv` under the ignored smoke output directory.
- The optional TransNetV2 source is migrated but runtime verification is skipped because Torch, `transnetv2-pytorch`, and model weights are intentionally not installed for the lightweight Phase 02 run.
- No OpenCLIP, FAISS, OCR, YOLO, Elasticsearch, FastAPI, or other Phase 03+ code was started.

## Phase 03 semantic verification

- Reused Lab 3's exact default `ViT-B-32` / `laion2b_s34b_b79k` model, OpenCLIP preprocessing/tokenizer, RGB conversion, `eval()`, `torch.no_grad`, image batch size `32`, text batch size `64`, and L2 normalization.
- Reused Lab 4's contiguous normalized `float32` vectors, exact `faiss.IndexFlatIP`, inner-product scoring, descending search order, and metadata row lookup.
- Verified real CPU image and text encoding with the downloaded OpenCLIP weights; both produced 512-dimensional unit-normalized vectors.
- Built and reloaded an ignored persistent index from the five real JPEGs in `artifacts/keyframes/phase02-smoke`; outputs are `image_embeddings.npy`, `keyframes.faiss`, `metadata.json`, and `manifest.json` under `artifacts/indexes/phase03-smoke`.
- Search metadata stores resolved absolute JPEG paths because the standalone search command receives only the index location; tests verify the persisted paths exist after reload.
- Verified `a man riding a motorcycle` against those keyframes. Rank 1 was `L01_V001_S0002_K01` at `1.56s` with score `0.16651125252246857`.
- Added semantic dependencies as the optional `semantic` extra so ingestion remains independently installable.
- No OCR, YOLO, Elasticsearch, FastAPI, UI, temporal retrieval, reranking, or other Phase 04+ implementation was started.
- Phase 03 verification established engineering functionality only. Retrieval quality was not benchmarked or validated.

## Phase 04 initial lightweight verification

- Added lazy EasyOCR, PaddleOCR, and YOLOv8n adapters with source defaults `0.3`, `0.3`, and `0.4`; deterministic tests exercise their external-library boundary shapes without installing models or weights.
- Added the Lab 6 document schema and Phase 02 timestamp/keyframe mapping adapter.
- Added the exact ES 8.x `vi_folded` mapping, cosine dense-vector field, explicit index recreation, Lab 6 bulk request parameters, refresh, and count logic.
- Installed and imported the Elasticsearch 8.x Python client. `localhost:9200` returned unavailable, so no live index was deleted, created, populated, or searched.
- Added verified BM25, flexible COCO object, and semantic kNN request logic. These tests validate request mathematics and metadata contracts, not Elasticsearch ranking quality.
- Reconciled the existing RRF and weighted fusion functions with Lab 6, including deterministic and endpoint tie behavior. Defaults remain `rrf_k=60` and configured alpha `0.25`; no tuning was performed.
- Made Lab 6's `RN50`/`openai`/batch-`64` identity explicit and separate from Phase 03's `ViT-B-32`/`laion2b_s34b_b79k` baseline. RN50 model inference was not run.
- No FastAPI, UI, temporal retrieval, reranking, or Phase 05+ implementation was started.

## Phase 04 real runtime verification

- Installed the existing optional extras without changing constraints: EasyOCR 1.7.2, PaddleOCR 3.7.0, PaddlePaddle 3.3.1, Ultralytics 8.4.120, OpenCLIP 3.3.0, and Elasticsearch Python client 8.19.3.
- Ran EasyOCR and PaddleOCR on all five existing Phase 02 JPEGs at threshold `0.3`. EasyOCR extracted `00:00:00` (`0.982596`) and `480` (`0.999869`) from frame 1. PaddleOCR final text included `00:00:00.480` (`0.990175`), `12` (`0.999941`), and frame 3 text `I` (`0.479595`).
- Ran the real `yolov8n.pt` detector at confidence `0.4` on all five keyframes. Every object count was empty; the frames are synthetic color/test patterns, so the later real `person:1` object query correctly returned zero hits.
- Ran the exact Lab 6 `RN50` / `openai` encoder on CPU. Five image vectors and the text query vector were 1024-dimensional `float32` and unit-normalized within tolerance.
- Downloaded the official Elasticsearch 8.11.4 Windows distribution, matched its published SHA-512, used its bundled JDK with `-Xms1g -Xmx1g`, disabled security, and ran single-node mode exactly as Lab 6. The service and Python client both reported server 8.11.4 reachable.
- Recreated the `keyframes` index with the preserved mapping and indexed five real `SearchDocument` records. Runtime distributions, logs, server data, weights, and keyframes remain ignored.
- BM25 query `00` returned frame `L01_V001_S0001_K01` at `0.48s` with score `1.6309344` and indexed PaddleOCR text `00:00:00.480 12`.
- Semantic kNN query `color test pattern` returned five frames; rank 1 was `L01_V001_S0005_K01` at `4.88s` with score `0.61126626`.
- RRF (`rrf_k=60`) and weighted fusion (`alpha=0.25`) both returned five metadata-rich results. Their rank 1 was `L01_V001_S0001_K01`, with scores `0.03252247488101534` and `0.865659344326857` respectively.
- This smoke run verifies execution and integration only. It does not benchmark OCR, detection, semantic, lexical, object, or hybrid retrieval quality.
- No FastAPI, UI, temporal retrieval, reranking, tuning, new model, full-dataset indexing, or Phase 05 work was started.

## Behavior and compatibility decisions

1. **No model substitution:** Lab 3's primary OpenCLIP source is `ViT-B-32` with `laion2b_s34b_b79k`; Lab 4/Lab 6 use `RN50` with `openai`. Phase 03 selected the Lab 3 pair explicitly in code and `configs/default.yaml`; these model families are not treated as equivalent.
2. **Normalized inner product:** Image and text embeddings are L2-normalized before dot product or `faiss.IndexFlatIP`; this is the cosine-similarity contract.
3. **Frame contract:** Frames are zero-based, shot ends are inclusive, and timestamps are derived from `frame_idx / fps`.
4. **Optional heavy dependencies:** TransNetV2, OCR engines, YOLO, OpenCLIP, FAISS, Elasticsearch, and their weights/services remain optional by phase. Phase 03 places Torch, torchvision, Pillow, OpenCLIP, and FAISS in the `semantic` extra and keeps their imports lazy where practical.
5. **Generated data remains external:** Videos, keyframes, embeddings, Parquet/JSON outputs, indexes, Elasticsearch data, model weights, and raw benchmark artifacts remain ignored.
6. **Elasticsearch lifecycle separation:** Lab 6 downloads/starts a local unsecured Elasticsearch process and deletes/recreates the `keyframes` index. Later packaging must separate service startup from index operations while retaining index semantics.
7. **PaddleOCR is Lab 6's selected OCR output:** EasyOCR is retained for comparison; downstream Lab 6 documents use PaddleOCR text.
8. **Deferred experiments stay deferred:** Vietnamese translation, SigLIP/SigLIP2, GVE/Qwen, vector-concatenation fusion, CrossEncoder reranking, RRF/alpha tuning, VLM captioning, temporal retrieval, UI, submission, and deployment are not part of the first runnable reuse baseline.
9. **Dependency anomaly to verify later:** `lab1-fastapi-student/requirements.txt` contains `httpx2==2.7.0`; Phase 05 must verify the intended package rather than copying the pin blindly.
10. **Persistence compatibility:** Lab 4 constructed FAISS in memory and did not save it. Phase 03 adds `faiss.write_index`/`read_index` plus JSON metadata and a schema-versioned manifest. This is packaging/persistence only; index type, vectors, scores, and ranking mathematics are unchanged. Reserved manifest invariants and loaded artifacts are validated to prevent silent mismatches.
11. **Mapping compatibility:** Lab 3 used a Parquet image mapping. The production builder consumes Phase 02's existing CSV and stores the four retrieval fields in JSON, avoiding another dataframe dependency while retaining row order and `.npy` embeddings. It resolves the CSV's relative image paths to absolute paths because `search` has no ingestion-root argument.
12. **Normalization enforcement:** Lab 3 and Lab 4 produce unit-normalized vectors before exact inner-product search. The FAISS boundary validates finite unit vectors and queries rather than renormalizing them, preserving the source mathematics and making contract violations explicit.
13. **Model-backed integrations remain optional:** EasyOCR/PaddleOCR/PaddlePaddle live in the `ocr` extra, Ultralytics in `objects`, and the Elasticsearch 8 client in `search`. Production modules import these dependencies lazily so ingestion and FAISS usage do not require them.
14. **No notebook-managed Elasticsearch service:** Lab 6 downloaded Elasticsearch 8.11.4, disabled security, started a subprocess, and waited for readiness. Production code accepts a client and exposes explicit index recreation only; downloading/starting/deleting a service is an operator action, not an import side effect.
15. **Lab 6 time-field compatibility:** Elasticsearch documents retain the source field `ts`; the document adapter accepts Phase 02's `timestamp` and converts it to `ts`.
16. **Framework-neutral retrieval errors:** Object parsing raises `ValueError` instead of the Lab 6 API cell's `HTTPException`. Phase 05 may translate these errors at the API boundary without changing parsing semantics.
17. **Semantic kNN candidate source:** Phase 04 preserves notebook `num_candidates=100`. The API cell's value `200` belongs to deferred Phase 05 behavior and is not silently substituted.
18. **No quality claim:** Phase 04 verifies adapters, request construction, index schemas, and fusion mathematics. It does not establish OCR accuracy, detector accuracy, BM25/kNN/hybrid retrieval quality, or superiority over the Phase 03 baseline.
19. **Extracted-boundary validation:** Production adapters reject blank frame IDs, empty OCR ground truth, non-finite/non-unit semantic vectors, non-positive candidate pools, and kNN `k` larger than `num_candidates`. These inputs were absent from the notebook; the checks prevent silent ID collisions or invalid Elasticsearch requests without changing valid Lab 6 behavior.

## Status values

```text
NOT STARTED
MAPPED
MIGRATED
VERIFIED
BLOCKED
```

Any later behavior change must record the original behavior, new behavior, reason, whether it is compatibility-only or semantic, and verification status.
