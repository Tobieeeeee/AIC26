# Lab Reuse Map

This file records source provenance and verification for the first MVR baseline. Phase 01 mapped the labs, Phase 02 migrated ingestion, Phase 03 migrated the OpenCLIP/FAISS engineering baseline, Phase 04 migrated Lab 6 multimodal adapters and retrieval request logic, and Phase 05 migrated the FastAPI serving contracts.

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
| Lab 6 | Same notebook, cells 17 and 28 | Flexible object query parsing/search | `src/mvr/retrieval/object_search.py` | Preserved all 80 COCO labels, `class:non-negative-count`, comma parsing, whitespace, `rsplit`, unknown/duplicate rejection, and Elasticsearch dynamic `range gte` filters. Production errors remain framework-neutral `ValueError`; Phase 05 translates them to HTTP `400` at the route boundary. | VERIFIED |
| Lab 6 | Same notebook, cells 18-19 | CLIP text vector and Elasticsearch semantic kNN | `src/mvr/retrieval/semantic.py` | Added an injected Elasticsearch retriever preserving normalized OpenCLIP text output, `clip_vector`, `size=k`, `k`, and notebook `num_candidates=100`. Real RN50 text encoding and Elasticsearch kNN returned all five smoke keyframes for `color test pattern`. Phase 05 explicitly supplies the API cell's separate `200` value for hybrid serving; no quality conclusion was made. | VERIFIED |
| Lab 6 | Same notebook, cells 19 and 28 | Reciprocal Rank Fusion | `src/mvr/retrieval/hybrid.py` | Verified the existing richer `(id, score)` result preserves 1-based reciprocal ranks, default `rrf_k=60`, symmetric branch treatment, deterministic ID ties, validation, and top-K truncation. | VERIFIED |
| Lab 6 | Same notebook, cell 25 | Weighted hybrid fusion | `src/mvr/retrieval/hybrid.py` | Preserved independent per-query min-max normalization, equal-score-to-`1.0`, union candidates, endpoint active-branch tie behavior, deterministic IDs, and configured alpha `0.25`; no alpha study or tuning was performed. | VERIFIED |
| Lab 6 | Same notebook, cells 6-19 and 25 | Real multimodal runtime flow | `tests/integration/test_lab6_runtime.py` | Added an opt-in five-keyframe integration test that runs real EasyOCR, PaddleOCR, YOLOv8n, RN50, `SearchDocument`, Elasticsearch indexing, BM25, object filtering, semantic kNN, RRF, and weighted fusion. It prints representative metadata-rich Top-K results and remains skipped in lightweight CI. | VERIFIED |
| Lab 6 | Same notebook, cells 19, 22, and 25 | Query-group evaluation and fusion studies | `src/mvr/evaluation/metrics.py` | Preserve OCR-text versus semantic grouping, R@1/R@5/MRR, latency, RRF-k study, and alpha study as evaluation code; tuning remains out of the reuse baseline. | MAPPED |
| Lab 6 | Same notebook, cell 28 | FastAPI retrieval prototype | `src/mvr/api/app.py` | Reused health, text, object, RRF hybrid, lazy CLIP, `top_k` 1-100, `rrf_k` 1-1000/default `60`, top-30 branch candidates, API kNN candidates `200`, stable hit metadata, and static JPEG serving. Added the required Phase 03 FAISS semantic route. Routes use POST bodies instead of notebook GET query parameters; Elasticsearch routes are registered only after explicit configuration, reachability, and index checks. | VERIFIED |
| Lab 6 | Same notebook, cell 29 | API smoke checks | `tests/unit/api/test_app.py`, real Uvicorn smoke run | Reused success, malformed-object `400`, static image, and OpenAPI checks. External services are injected for twelve lightweight contract tests; the real five-keyframe Uvicorn run separately verified every registered endpoint and JPEG serving. | VERIFIED |
| FastAPI Lab | `lab1-fastapi-student/app/main.py` | FastAPI app/route structure and query validation | `src/mvr/api/app.py` | Reused application metadata, typed validation bounds, route functions, explicit response models, and testable app construction. Mock search was replaced by the existing FAISS and Lab 6 retrieval services; model/index loading is lazy rather than per request. | VERIFIED |
| FastAPI Lab | `lab1-fastapi-student/app/models.py` | Pydantic response schemas | `src/mvr/api/schemas.py` | Adapted the explicit models to health, POST request bodies, ranked keyframe hits, optional OCR/object/highlight/rank metadata, and generated OpenAPI validation. | VERIFIED |
| FastAPI Lab | `lab1-fastapi-student/tests/test_app.py` | API contract tests | `tests/unit/api/test_app.py` | Reused TestClient success, empty result, required/nonblank query, limit bounds, limit behavior, response schema, health, and order checks. Dependency injection keeps tests lightweight while exercising real routes, schemas, and object parsing. | VERIFIED |
| FastAPI Lab | `lab1-fastapi-student/.github/workflows/ci.yml` | GitHub Actions pytest workflow | `.github/workflows/ci.yml` | Preserved checkout/setup/editable-install/pytest structure. CI now installs only `dev,api`; model, Elasticsearch, OCR, and YOLO extras and opt-in runtime tests remain excluded. Existing target action versions were retained. | VERIFIED |

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

## Phase 05 API verification

- Added `GET /health` and `POST /search/semantic` for the Phase 03 FAISS baseline. Health validates and caches the persisted FAISS artifact plus required runtime packages; the manifest-selected OpenCLIP encoder still loads once on the first search. The verified real query returned the same rank-1 frame and score as the Phase 03 CLI.
- Added `POST /search/text`, `/search/object`, and `/search/hybrid` only when an explicitly configured Elasticsearch service is reachable and the configured index exists at application startup. Without that runtime, the routes are absent from OpenAPI rather than returning placeholder successes.
- Preserved Lab 6 BM25/object request behavior and RRF fusion. The hybrid API retains branch size `30`, `rrf_k=60`, and the notebook API's kNN `num_candidates=200`; no retrieval parameter was tuned.
- Added Pydantic request and response models with `top_k` 1-100, `rrf_k` 1-1000, nonblank semantic/hybrid queries, non-negative text object filters, ranked keyframe fields, and optional OCR/object/highlight/rank metadata.
- Converted the source GET query-parameter routes to the required POST JSON bodies. Framework-neutral object parser errors become HTTP `400`; unavailable configured search runtimes become HTTP `503`.
- Added optional static keyframe serving under `/keyframes` when `MVR_KEYFRAME_DIR` exists. Both FAISS and Elasticsearch results inside that root return retrievable URLs using the Phase 02 nested `video_id/frame_id.jpg` layout; a configured root rejects semantic paths outside it instead of exposing unrelated local paths.
- Verified Uvicorn startup against the five real ignored keyframes and indexes. Health reported five Elasticsearch documents, semantic loaded lazily, BM25 `00` returned the known OCR hit, `person:1` returned a real empty result, RRF hybrid returned three results, malformed objects returned `400`, and a JPEG returned `200 image/jpeg`.
- Added twelve lightweight TestClient tests and kept CI on `.[dev,api]`; default CI does not install or execute OpenCLIP, FAISS, Elasticsearch, OCR, YOLO, datasets, or model downloads.
- Verified that the FastAPI Lab's `httpx2` dependency is a real Starlette TestClient package, not a typo. The compatible range `httpx2>=2.7,<3` replaces the lab's exact pin while retaining Python 3.10+ packaging.
- No UI, temporal retrieval, reranking, query understanding, submission, deployment, model tuning, or Phase 06 work was started.

## Behavior and compatibility decisions

1. **No model substitution:** Lab 3's primary OpenCLIP source is `ViT-B-32` with `laion2b_s34b_b79k`; Lab 4/Lab 6 use `RN50` with `openai`. Phase 03 selected the Lab 3 pair explicitly in code and `configs/default.yaml`; these model families are not treated as equivalent.
2. **Normalized inner product:** Image and text embeddings are L2-normalized before dot product or `faiss.IndexFlatIP`; this is the cosine-similarity contract.
3. **Frame contract:** Frames are zero-based, shot ends are inclusive, and timestamps are derived from `frame_idx / fps`.
4. **Optional heavy dependencies:** TransNetV2, OCR engines, YOLO, OpenCLIP, FAISS, Elasticsearch, and their weights/services remain optional by phase. Phase 03 places Torch, torchvision, Pillow, OpenCLIP, and FAISS in the `semantic` extra and keeps their imports lazy where practical.
5. **Generated data remains external:** Videos, keyframes, embeddings, Parquet/JSON outputs, indexes, Elasticsearch data, model weights, and raw benchmark artifacts remain ignored.
6. **Elasticsearch lifecycle separation:** Lab 6 downloads/starts a local unsecured Elasticsearch process and deletes/recreates the `keyframes` index. Later packaging must separate service startup from index operations while retaining index semantics.
7. **PaddleOCR is Lab 6's selected OCR output:** EasyOCR is retained for comparison; downstream Lab 6 documents use PaddleOCR text.
8. **Deferred experiments stay deferred:** Vietnamese translation, SigLIP/SigLIP2, GVE/Qwen, vector-concatenation fusion, CrossEncoder reranking, RRF/alpha tuning, VLM captioning, temporal retrieval, UI, submission, and deployment are not part of the first runnable reuse baseline.
9. **FastAPI test client dependency verified:** `lab1-fastapi-student/requirements.txt` contains the intentional `httpx2==2.7.0`. FastAPI 0.141.1/Starlette 1.6.0 emitted a deprecation warning while falling back to legacy `httpx`; installing `httpx2` removed it. Production uses the compatible range `httpx2>=2.7,<3` rather than blindly copying the exact lab pin.
10. **Persistence compatibility:** Lab 4 constructed FAISS in memory and did not save it. Phase 03 adds `faiss.write_index`/`read_index` plus JSON metadata and a schema-versioned manifest. This is packaging/persistence only; index type, vectors, scores, and ranking mathematics are unchanged. Reserved manifest invariants and loaded artifacts are validated to prevent silent mismatches.
11. **Mapping compatibility:** Lab 3 used a Parquet image mapping. The production builder consumes Phase 02's existing CSV and stores the four retrieval fields in JSON, avoiding another dataframe dependency while retaining row order and `.npy` embeddings. It resolves the CSV's relative image paths to absolute paths because `search` has no ingestion-root argument.
12. **Normalization enforcement:** Lab 3 and Lab 4 produce unit-normalized vectors before exact inner-product search. The FAISS boundary validates finite unit vectors and queries rather than renormalizing them, preserving the source mathematics and making contract violations explicit.
13. **Model-backed integrations remain optional:** EasyOCR/PaddleOCR/PaddlePaddle live in the `ocr` extra, Ultralytics in `objects`, and the Elasticsearch 8 client in `search`. Production modules import these dependencies lazily so ingestion and FAISS usage do not require them.
14. **No notebook-managed Elasticsearch service:** Lab 6 downloaded Elasticsearch 8.11.4, disabled security, started a subprocess, and waited for readiness. Production code accepts a client and exposes explicit index recreation only; downloading/starting/deleting a service is an operator action, not an import side effect.
15. **Lab 6 time-field compatibility:** Elasticsearch documents retain the source field `ts`; the document adapter accepts Phase 02's `timestamp` and converts it to `ts`.
16. **Framework-neutral retrieval errors:** Object parsing raises `ValueError` instead of the Lab 6 API cell's `HTTPException`. Phase 05 translates these errors to HTTP `400` at the API boundary without changing parsing semantics.
17. **Semantic kNN candidate source:** Phase 04's reusable Elasticsearch retriever preserves notebook `num_candidates=100`. Phase 05's hybrid API wrapper explicitly uses the Lab 6 API cell's separate `num_candidates=200`; neither value was tuned or silently substituted.
18. **No quality claim:** Phase 04 verifies adapters, request construction, index schemas, and fusion mathematics. It does not establish OCR accuracy, detector accuracy, BM25/kNN/hybrid retrieval quality, or superiority over the Phase 03 baseline.
19. **Extracted-boundary validation:** Production adapters reject blank frame IDs, empty OCR ground truth, non-finite/non-unit semantic vectors, non-positive candidate pools, and kNN `k` larger than `num_candidates`. These inputs were absent from the notebook; the checks prevent silent ID collisions or invalid Elasticsearch requests without changing valid Lab 6 behavior.
20. **Conditional API exposure:** Lab 6 assumed localhost Elasticsearch and registered every route unconditionally. Phase 05 requires `MVR_ES_URL` explicitly and registers text/object/hybrid only when the backend and index are reachable. This compatibility change keeps lightweight imports/CI deterministic and prevents unavailable services from appearing healthy.
21. **POST request contracts:** Lab 6 used GET query parameters, while the approved Phase 05 contract requires POST endpoints. Query values and all preserved bounds/defaults moved into typed JSON request models; retrieval calls and ranking mathematics are unchanged.
22. **API failure reporting:** Health marks missing/corrupt semantic artifacts or missing semantic packages unavailable. Backend failures use controlled HTTP `503` responses, while diagnostic exceptions stay in server logs. This changes only failure presentation, not successful retrieval behavior.

## Textual KIS task pack (post-baseline) — reuse without lab migration

The Textual KIS competition-ready work (branch `feat/kis-competition-ready`,
Tasks 01–09; see `docs/CODEX_PROGRESS.md` and `docs/AIC_KIS_FINAL_REPORT.md`)
migrated **no new lab code**. It consumed already-migrated production modules
and added KIS-specific logic on top, so it introduces no new source-lab rows
above. Recorded here for completeness per rule 18 of `AGENTS.md`:

| Existing module consumed | Consumed by | Lab-derived behavior changed? |
|---|---|---|
| `mvr.ingestion.probe.probe_video` | frame registry, refinement, E2E | No — called unmodified. |
| `mvr.ingestion.extract.extract_keyframes` | exact-frame refinement window decode | No — called unmodified; the file was deliberately not edited. |
| `mvr.ingestion.models.Keyframe` / `VideoInfo` | registry and refinement contracts | No. |
| `mvr.indexing.faiss_index.FaissIndex`, `mvr.indexing.semantic_builder.build_semantic_index` | real AIC index build | No — an adapter emits the `keyframe_mapping.csv` shape the builder already expects instead of modifying the builder. |
| `mvr.retrieval.semantic.SemanticRetriever`, `mvr.retrieval.hybrid.rrf_fuse` | KIS orchestration | No — retrieval and fusion mathematics untouched. |
| `mvr.api.app.create_app`, `mvr.api.schemas` | `POST /search/kis` | No — new route follows the existing injectable-service/validation conventions; existing routes unchanged. |

Two behavior notes for KIS-owned code (not lab-derived):

23. **Refinement failure isolation:** `retrieval.frame_refinement.refine_top_candidates`
    catches a per-candidate `OSError`/`RuntimeError` and falls back to that candidate's
    unrefined coarse result instead of aborting the batch. This was added after an
    intermittent FFmpeg pipe failure was observed under sustained real load
    (heavy CPU OpenCLIP inference interleaved with repeated subprocess decodes).
    It affects only failure handling; successful refinement is unchanged, and no
    frame index is ever guessed.
24. **Frame identity source:** KIS predictions use the organizer `map_keyframes`
    `frame_idx` column as the original-video frame ID. The `n` ordinal and keyframe
    filenames are used only as lookup keys, never as the submitted frame ID.

---

The TRAKE competition-ready work (branch `feat/trake-competition-ready`; see
`docs/CODEX_PROGRESS.md` and `docs/AIC_TRAKE_IMPLEMENTATION_NOTES.md`) also
migrates no new lab code in Phase 01. It makes two unavoidable behavior
changes to existing KIS-owned production code, required before TRAKE code
depends on it (see `docs/AIC_TRAKE_IMPLEMENTATION_NOTES.md` for full root
cause and blast-radius detail):

25. **Resolver exception type:** `ingestion.aic_dataset._normalize_ref()`
    (and its second caller, `keyframe_path()`'s directory scan) raised a
    bare `ValueError` for a non-numeric keyframe reference instead of
    `MissingFrameMappingError`. Because `retrieval.kis.resolve_original_frame_id()`
    only ever caught `MissingFrameMappingError`, a candidate whose
    `keyframe_id` was this repo's own (non-numeric) pipeline format crashed
    the entire `search_kis()` call instead of being dropped, and this
    crash's likelihood scales with the number of `search_kis()` calls per
    query — exactly what TRAKE's k-events-per-query shape does. Fixed at
    the root: `_normalize_ref()` now raises `MissingFrameMappingError`.
    Purely an exception-type correction; no change to which frame
    references successfully resolve. Verified: full regression suite green,
    including the (corrected) pre-existing regression test for this exact
    crash path.
26. **`search_kis()`'s `top_k` ceiling raised 100 -> 500:** TRAKE's
    per-event video-shortlisting step (Phase 05) needs a wider candidate
    pool than plain KIS's Top-100 qualifying-round cap, mirroring the
    course lab's own 500-wide `candidate_videos()` shortlisting step. The
    public `POST /search/kis` HTTP contract (`api/schemas.py`) is
    unaffected — this raises only the ceiling of `search_kis()`'s own
    Python parameter validation; the default and all existing callers at
    `top_k<=100` are unchanged. Verified via a new test asserting
    `top_k=300` now succeeds, alongside all pre-existing `top_k` boundary
    tests passing unmodified.

TRAKE Phase 04 ports three chain-alignment algorithms from the course lab's
`trake_align.py` (`dp_align` -> `AtlasAlignment`, `dp_align_soft` ->
`LuciferAlignment`, `two_pointer_clips` -> `UceseAlignment`, plus
`first_touch`) into `src/mvr/retrieval/trake_align.py`. The lab file is
course material kept outside this repository, not one of the migration
source labs in the table above, so it has no source-row entry there. The
ports preserve the reference algorithms' mathematics, defaults (`alpha`,
`beta`, `lam=0.8`, `min_sim`, `T_win`), scoring formulas, and documented
behavioral differences, and are verified against the reference's own
`python trake_align.py` self-check values. Three deviations are recorded
here as required:

27. **Per-video instead of whole-corpus scope.** The lab's alignment
    functions each scan every video in one call (`video_spans()`,
    `candidate_videos(..., pre_topk)`). This port takes one video's score
    matrix per call, because video shortlisting now happens earlier via
    `search_kis()` in Phase 05 orchestration. Only the reference's
    per-video-span core logic is ported; the outer scanning helpers are not
    needed. Consequence: `two_pointer_clips`'s `topk` was a GLOBAL per-event
    hit threshold across all videos and is now scoped to one video, so the
    lab's own `two_pointer_clips` self-check numbers are not numerically
    reproducible here by construction. Its documented core property (a
    winning chain need not be chronologically ordered) is preserved and
    explicitly tested instead.
28. **Lucifer `lam` threaded through the state key.** The lab's
    `_soft_key()` hard-codes `0.8` when ranking intermediate DP states,
    while `dp_align_soft()`'s final score uses the caller's `lam` — the two
    disagree for any `lam != 0.8`. This port uses the configured `lam` in
    both places. Verified to leave behavior identical at the lab's own
    default `lam=0.8` (the reference self-check value `1.85 * 0.8` still
    matches exactly), so this is a consistency correction, not a tuning
    change.
29. **Lucifer accumulates the gap budget across skipped events.** The lab
    indexes the gap constraint by the current event (`gaps[j-1]`) even when
    the events in between were skipped, so a surviving time span is judged
    against only the last hop's budget although a skipped event still
    consumes real elapsed time. Measured example: with
    `gaps = ([0,10], [0,10])`, E1 invisible, E0 at t=0 and E2 at t=18
    (inside the combined 20s budget), the reference formula charges
    `(18-10)^2` and the DP then discards the perfectly findable E0,
    returning `(None, None, f2)` instead of `(f0, None, f2)`. This port sums
    the bounds of every gap traversed since the last MATCHED event. The
    lab's own self-check only ever passes `gaps=None`, so it never exercises
    this path. Verified: identical to the reference when no event is
    skipped, and the regression scenario above now returns
    `(f0, None, f2)`.

TRAKE Phase 06 (`docs/AIC_TRAKE_BENCHMARK_RESULTS.md`) measured that three of
the ported defaults above were actively wrong at TRAKE's per-video scope
(deviation 27) and produced a meaningless comparison until fixed. Recorded
here as required by rule 18:

30. **U-CESE `topk`: lab default `300` → measured default `10`.** The lab's
    `two_pointer_clips(topk=300)` is a global per-event hit threshold across
    an entire corpus scan; at the per-video scope introduced by deviation 27,
    `300` exceeds the number of resolvable frames in almost any single video,
    so every event collapsed onto the same frame and every chain scored
    `0.000`. A real sweep (`topk` in `{1, 3, 5, 10, 20}`) found mean final
    score peaking at `10` (`0.413`) and dropping off on both sides (`1`→
    `0.190`, `3`→`0.300`, `5`→`0.380`, `20`→`0.253`). `10` is the measured
    default, not a guess or the lab's own value.
31. **Lucifer `min_sim`: lab default `0.05` → measured default `0.20`.** The
    lab's `dp_align_soft(min_sim=0.05)` is below every real CLIP cosine
    similarity observed in this corpus, so the skip mechanism (Lucifer's
    entire distinguishing feature versus ATLAS) never fired — every Lucifer
    chain was numerically identical to the corresponding ATLAS chain. A real
    sweep found `0.20` correct (mean final score `0.639`, matching ATLAS) and
    a sharp cliff just above it: `0.22`→`0.530`, `0.25`→`0.473`. This is
    documented as a cliff, not a plateau — values are not interchangeable
    within this range the way ATLAS's `top_n` is.
32. **`top_n` (return N-best distinct chains per video) added to all three
    strategies; lab reference returns only one.** Each lab function
    (`dp_align`, `dp_align_soft`, `two_pointer_clips`) returns a single
    best-scoring span. Wired unmodified into a real API/benchmark context,
    this meant a video contributing only 1 of the 100-prediction budget's
    slots even when several distinct valid chains existed in it, discarding
    most of the budget by construction. Added `top_n` (default `5`) to all
    three strategies to return the `N` best distinct chains per video
    instead. In isolation this alone lifted ATLAS's mean final score from
    `0.417` to `0.660` (later refined to `0.639` alongside the gap-phrase
    fixes below); a sweep over `top_n` in `{5, 10, 20}` showed the score
    plateauing across the whole range, so `5` (matching the lab's per-video
    intuition most closely while still using the budget) was kept rather
    than raised further.

Separately, Phase 06 also measured that the gap-phrase table's own no-clue
default (`(0, 300)` seconds) outperforms a hand-picked narrower default
(`(0, 60)`): `0.639` vs `0.633` mean final score on the same 6-query set.
This is not a lab-reference deviation (`GAP_PHRASE_TABLE` and its defaults are
new TRAKE-owned code, not ported from the lab), so it is recorded here for
completeness rather than as a numbered deviation.

**Cross-track reuse (post-Phase-09, not lab-derived):** `search_trake()`'s
optional query translation reuses `mvr.enrichment.qna_translation`
(`QueryTranslator`, `translate_for_retrieval`, `HelsinkiViEnTranslator`)
unchanged from the Q&A track — no new translation logic was written for
TRAKE. Consistent with this file's reuse-first discipline even though the
source is a sibling production module, not a course lab. Real measured
effect on 6 natural-Vietnamese benchmark queries: mean final score
`0.000` (no translation) `-> 0.547` (translated), vs. `0.639` for the same
queries hand-written in English. See `docs/AIC_TRAKE_FINAL_REPORT.md`'s
"Query Translation" section.

---

## Task05A v1 — Task04 artifact consumption without new lab migration

Task05A v1 migrates no new lab implementation and does not change Task04 model
behavior. It consumes immutable Task04 artifacts and adds index-document
preparation under `src/mvr/indexing/`.

| Task04 artifact/component | Task05A target | Behavior preserved / compatibility handling | Verification |
|---|---|---|---|
| Task04B OCR per-keyframe JSON (`text`, `conf`, `support_count`) | `indexing.filtering`, `indexing.task05a` | Does not rerun OCR. Applies only the approved Task05A confidence, temporal-support/high-confidence, punctuation-only, normalization, and intra-keyframe dedup gates. Numeric-only text and Vietnamese diacritics are preserved. | VERIFIED |
| Task04C ASR artifact contract | `indexing.filtering`, `indexing.task05a` | No ASR artifact exists in the current real corpus, so no fields/confidence values are invented. Empty ASR output and `confidence_filter_active=false` are recorded. | VERIFIED |
| Task04A Faster R-CNN JSON and explicit person/car counts | `indexing.filtering.project_object_metadata` | Does not rerun inference, thresholding, or NMS. Existing person/car counts are authoritative; other retained Task04 labels are counted and projected deterministically. | VERIFIED |
| Task04A CLIP `.npy` vectors and mapping CSV | `indexing.validation.validate_clip_vector`, `indexing.task05a` | Does not rerun CLIP or suppress near-duplicates. Validates numeric dtype, dimension, finiteness, row existence, and full canonical identity before emitting a vector reference. | VERIFIED |
| Task04 branch identity | `indexing.task05a` | The real CLIP/Object artifacts use a 643-row TransNetV2 mapping while the selected canonical OCR/ingest mapping has 677 rows. Task05A reports incompatible/missing rows and leaves their branch flags false; it does not repair or positional-join them. | VERIFIED |

## Task05B v1 — Elasticsearch/BM25 lexical branch reuse

Task05B consumes Task05A processed documents and reuses only the appropriate
existing Elasticsearch/Lab infrastructure conventions. The legacy Lab 6
multimodal mapping/query behavior remains available and unchanged.

| Existing component | Task05B target | Preserved/adapted behavior | Verification |
|---|---|---|---|
| `src/mvr/indexing/elasticsearch_index.py` / Lab 6 Elasticsearch 8.x bulk lifecycle | `src/mvr/indexing/lexical_index.py` | Reused injected client, helper-based bulk actions, chunk size 50, request timeout 120, explicit lifecycle, and final refresh/count conventions. Task05B uses its locked lexical-only mapping and surfaces partial failures instead of using the legacy vector/object schema. | VERIFIED |
| `src/mvr/retrieval/lexical.py` / Lab 6 `multi_match` BM25 request pattern | `src/mvr/retrieval/lexical_bm25.py` | Reused Elasticsearch `multi_match` execution. Task05B intentionally removes the legacy OCR boost, caption fallback, match-all behavior, object filters, and highlighting to satisfy equal weighting over exactly `ocr_text` and `asr_text`. Default Elasticsearch BM25 remains unchanged. | VERIFIED |
| Task05A processed-document contract | `src/mvr/indexing/lexical_index.py` | Uses authoritative `indexability.lexical`, `ocr.joined`, `asr.joined`, and canonical identity fields directly. No Task04/raw OCR/ASR reads, filtering, normalization, or deduplication occur. | VERIFIED |

## OCR production checkpoint replacement (OCR-Fix-02)

This approved task intentionally changes the production checkpoint sources while
preserving the DeepSolo + PARSeq architecture, crop/rectify behavior, and
F-1/F/F+1 temporal consolidation.

| Previous production source | Replacement | Behavior/compatibility change | Verification | Status |
|---|---|---|---|---|
| Ambiguous Hugging Face object named `deepsolo_res50_finetune_synth.pth`, saved as a Total-Text/TextOCR filename | Official DeepSolo Total-Text ResNet-50 fine-tuned artifact `tt_res50_finetune_synth-tt-mlt-13-15-textocr.pth`, source revision `dbadae995035246bad3376c7a44c015c69e9b313`, SHA256 `6342945f34781ea90340d11b089c091327115d381fceb62c87b5cd2a82efeed3` | Checkpoint identity and canonical runtime filename are corrected; detector architecture and intended runtime config are unchanged. | Locked fetch, host/worker preflight, real GPU model-load/inference smoke, and five-keyframe OCR subset | VERIFIED |
| `Bao-Minh/PARSeq-Vietnamese` mutable-main mirror renamed from `parseq_vi.ckpt` to `best-parseq.ckpt` | `lynguyenminh/vietnamese-scenetext-detection-recognition` commit `76cc5f3cc6268457aac764653400fdff681f8271`, `weights.zip` member `weights/rec/best-parseq.ckpt`, SHA256 `8089b13c5ad115a96a608c6401eaab36b081393ea0a8323537b29a2dc80168f5` | Production source is intentionally replaced by the task-selected public Vietnamese-finetuned checkpoint. The existing PARSeq loader remains compatible and crop/rectify behavior is unchanged. | Bundle/member hashing, real GPU model-load/inference smoke, and five-keyframe OCR subset | VERIFIED |

The fetch path now fails closed, publishes only after hash verification, and
never treats an existing filename as evidence of identity. Generated OCR
manifests record both exact checkpoint identities, revisions, hashes, runtime
paths/configs, and pipeline version. No OCR quality comparison was performed.

## Status values

```text
NOT STARTED
MAPPED
MIGRATED
VERIFIED
BLOCKED
```

Any later behavior change must record the original behavior, new behavior, reason, whether it is compatibility-only or semantic, and verification status.

## INGEST-KF04 hybrid selector experiment

This task is an explicitly approved semantic architecture change, not a silent
alteration of the Shot Boundary Lab reuse baseline. Its implementation remains
available as the explicit `hybrid_temporal_content` mode. The later production
freeze below supersedes its default status without changing its algorithm.

| Original/reused behavior | New production behavior | Reason and compatibility | Verification | Status |
|---|---|---|---|---|
| Floor midpoint for every shot; for `duration >= 12s`, add frames every 8s from shot start. | Canonical `hybrid_temporal_content`: 1s candidates, pinned MetaCLIP-2 local novelty, grayscale/Canny change events, rare-event preservation, mandatory midpoint, then repair all temporal gaps to <=8s. | Explicit KF04 architecture decision. Canonical identity, inclusive shot bounds, frame/timestamp semantics, and mapping CSV remain unchanged. The original algorithm remains byte-for-byte unchanged behind explicit `periodic` mode with 12s/8s defaults. | Deterministic CPU-only invariant/cache/reuse tests and real L21_V001 regression. | VERIFIED |
| Downstream MetaCLIP encoded selected JPEGs independently. | Ingestion publishes selected candidate rows as a validated standard MetaCLIP provider artifact; the feature runner automatically reuses it only when model/revision/preprocessing/normalization/canonical identities and hashes match. | Avoids duplicate inference without coupling other embedding providers. Missing or mismatched artifacts fail validation; canonical hybrid selection never falls back to another model/selector. | Provider artifact audit and positive/negative reuse tests. | VERIFIED |

## CANONICAL INGESTION V1 production freeze

| Previous default | Frozen production default | Reason and compatibility | Verification | Status |
|---|---|---|---|---|
| `hybrid_temporal_content` with PySceneDetect in runnable entrypoints | TransNetV2 at `0.5` plus periodic midpoint + `duration >= 12s` / every `8s` | Approved Batch1 freeze based on the four-video/99-GT re-evaluation. Periodic mathematics and canonical identity/mapping semantics are unchanged. PySceneDetect, midpoint, hybrid, and provider experiments remain explicit non-default modes. | CLI, script, config, and library default-resolution tests; default pipeline isolation test; focused ingestion regressions. | VERIFIED |

## SELECTOR-HYBRID-MIGRATION current production freeze

This later approved migration supersedes the V1 selector default above without
changing TransNetV2, canonical keyframe identity, or the retained periodic
implementation.

| Previous production behavior | Current production behavior | Reason and compatibility | Verification | Status |
|---|---|---|---|---|
| TransNetV2 `0.5` plus periodic midpoint + 12s/8s | TransNetV2 `0.5` plus `hybrid_temporal_content`: 1s candidates, pinned MetaCLIP-2 local novelty, grayscale/Canny change events, mandatory midpoint, and temporal repair to `<=8s` | Approved four-video/seven-query benchmark favored hybrid over semantic dedup and periodic on retrieval Recall@1/MRR. Periodic and midpoint remain explicit reproduction modes; semantic dedup remains available for larger re-evaluation. | Selector/default/config unit suites plus local/cloud R2-runner call-contract regression; no full Batch1 rerun in this wiring fix. | VERIFIED |

### R2 ingestion-only operational wiring

No detector, selector, identity, or extraction behavior changes here. The
operations runner reuses canonical ingestion unchanged and publishes the
per-video final keyframes, selected embeddings, mappings, metadata, and
manifests under the run-scoped ingestion staging prefix. The regenerable
`selector_cache/` candidate tree is excluded from R2 to avoid publishing
intermediate JPEGs and vectors; it remains in local output unless optional
post-success eviction is requested. This is an operational packaging change
only: selector computation and local output are unchanged.
The runner publishes a success marker last and records resume completion only
after a successful upload. Direct per-video promotion to the shared corpus
mapping is deliberately not performed. Verified with mocked R2 exclusion,
tree-preservation, partial-upload, stage-skipping, and resume-state
regressions; one live L21_V001 canary preceded the cache-exclusion change, but
no full Batch1 run was executed.

### R2 ingestion consolidation boundary

No detector, selector, threshold, mapping identity, or extraction behavior is
changed. `mvr.operations.ingestion_promotion` consumes the run-scoped outputs
above and adds the missing single-writer boundary: raw/staging completeness
audit, success-marker verification, per-video schema/identity validation,
global mapping consolidation, and preparation of a server-side keyframe copy
manifest. Prepared bundles stay under
`data/staging/ingestion-consolidated/<promotion-id>/` and publish `_READY.json`
last. Activation remains separate and fail-closed when the candidate would
invalidate an already enriched canonical collection. Synthetic unit coverage
and the real 872-video Batch1 preparation verified the operational contract.

### R2 ingestion activation boundary

No detector, selector, model, threshold, or keyframe identity rule is changed.
`mvr.operations.ingestion_activation` completes the operational single-writer
boundary around the reused ingestion output: immutable candidate verification,
explicit stale-downstream acknowledgement, recoverable canonical backup,
resumable server-side copies, exact destination verification, manifest-last
activation, and exact stale-image cleanup. The real Batch1 activation replaced
the earlier L24 canonical identity only after a 3,579-object backup and verified
133,265 final keyframes. Existing L24 enrichment is retained physically but is
declared stale and requires rebuild. Synthetic activation regressions and the
live R2 activation verified this compatibility boundary.

## Milvus production semantic runtime fix

This approved runtime fix migrates no new lab implementation. It reuses the
Task05 Milvus client, schema, backend, manifest, provider-query encoders,
independent provider retriever, and rank-only semantic RRF implementation.

| Previous runtime behavior | New production behavior | Reason and compatibility | Verification | Status |
|---|---|---|---|---|
| Production configuration named `milvus`, but FastAPI always constructed `LazyFaissSemanticService`. | FastAPI constructs `MilvusSemanticService` by default; `faiss_reference` remains an explicit-only test/parity selection. Milvus unavailability reports semantic not-ready and returns an explicit search error, with no FAISS fallback. | Closes the audited config/runtime mismatch without changing provider models, dimensions, inner-product metric, or RRF mathematics. | Default-resolution, failure-path, API, KIS, TRAKE, Q&A, hybrid, and production-default regressions. | VERIFIED |
| The Task05 service exposed provider-field search but was not a complete query-to-fused-results API service. | The existing three provider query encoders route only to `beit3_vector` (768), `metaclip2_vector` (512), and `pe_vector` (1024), then reuse rank-only RRF with `k=60`. | Compatibility wiring only; required-provider failure policy remains fail-closed and raw scores are never summed across providers. | Deterministic unit routing/RRF assertions plus a five-entity live Milvus FLAT fixture compared with real FAISS exact search. | VERIFIED |
| Readiness checked field presence but did not expose dimensions, loaded state, or all-vector-index readiness to the API. | Readiness distinguishes configured, reachable, collection-ready, and loaded state and validates all canonical dimensions and vector indexes. | Makes runtime state observable; it does not relax schema or manifest checks. | Live Docker Milvus 2.4.13 status and FastAPI health smoke. | VERIFIED |

## Task05B schema 1.2 — Vietnamese exact/folded/fuzzy fallback

This approved lexical-search change preserves Task05A `_source` text and the
existing composite Elasticsearch identity. It changes only Task05B analysis
and query orchestration; dense retrieval, Milvus, semantic RRF, and embedding
queries remain unchanged.

| Previous behavior | Schema 1.2 behavior | Reason / compatibility | Verification | Status |
|---|---|---|---|---|
| `ocr_text` and `asr_text` used only `vi_folded`; every query was one non-phrase `multi_match`. | Base fields use `vi_exact` (`standard` + `lowercase`) and `.folded` subfields use `vi_folded` (`standard` + `lowercase` + `asciifolding`). Search runs exact phrase, then folded phrase only on zero hits, then bounded fuzzy only when both phrase stages return zero. | Preserves diacritics for precision while retaining accent-insensitive recall. Existing hit schema, `top_k`, and `_id = <video_id>/<keyframe_id>` are unchanged. Python normalization is NFC + whitespace collapse + trim + casefold and never removes accents. | Unit request/orchestration/identity regressions plus opt-in Elasticsearch 8.11.4 analyzer and four-query integration test. | VERIFIED |

## RF-DETR production runtime pin

This approved fix does not migrate a new lab detector. It completes the
repository's existing RF-DETR production adapter and preserves the existing
Task04/Task05 object projection contract.

| Previous behavior | Pinned production behavior | Compatibility handling | Verification | Status |
|---|---|---|---|---|
| `objects/rfdetr.py` declared RF-DETR but always raised a pin error; `objects/rf_detr.py` separately defaulted to Medium and allowed package-managed downloads. | One canonical adapter loads only `RFDETRLarge` / `rf-detr-large-2026.pth` with `rfdetr==1.9.3`, local size/SHA-256 verification, CUDA batch 1, COCO labels, confidence 0.4, and no fallback/download. | The old module is a compatibility import only. Faster R-CNN and YOLO remain explicit reference/legacy backends. | Offline unit boundaries plus a real 20-image/20-video CUDA smoke repeated twice. | VERIFIED |
| Task04 object JSON provided one envelope per keyframe; DATA-02 object Parquet requires one row per detection. | The same normalized detections publish Task05A envelopes and separate flat staging rows with `keyframe_id`, `video_id`, `frame_id == frame_idx`, label/class/confidence/box fields. | Task05A continues to project labels/counts without a second threshold or NMS. Object Parquet identity now preserves multiple boxes per keyframe. | Real staging → schema → Parquet → promote → Polars join, Task05A projection, and Elasticsearch filter representation. | VERIFIED |
