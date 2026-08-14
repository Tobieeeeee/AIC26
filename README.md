# Multimedia Video Retrieval Competition System

Repo khung để biến các lab hiện có thành một hệ thống thi Multimedia Video Retrieval có thể phát triển theo module.

This is a starter/base repository for an end-to-end Multimedia Video Retrieval competition system. It defines the shared architecture and includes a small tested retrieval foundation; the complete competition system is not implemented yet.


## Development Convention

This repository separates experimental work from production implementation.

- Production system code belongs under `src/mvr/` as importable `.py` modules.
- Experimental notebooks belong under `notebooks/experiments/`.
- Original lab/reference notebooks remain under `notebooks/reference/`.
- The production pipeline must not depend on manually executing Jupyter notebooks.
- Successful notebook experiments should be promoted into tested Python modules before integration.

All coding agents and contributors must read `AGENTS.md` before implementing features.


## Development status

Existing foundation:

- repository architecture, configuration, and documentation;
- hybrid retrieval utilities and unit tests;
- CI configuration;
- original reference notebooks.

Available now:

- verified video probe, PySceneDetect shot detection, keyframe selection, exact FFmpeg extraction, and metadata mapping;
- `py -3.12 -m mvr.cli ingest` for one video or a directory of videos.
- verified OpenCLIP `ViT-B-32` image/text embeddings and persistent exact FAISS `IndexFlatIP` search;
- `py -3.12 -m mvr.cli build-index` and `py -3.12 -m mvr.cli search` for semantic keyframe retrieval.
- lightweight-verified Lab 6 adapters for EasyOCR, PaddleOCR, YOLOv8n, Elasticsearch documents/index requests, BM25, object filtering, semantic kNN, RRF, and weighted fusion.

Planned:

- query understanding, temporal retrieval, and reranking;
- FastAPI serving, competition UI, submission adapter, and complete evaluation framework.

## Pipeline

```text
Video
  -> probe metadata
  -> shot boundary detection
  -> keyframe selection/extraction
  -> OCR + object detection + visual embedding
  -> document/metadata packaging
  -> lexical/vector/object indexes
  -> BM25 / semantic / object search
  -> hybrid fusion / reranking
  -> FastAPI
  -> competition UI / submission layer
```

## Nguyên tắc

- Notebook dùng cho thử nghiệm, không phải production pipeline.
- `src/` chứa code chạy thật; mọi model/backend có interface riêng để thay được.
- Raw video, keyframe, embedding, model weights và index không commit lên Git.
- Mỗi thay đổi đi qua feature branch + pull request + test.
- Benchmark phải dùng cùng query set/ground truth khi so sánh model hoặc fusion.

## Bắt đầu

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

## Verified ingestion command

The Phase 02 smoke run used Python 3.12, FFmpeg/ffprobe, and the default PySceneDetect settings:

```powershell
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m mvr.cli ingest "D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder\shot-boundary-lab-student\student\data\videos\L01_V001.mp4" --output "artifacts\keyframes\phase02-smoke"
```

The output contains `shots.json`, `keyframe_mapping.csv`, benchmark JSON/CSV, and JPEG keyframes. The input may also be a directory. TransNetV2 is preserved as an optional backend but was not runtime-verified in Phase 02.

## Verified semantic retrieval commands

The Phase 03 smoke run reused the five real keyframes produced above and ran OpenCLIP and FAISS on CPU:

```powershell
py -3.12 -m pip install -e ".[dev,semantic]"
py -3.12 -m mvr.cli build-index --input "artifacts\keyframes\phase02-smoke" --output "artifacts\indexes\phase03-smoke" --device cpu
py -3.12 -m mvr.cli search "a man riding a motorcycle" --index "artifacts\indexes\phase03-smoke" --top-k 5 --device cpu
```

The build writes ignored `image_embeddings.npy`, `keyframes.faiss`, `metadata.json`, and `manifest.json` files. Search prints JSON results containing `rank`, `frame_id`, `video_id`, `timestamp`, `score`, and `image_path`.

## Phase 04 optional integrations

The Phase 04 adapters preserve Lab 6 defaults, but their external runtimes have different verification levels:

- Elasticsearch 8.x client installation and request construction are verified; no service was available at `localhost:9200`, so live indexing/search was not run.
- EasyOCR and PaddleOCR/PaddlePaddle remain in the optional `ocr` dependency extra; their model runtimes and OCR quality were not tested.
- Ultralytics YOLOv8n remains in the optional `objects` dependency extra; model weights were not downloaded and detection quality was not tested.
- Lab 6's Elasticsearch vector path explicitly uses OpenCLIP `RN50` / `openai`, separate from Phase 03's `ViT-B-32` FAISS index. RN50 inference was not run in Phase 04.

The verified lightweight setup and test commands were:

```powershell
py -3.12 -m pip install -e ".[dev,semantic,search]"
py -3.12 -m pytest -q
```

These checks validate packaging, schemas, request construction, error handling, and fusion mathematics. Retrieval quality has not been benchmarked or validated.

Đọc tiếp:

- `docs/SOURCE_FILE_ANALYSIS.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
- `docs/GIT_WORKFLOW.md`
