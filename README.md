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

Planned:

- OCR, object detection, and image/text embedding pipelines;
- FAISS/Elasticsearch indexing and semantic, OCR, object, and hybrid retrieval integration;
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

Đọc tiếp:

- `docs/SOURCE_FILE_ANALYSIS.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
- `docs/GIT_WORKFLOW.md`
