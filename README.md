# Multimedia Video Retrieval Competition System

Repo khung để biến các lab hiện có thành một hệ thống thi Multimedia Video Retrieval có thể phát triển theo module.

This is a starter/base repository for an end-to-end Multimedia Video Retrieval competition system. It defines the shared architecture and includes a small tested retrieval foundation; the complete competition system is not implemented yet.

## Development status

Existing foundation:

- repository architecture, configuration, and documentation;
- hybrid retrieval utilities and unit tests;
- CI configuration;
- original reference notebooks.

Planned or in progress:

- video ingestion, shot detection, and keyframe extraction integration;
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

Đọc tiếp:

- `docs/SOURCE_FILE_ANALYSIS.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
- `docs/GIT_WORKFLOW.md`
