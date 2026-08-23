# AI Challenge 2026 - Okchokchok..

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
- a verified FastAPI serving layer with health and semantic search, plus text, object, and RRF hybrid routes when the configured Elasticsearch index is reachable.
- real-data-verified Textual KIS support: an AIC original-frame registry, the official KIS evaluator (`R@1/5/20/50/100` and per-query Final Score), Top-100 KIS orchestration behind `POST /search/kis`, exact-frame refinement on the original video, a reproducible KIS benchmark, and prediction validation/export. See `docs/AIC_KIS_FINAL_REPORT.md` for measured results and limitations.
- a CPU-runtime-verified per-keyframe ASR feature pipeline with independent +/-3 second audio windows, Silero VAD gating, the exact Vietnamese ChunkFormer provider, canonical JSON artifacts, resume/force behavior, and an isolated CPU Dockerfile. See `docs/ASR_KEYFRAME_PIPELINE.md` for commands and remaining GPU/quality gates.
- a per-keyframe object detection stage using RF-DETR (COCO-pretrained), producing one JSON per keyframe plus `shot_summary.json`/`manifest.json`; part of `scripts/run_feature_extraction.py`. See `docs/RF_DETR_OBJECT_EXTRACTION.md` for setup, commands, and the output contract.

Planned:

- query understanding, temporal retrieval, and reranking;
- competition UI, organizer submission transport, and complete evaluation framework.

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
pip install -e requirement.text
pytest -q python -m
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

## Verified Phase 04 runtime integration

The real CPU smoke test uses the five ignored Phase 02 keyframes and preserves Lab 6's EasyOCR/PaddleOCR threshold `0.3`, `yolov8n.pt` threshold `0.4`, OpenCLIP `RN50` / `openai` batch `64`, Elasticsearch 8.11.4 mapping, kNN candidates `100`, RRF `60`, and weighted alpha `0.25`. It recreates the local development index named `keyframes`.

Install the optional runtimes with the command that was verified locally:

```powershell
py -3.12 -m pip install -e ".[dev,semantic,search,ocr,objects]"
```

The smoke run used the official Elasticsearch 8.11.4 Windows archive after verifying its published SHA-512. Once extracted under the ignored runtime directory, it was started with the original Lab 6 settings:

```powershell
$esRuntimeRoot=(Resolve-Path -LiteralPath "artifacts\indexes\phase04-elasticsearch").Path
$esInstall=Join-Path $esRuntimeRoot "elasticsearch-8.11.4"
$esBat=Join-Path $esInstall "bin\elasticsearch.bat"
$env:ES_JAVA_OPTS="-Xms1g -Xmx1g"
Start-Process -FilePath "cmd.exe" -ArgumentList @("/c",$esBat,"-E","xpack.security.enabled=false","-E","discovery.type=single-node") -RedirectStandardOutput (Join-Path $esRuntimeRoot "es_boot.stdout.log") -RedirectStandardError (Join-Path $esRuntimeRoot "es_boot.stderr.log") -WindowStyle Hidden
```

Run the opt-in real-runtime integration test only after Elasticsearch is reachable:

```powershell
$env:MVR_RUN_PHASE04_RUNTIME_TESTS="1"
py -3.12 -m pytest -q -s tests\integration\test_lab6_runtime.py
py -3.12 -m pytest -q
```

The runtime test performs real EasyOCR, PaddleOCR, YOLOv8n, and RN50 inference; creates five `SearchDocument` records; indexes them in Elasticsearch; and runs BM25, object, semantic kNN, RRF, and weighted hybrid retrieval. The synthetic smoke frames contained no YOLO-detectable COCO objects, so the real `person:1` object request returned zero hits. This is runtime verification only; retrieval quality has not been benchmarked or validated.

## Verified API startup and requests

Install the lightweight API stack with the confirmed FastAPI/Starlette test client package. Add the existing `semantic` and `search` extras for the real local retrieval smoke run:

```powershell
py -3.12 -m pip install -e requierment.txt
```

Start Elasticsearch as shown in the Phase 04 section, then configure the existing ignored smoke artifacts and start Uvicorn:

```powershell
$env:MVR_SEMANTIC_INDEX=(Resolve-Path -LiteralPath "artifacts\indexes\phase03-smoke").Path
$env:MVR_DEVICE="cpu"
$env:MVR_ES_URL="http://localhost:9200"
$env:MVR_ES_INDEX="keyframes"
$env:MVR_KEYFRAME_DIR=(Resolve-Path -LiteralPath "artifacts\keyframes\phase02-smoke\keyframes").Path
py -3.12 -m uvicorn mvr.api.app:app --host 127.0.0.1 --port 8015
```

These requests were verified against the five-keyframe smoke indexes:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8015/health"

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8015/search/semantic" `
  -ContentType "application/json" `
  -Body '{"query":"a man riding a motorcycle","top_k":3}'

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8015/search/text" `
  -ContentType "application/json" `
  -Body '{"query":"00","top_k":3}'

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8015/search/object" `
  -ContentType "application/json" `
  -Body '{"want":"person:1","top_k":3}'

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8015/search/hybrid" `
  -ContentType "application/json" `
  -Body '{"query":"00","top_k":3,"rrf_k":60}'
```

`/search/semantic` is always registered, but returns `503` when no valid `MVR_SEMANTIC_INDEX` is configured. The text, object, and hybrid routes are registered only when `MVR_ES_URL` is explicitly configured, reachable at startup, and contains `MVR_ES_INDEX`; the health payload reports the actual runtime state instead of treating unavailable dependencies as healthy.

## Verified Textual KIS commands

KIS returns competition predictions as `<video_id>, <frame_id>` pairs on the
**original** video (at most 100 per query). Point `MVR_AIC_DATASET_ROOT` at a
local AIC dataset root containing `videos/`, `keyframes/<video_id>/`, and
`map_keyframes/<video_id>.csv`; never commit that data.

Build the searchable index directly from that dataset root first. This encodes
the organizer keyframes with the same OpenCLIP/FAISS pipeline used elsewhere;
by default it indexes every video that has both `keyframes/<video_id>/` and
`map_keyframes/<video_id>.csv` present locally, and `--videos` narrows that:

```powershell
py -3.12 -m pip install -e ".[dev,api,semantic]"
py -3.12 -m mvr.cli build-aic-index `
  --dataset-root "<local AIC dataset root>" `
  --output "artifacts\indexes\aic_l21" `
  --device cpu
```

Encoding is the slow step: roughly 0.055s per keyframe on CPU, so a 29-video /
7800-keyframe L21 subset takes about 7 minutes plus a one-time model download.

`POST /search/kis` is always registered and needs only the semantic index —
it resolves each retrieved keyframe to its original frame index and drops any
candidate it cannot verify rather than guessing:

```powershell
$env:MVR_SEMANTIC_INDEX=(Resolve-Path -LiteralPath "artifacts\indexes\aic_l21").Path
$env:MVR_AIC_DATASET_ROOT="<local AIC dataset root>"
$env:MVR_DEVICE="cpu"
py -3.12 -m uvicorn mvr.api.app:app --host 127.0.0.1 --port 8015

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8015/search/kis" `
  -ContentType "application/json" `
  -Body '{"query":"a geyser of water shooting high into the sky","top_k":5}'
```

The benchmark command scores a query set with the official KIS evaluator and
writes `records.json` plus `summary.json` under an ignored output directory:

```powershell
py -3.12 -m mvr.cli benchmark-kis `
  --index "artifacts\indexes\aic_l21" `
  --gt "artifacts\verification\aic_kis\gt_queries.json" `
  --dataset-root "<local AIC dataset root>" `
  --output "artifacts\verification\aic_kis" `
  --device cpu
```

Opt-in real-data verification (both skip unless explicitly enabled):

```powershell
$env:MVR_RUN_AIC_KIS_E2E="1"; $env:MVR_AIC_DATASET_ROOT="<local AIC dataset root>"
py -3.12 -m pytest tests\integration\test_aic_kis_e2e.py -v

$env:MVR_RUN_FRAME_REFINEMENT_REAL_TESTS="1"
py -3.12 -m pytest tests\integration\test_frame_refinement_real.py -v
```

Exact-frame refinement is implemented and verified against real decoded video,
but is **not enabled by default**: the measured benchmark showed identical
accuracy at roughly 3500x the latency. `docs/AIC_KIS_FINAL_REPORT.md` records
that evidence and the remaining limitations.

## Verified TRAKE commands

TRAKE (`TRAKE = KIS × k events, same video, correct order`) finds an ordered
chain of `k` frames in a **single** video that each match one described event,
in the declared order. It returns competition predictions as
`<video_id>, [frame_id, frame_id, ...]` (one chain of `k` frame IDs per
prediction, up to 100 predictions), scored by the official partial-credit
R-Score/Mean-Top-k-R/Final Score formula
(`tolerance=1` frame, matching the course lab default).

TRAKE reuses the **same FAISS index built by `build-aic-index`** above — there
is no separate index-build step. It is currently **API/library-only: there is
no `mvr.cli` subcommand for it.** Use `POST /search/trake` (shown below) or
call `mvr.retrieval.trake.search_trake()` directly from Python.

Each event's shortlist of candidate videos comes from the existing
`search_kis()` (reused unchanged); per-video scoring reconstructs every
resolvable frame's vector directly from the FAISS index
(`IndexFlatIP.reconstruct()`, no re-encoding) and scores it against each
event's text vector. One of three chain-alignment strategies then picks the
best-ordered chain per video. `atlas` (hard dynamic-programming alignment) is
the default — it matched `lucifer`'s accuracy on every real-data benchmark
query while running faster and without a tuning cliff; see
`docs/AIC_TRAKE_BENCHMARK_RESULTS.md` for the full parameter sweeps and
per-query numbers behind that choice.

```powershell
$env:MVR_SEMANTIC_INDEX=(Resolve-Path -LiteralPath "artifacts\indexes\aic_l21").Path
$env:MVR_AIC_DATASET_ROOT="<local AIC dataset root>"
$env:MVR_DEVICE="cpu"
py -3.12 -m uvicorn mvr.api.app:app --host 127.0.0.1 --port 8015

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8015/search/trake" `
  -ContentType "application/json" `
  -Body '{
    "events": [
      {"text": "a car crashes into a pole at night"},
      {"text": "a crowd gathers around the wreck"},
      {"text": "police arrive and set up a cordon"}
    ],
    "gap_phrases": ["rồi", "ngay sau đó"]
  }'
```

`gap_phrases` takes one Vietnamese connector phrase per gap between
consecutive events (`len(events) - 1` entries; `""` falls back to the phrase
table's no-clue default). Raw `[min, max]` second ranges are deliberately not
accepted directly — the real-data benchmark measured the phrase table's own
defaults outperforming hand-picked numbers (mean final score 0.639 vs 0.633).
`alignment_strategy` (`"atlas" | "lucifer" | "ucese"`) is an optional override
of the evidence-backed default, for manual testing only.

Event text can be Vietnamese if query translation is explicitly enabled —
off by default, since the benchmark above used untranslated English text:

```powershell
$env:MVR_TRAKE_TRANSLATE_QUERIES="1"
# first run only, downloads ~300MB to the HuggingFace cache:
$env:MVR_TRAKE_ALLOW_MODEL_DOWNLOAD="1"
```

Real measured effect (6 benchmark queries rewritten as natural Vietnamese,
same real pipeline): mean final score **0.000 without translation ->
0.547 with it** (English baseline on the same queries: 0.639). See
`docs/AIC_TRAKE_FINAL_REPORT.md`'s "Query Translation" section for the full
before/after table. A missing/uncached model degrades to untranslated
retrieval rather than an error.

Opt-in real-data verification (skips unless explicitly enabled):

```powershell
$env:MVR_RUN_AIC_TRAKE_E2E="1"; $env:MVR_AIC_DATASET_ROOT="<local AIC dataset root>"
py -3.12 -m pytest tests\integration\test_aic_trake_e2e.py -v
```

`docs/AIC_TRAKE_FINAL_REPORT.md` and `docs/AIC_TRAKE_BENCHMARK_RESULTS.md`
record the full benchmark evidence, the real end-to-end run, and known
limitations — most notably that benchmark queries were written in English
(the index's OpenCLIP model is English-trained); an opt-in translation layer
(above) closes most, not all, of the gap for Vietnamese queries.

Đọc tiếp:

- `docs/SOURCE_FILE_ANALYSIS.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
- `docs/GIT_WORKFLOW.md`
