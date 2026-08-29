  # Offline Baseline Model Audit

## Scope and method

This is a source-only audit of the offline paths in the current repository. No
model was downloaded and no extraction or inference was run. `Actual baseline`
means a model or tool selected by a public production entrypoint with no
model-selection overrides; it does **not** mean that an adapter merely exists.

There is no single default command that runs ingestion, OCR, ASR, visual
features, and object detection end to end. The repository exposes independent
entrypoints for those stages. The table therefore identifies the no-override
baseline for each individual public stage. The `configs/default.yaml` values
are declarations only: no source under `src/mvr/` or `scripts/` loads that YAML
file, so CLI/script defaults decide the executable behavior.

| Stage | Component | Actual baseline | Exact model/checkpoint | Default? | Evidence |
|---|---|---|---|---|---|
| Ingestion | Video discovery and probing | Python recursive discovery + FFprobe | No ML model | Yes | `src/mvr/ingestion/pipeline.py:17-49`; `src/mvr/ingestion/probe.py:9-22` |
| Ingestion | Shot detection | PySceneDetect `ContentDetector` | Algorithmic content-change detector; threshold `27.0` | Yes | `src/mvr/cli.py:49-73`; `src/mvr/ingestion/shot_detection/pyscenedetect.py:1-16` |
| Ingestion | Keyframe extraction | Midpoint/long-shot sampling + FFmpeg exact-frame decode/encode | No ML model | Yes | `src/mvr/ingestion/keyframes.py:6-75`; `src/mvr/ingestion/extract.py:95-265` |
| OCR | Detection and recognition | EasyOCR `Reader(["vi", "en"])` | **UNRESOLVED:** EasyOCR internal detector/recognizer checkpoints are not named or pinned by this path | Yes | `src/mvr/cli.py:224-240,407-440`; `src/mvr/enrichment/ocr/scene_text.py:291-414` |
| OCR | Temporal consolidation | Normalized-Levenshtein + box-IoU grouping over `F-1/F/F+1` | No ML model | Yes | `src/mvr/enrichment/ocr/scene_text.py:37-141,216-288,415-499` |
| ASR | VAD | Silero VAD | `silero_vad`; package/model revision is not pinned in source | Yes | `src/mvr/cli.py:137-221`; `src/mvr/enrichment/asr/vad.py:38-90` |
| ASR | Speech recognition | ChunkFormer | `khanhld/chunkformer-ctc-large-vie` at revision `14a68262fa5fc6d497cb107c18bc5a9308ef21ca` | Yes | `src/mvr/enrichment/asr/chunkformer.py:11,40-100`; `configs/default.yaml:21-38` |
| Visual | Image/text embedding | OpenCLIP | `ViT-B-32` / `laion2b_s34b_b79k`; 512-dimensional, L2-normalized vectors | Yes | `src/mvr/enrichment/embeddings/openclip.py:10-99`; `src/mvr/cli.py:76-89,388-396`; `src/mvr/indexing/semantic_builder.py:12-54` |
| Object | Detection | Faster R-CNN | `fasterrcnn_resnet50_fpn` / `torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT`; exact underlying weight artifact is **UNRESOLVED** in the repository | Yes, for the Stage 2 feature runner | `scripts/run_feature_extraction.py:70-84,188-194`; `src/mvr/enrichment/objects/faster_rcnn.py:8-49` |

## Baseline verdict

**PASS_WITH_UNRESOLVED_MODEL_PROVENANCE**

The executable model/provider choice is identifiable for every requested
stage. Two checkpoint-level provenance gaps prevent an unqualified PASS:

1. the actual EasyOCR path instantiates `easyocr.Reader` without declaring the
   detector or recognizer checkpoint IDs; and
2. the actual Faster R-CNN path selects torchvision's version-dependent
   `...Weights.DEFAULT`, not an immutable checkpoint identifier.

The baseline is also split across independent commands, rather than one
end-to-end default runner. This is a configuration/orchestration discrepancy,
not a reason to substitute a model.

## Execution paths and evidence

### 1. Ingestion

**Entrypoint:** `py -3.12 -m mvr.cli ingest <input> --output <output>`.
The parser defaults to `methods=["pyscenedetect"]`,
`primary="pyscenedetect"`, threshold `27.0`, long-shot duration `12.0`, and
sampling interval `8.0` (`src/mvr/cli.py:49-73`). The defaults are covered by
`tests/unit/test_cli.py:5-30`.

**Actual work:** `run_pipeline` discovers supported video extensions,
probes each input through FFprobe, constructs the selected shot detector,
selects midpoint frames (and periodic frames in long shots), extracts those
frames through two FFmpeg processes, and writes metadata/benchmark artifacts
(`src/mvr/ingestion/pipeline.py:17-99`).

* Video probing is FFprobe, not ML: `src/mvr/ingestion/probe.py:9-22`.
* Shot detection is PySceneDetect's `ContentDetector(threshold=27.0)`, not
  TransNetV2: `src/mvr/ingestion/shot_detection/pyscenedetect.py:1-16`.
* Keyframe selection is a deterministic midpoint plus periodic sampling
  algorithm, not ML: `src/mvr/ingestion/keyframes.py:6-75`.
* Exact JPEG extraction is FFmpeg raw `rgb24` decoding followed by an FFmpeg
  JPEG encode at `q:v=2`: `src/mvr/ingestion/extract.py:95-265`.

**Inputs:** one supported video or a directory (`.mp4`, `.mov`, `.mkv`,
`.avi`, `.webm`, `.m4v`).

**Outputs:** `shots.json`, `keyframe_mapping.csv`, `video_metadata.csv`,
`ingest_manifest.json`, `benchmark.json`/CSV, and `keyframes/<video_id>/*.jpg`
(`src/mvr/ingestion/metadata.py:8-76`). The mapping is zero-based and shot
end frames are inclusive.

**Configuration:** the equivalent fields are present at
`configs/default.yaml:1-6`, but the CLI does not read that YAML. The actual
defaults above are the CLI defaults.

**Available alternative:** `transnetv2` is a selectable, optional backend,
implemented in `src/mvr/ingestion/shot_detection/transnetv2.py:9-71`. It loads
the packaged `transnetv2-pytorch-weights.pth` at threshold `0.5`, but it is not
selected by the default CLI. `scripts/run_ingestion.py:6-30` is a second
entrypoint with different `15.0`/`5.0` long-shot defaults; it is not the
README's verified `mvr.cli ingest` baseline command.

### 2. OCR

**Entrypoint:** `py -3.12 -m mvr.cli enrich-ocr --mapping-csv <mapping>
--video-root <videos> --output <output>`. The public command calls
`process_dataset_ocr` directly (`src/mvr/cli.py:407-440`).

**Actual baseline:** each video is processed independently. The code extracts
`F-1`, `F`, and `F+1` sample frames, then creates
`easyocr.Reader(["vi", "en"], gpu=<CUDA availability>)` and calls
`reader.readtext` on each sample (`src/mvr/enrichment/ocr/scene_text.py:291-414`).
EasyOCR owns both text detection and recognition here; this path has no
separate crop/rectification model, and it does not instantiate PaddleOCR,
DeepSolo, or PARSeq.

* The actual default confidence threshold is `0.50`.
* Temporal deduplication groups candidates with normalized Levenshtein
  similarity `>=0.80` and bounding-box IoU `>=0.30`; it selects the
  highest-confidence representative and records `support_count`
  (`src/mvr/enrichment/ocr/scene_text.py:37-141,415-499`).
* The output manifest records `detector="easyocr"`,
  `recognizer="easyocr"`, offsets `[-1, 0, 1]`, and these thresholds
  (`src/mvr/enrichment/ocr/scene_text.py:472-499`).

**Inputs:** canonical `keyframe_mapping.csv`, raw source videos, and optional
co-located `shots.json` to bound temporal samples. **Outputs:** one JSON per
keyframe at `<output>/<video_id>/<keyframe_id>.json` plus `manifest.json`.

**Configured-but-not-executed discrepancy:** `configs/default.yaml:9-10`
states `ocr_engine: paddleocr` and a `0.3` threshold. No audited code loads the
file, and the public OCR command instead runs EasyOCR at `0.50`. Treat the
YAML value as a stale/declarative setting, not proof of the baseline.

**Available alternatives:**

* `EasyOcrEngine` and `PaddleOcrEngine` are reusable Lab 6 adapters
  (`src/mvr/enrichment/ocr/easyocr_engine.py`,
  `src/mvr/enrichment/ocr/paddleocr_engine.py`). PaddleOCR is used in the
  opt-in Lab 6 smoke test, not by `enrich-ocr`
  (`tests/integration/test_lab6_runtime.py:40-100`). Its source does not pin
  an individual text detector/recognizer model ID.
* `scripts/ocr/run_ocr_docker.ps1` invokes the separate
  `docker_ocr.py` path. That path uses DeepSolo detection with
  `deepsolo_r50_totaltext_textocr.pth` and PARSeq-Vietnamese recognition with
  `best-parseq.ckpt` (`src/mvr/enrichment/ocr/docker_ocr.py:213-254,395-491`).
  The supplied lock file maps those assets to the DeepSolo and
  Bao-Minh/PARSeq-Vietnamese Hugging Face sources
  (`docker/ocr/model_sources.lock.json`). It is an alternate Docker entrypoint,
  not the public CLI baseline; `docs/AIC_QNA_TEAM_TASKS.md:90-113` explicitly
  directs the team to use `enrich-ocr`, not `docker_ocr`.

### 3. ASR

**Entrypoint:** `py -3.12 -m mvr.cli build-asr-features --dataset-root <root>
--output <output>`. It is a first-class CLI command, separate from ingestion
and OCR (`src/mvr/cli.py:137-221,371-402`).

**Actual baseline:** one independent audio window per authoritative keyframe:

* FFprobe checks whether the video contains audio and FFmpeg decodes mono
  `float32` audio at 16 kHz (`src/mvr/enrichment/asr/audio.py:33-87`).
* The window is clamped to `[timestamp - 3 s, timestamp + 3 s]`
  (`src/mvr/enrichment/asr/audio.py:12-30`; CLI defaults at
  `src/mvr/cli.py:144-150`).
* Silero VAD (`silero_vad`) gates speech at threshold `0.5`, minimum speech
  `250 ms`, minimum silence `150 ms`, and adjacent regions are merged with a
  `250 ms` gap (`src/mvr/enrichment/asr/vad.py:38-90`; CLI lines `197-201`).
* Only VAD-selected speech regions call ChunkFormer. The provider loads
  `khanhld/chunkformer-ctc-large-vie` at pinned revision
  `14a68262fa5fc6d497cb107c18bc5a9308ef21ca`, writes temporary PCM WAV, then
  calls `endless_decode` with `chunk_size=64`, left context `128`, right
  context `128`, and `total_batch_duration=14400`
  (`src/mvr/enrichment/asr/chunkformer.py:11,40-100`).

The default device is CPU and model resolution is local-files-only; downloads
require `--allow-download` (`src/mvr/cli.py:150-163,202-206`). The config file
records the same intended provider/model/revision/device at
`configs/default.yaml:21-38`, but it is not the runtime configuration source.

**Inputs:** either an ingestion root (`keyframe_mapping.csv` plus
`video_metadata.csv`) or an organizer AIC root (`videos/` plus
`map_keyframes/`); this mapping choice is explicit in
`src/mvr/enrichment/asr/mapping.py:111-126`. **Outputs:**
`<output>/<video_id>/<keyframe_id>.json`, `manifest.json`, and `errors.json`
(`src/mvr/enrichment/asr/artifacts.py:15-99`).

**Alternatives:** no Whisper provider or Whisper execution path was found in
the audited production source. There is no fallback ASR model: failure to
resolve ChunkFormer fails clearly (`src/mvr/enrichment/asr/chunkformer.py:61-70`).

### 4. Visual features / CLIP

**Semantic baseline entrypoint:** `py -3.12 -m mvr.cli build-index --input
<ingestion-root> --output <index-root>`. With no model flags it constructs
`OpenClipEncoder(device=...)` and sends the CLI's default batch size `32` to
the semantic builder (`src/mvr/cli.py:76-89,388-396`).

**Actual baseline model:** OpenCLIP `ViT-B-32` with pretrained tag
`laion2b_s34b_b79k` (`src/mvr/enrichment/embeddings/openclip.py:10-53`). The
encoder converts images to RGB, uses OpenCLIP's paired preprocessing and
tokenizer, sets `eval()`, runs under `no_grad`, returns contiguous `float32`,
and L2-normalizes both image and text vectors
(`src/mvr/enrichment/embeddings/openclip.py:55-99`). The Phase 03 repository
verification records 512-dimensional unit vectors for this identity; the model
dimension is runtime-derived rather than hard-coded in the adapter.

**Inputs:** ingestion `keyframe_mapping.csv` and referenced JPEGs.
**Outputs:** the semantic builder writes `image_embeddings.npy` and a
persistent exact FAISS `IndexFlatIP` (`keyframes.faiss`, `metadata.json`,
`manifest.json`), recording the model/pretrained identity
(`src/mvr/indexing/semantic_builder.py:12-54`;
`src/mvr/indexing/faiss_index.py:60-150`).

**Legacy Stage 2 feature runner:**
`scripts/run_feature_extraction.py` independently defaults to provider
`clip`, the same `ViT-B-32`/`laion2b_s34b_b79k` pair, and a declared batch size
of `32` (`scripts/run_feature_extraction.py:30-84`). It writes
`<output>/clip/<video_id>.npy`, `clip_feature_mapping.csv`, and a manifest
(`src/mvr/enrichment/feature_extraction.py:27-115`). Its current legacy CLIP
loop calls `encode_images` with a one-image list, so the declared
`--embedding-batch-size` is not passed through to that loop; the effective
per-call image batch is one. This differs from the semantic builder's actual
batch-32 call and is reported, not changed.

**Baseline versus new/optional models:**

| Provider / path | AVAILABLE | CONFIGURED | ACTUAL_BASELINE | Classification and evidence |
|---|---:|---:|---:|---|
| OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` | Yes | Yes | Yes | Baseline for `build-index` and default `run_feature_extraction.py`; `src/mvr/enrichment/embeddings/openclip.py:10-53`, `scripts/run_feature_extraction.py:36-42`. |
| OpenCLIP `RN50` / `openai` | Yes | Yes, as `lab6_embedding_*` | No | Lab 6/API integration path; factory constants at `src/mvr/retrieval/semantic.py:11-43`; opt-in runtime test asserts 1024 dimensions at `tests/integration/test_lab6_runtime.py:113-124`. |
| BEiT-3 | Yes | Yes, optional provider | No | Requires explicit local checkpoint; `scripts/run_feature_extraction.py:109-130`; identity locked in `configs/embedding_providers.json`. |
| MetaCLIP-2 | Yes | Yes, optional provider | No | Requires explicit selection; local/cache-only unless `--allow-model-downloads`; `scripts/run_feature_extraction.py:132-148`. |
| Perception Encoder | Yes | Yes, optional provider | No | Requires explicit local checkpoint; `scripts/run_feature_extraction.py:150-176`. |

`run_all_features.ps1` defaults to the three new providers, but it always
passes `--skip-objects` (`scripts/run_all_features.ps1:7-34`). The current
full-corpus plan remains approval-gated and contains unresolved command
placeholders (`configs/full_corpus_build.json:19-30`). Therefore that wrapper
does not supersede the executable CLIP baseline.

### 5. Object detection

**Entrypoint:** the no-override object path is the default tail of
`python scripts/run_feature_extraction.py ...`. After its selected embeddings,
it instantiates `FasterRCNN` unless `--skip-objects` is passed
(`scripts/run_feature_extraction.py:178-194`).

**Actual baseline:** `fasterrcnn_resnet50_fpn`, with
`FasterRCNN_ResNet50_FPN_Weights.DEFAULT`, score threshold `0.5`, built-in
class-aware NMS enabled at `0.5`, RGB input, and labels from the selected
weights' metadata (`src/mvr/enrichment/objects/faster_rcnn.py:8-49`; manifest
fields at `src/mvr/enrichment/feature_extraction.py:117-195`). The repository
does not lock the `DEFAULT` enum to an immutable torchvision weight artifact,
so the precise downloadable checkpoint/taxonomy revision is intentionally
reported as **UNRESOLVED**, not inferred from external knowledge.

**Inputs:** Stage 1 mapping and its image paths. **Outputs:** one JSON per
keyframe under `<output>/objects/<video_id>/`, `shot_summary.json`, and
`objects/manifest.json` (`src/mvr/enrichment/feature_extraction.py:117-195`).

**Configured-but-not-executed discrepancy:** `configs/default.yaml:11-12`
declares `object_detector: yolov8n` and threshold `0.4`, but neither public
Stage 2 script nor `mvr.cli` reads that file. The actual no-override Stage 2
runner uses Faster R-CNN.

**Available alternative:** `YoloObjectDetector` loads `yolov8n.pt`, runs
Ultralytics prediction at confidence `0.4`, and returns per-frame counts
(`src/mvr/enrichment/objects/yolo.py:9-35`). It is used by the opt-in Lab 6
runtime test (`tests/integration/test_lab6_runtime.py:40-113`), not by the
default Stage 2 feature runner. It is therefore available/config-declared and
Lab 6 optional, but not the actual Stage 2 baseline.

## Configuration and execution discrepancies

1. **No consumed common default configuration.** `configs/default.yaml`
   describes PySceneDetect, PaddleOCR, YOLOv8n, OpenCLIP, and ASR defaults, but
   the audited entrypoints use argparse/function literals. This makes YAML
   presence insufficient evidence of runtime selection.
2. **OCR conflict.** YAML selects PaddleOCR `0.3`; the public
   `enrich-ocr` command uses EasyOCR `0.50` with temporal consolidation.
3. **Object conflict.** YAML selects YOLOv8n `0.4`; the default Stage 2
   script uses Faster R-CNN ResNet-50 FPN `DEFAULT`, score/NMS `0.5`.
4. **Feature-runner split.** The normal semantic baseline is OpenCLIP
   `ViT-B-32`/LAION. The multi-provider PowerShell wrapper defaults to
   BEiT-3/MetaCLIP-2/Perception Encoder, skips objects, and cannot execute the
   full-corpus plan until its placeholders and assets are resolved. It is a new
   optional workflow, not a baseline replacement.
5. **Ingestion helper split.** `scripts/run_ingestion.py` uses 15-second
   long-shot and 5-second sampling defaults, while the tested/public CLI and
   config declaration use 12/8. The report treats `mvr.cli ingest` as baseline
   because it is the documented verified command and has direct default tests.

## Required pipeline diagram

```text
VIDEO
  |
  v
INGESTION
  FFprobe -> PySceneDetect ContentDetector(threshold=27.0)
  -> midpoint/long-shot sampling -> FFmpeg exact JPEG extraction
  |
  v
KEYFRAMES + keyframe_mapping.csv
  |
  +-- OCR     -> EasyOCR Reader(vi,en) [detector/recognizer checkpoint unresolved]
  |               -> F-1/F/F+1 Levenshtein + IoU consolidation
  |
  +-- ASR     -> FFmpeg mono 16 kHz window -> Silero VAD
  |               -> ChunkFormer khanhld/chunkformer-ctc-large-vie@14a68262...
  |
  +-- VISUAL  -> OpenCLIP ViT-B-32 / laion2b_s34b_b79k -> L2-normalized vectors
  |               -> exact FAISS IndexFlatIP for the semantic baseline
  |
  `-- OBJECT  -> Faster R-CNN fasterrcnn_resnet50_fpn / torchvision DEFAULT
                  score=0.5, NMS=0.5 (exact default-weight artifact unresolved)
```

## Final note

This audit identifies current executable defaults; it does not establish a
model-quality ranking, claim that all individual paths have been run together,
or authorize a model/configuration change.
