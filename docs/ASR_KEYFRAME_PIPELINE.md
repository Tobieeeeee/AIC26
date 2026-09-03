# Per-keyframe ASR pipeline

This component creates a Vietnamese ASR artifact for every authoritative
keyframe. It does **not** transcribe a complete video and map a global
transcript back to keyframes.

```text
keyframe timestamp t
  -> decode mono 16 kHz audio [t - 3 s, t + 3 s], clamped to the video
  -> run Silero VAD on that keyframe window
  -> transcribe only detected speech regions with ChunkFormer
  -> write one JSON file for that keyframe
```

The VAD and ASR models are instantiated once for the CLI job. Each keyframe
still has its own audio window and its own VAD call. A window without detected
speech produces `text: ""` and `segments: []`; ASR is not called.

The activated-R2 full enrichment runner uses
`FfmpegCachedAudioWindowProvider`: it decodes one video's complete mono 16 kHz
audio to a memory-mapped PCM file, slices the same independent windows locally,
and evicts that PCM when advancing to the next video. This preserves the
per-keyframe +/-3 second, VAD, and ChunkFormer contract while removing one
FFmpeg process launch per keyframe. The original independent-window provider
remains available for the standalone CLI and compatibility tests.

## Authoritative input mappings

Two existing repository layouts are supported:

1. An ingestion output root containing `keyframe_mapping.csv` and
   `video_metadata.csv`. All identity fields, including `shot_id`, are read
   directly from those files.
2. An organizer AIC root containing `videos/<video_id>.mp4` and
   `map_keyframes/<video_id>.csv`. `keyframe_id`, `frame_idx`, and `timestamp`
   come from `n`, `frame_idx`, and `pts_time`. That organizer CSV has no shot
   identity, so `shot_id` is `null`; the pipeline never guesses one.

In particular, `frame_idx` is never calculated as `timestamp * fps`.

## Output

```text
<output>/
  <video_id>/<keyframe_id>.json
  manifest.json
  errors.json
```

Segment `start` and `end` values are absolute timestamps in the original
video. `errors.json` distinguishes `PROCESSING_ERROR` from valid no-speech
artifacts and records `NO_AUDIO_STREAM` warnings. Failed artifacts retain the
canonical JSON shape and are retried on resume. Existing valid artifacts are
skipped unless `--force` is supplied.

## Native commands

The `asr` optional extra declares the exact ChunkFormer and Silero packages.
Upstream `chunkformer==1.2.2` also declares DeepSpeed, whose isolated build
cannot see an already installed Torch on the verified Windows/Python 3.13
host. Therefore a clean native Windows install is currently blocked; do not
downgrade the project or substitute another model. Use the isolated Linux CPU
image below. In an already prepared compatible Linux/Python 3.11 environment,
the project extra used by the final Docker build step is:

```powershell
python -m pip install -e ".[dev,asr]"
```

The Dockerfile installs pinned CPU Torch first and then pinned DeepSpeed with
build isolation disabled, which is the minimal workaround for the upstream
packaging issue. The default runtime policy is local-files-only. Use
`--allow-download` only when downloading the model into the configured
Hugging Face cache is intended.

Subset smoke run:

```powershell
py -3.12 -m mvr.cli build-asr-features `
  --dataset-root "D:\datasets\aic" `
  --output "artifacts\features_aic\asr" `
  --videos L21_V010,L21_V025 `
  --device cpu
```

Production run, with the exact target model and default +/-3 second window:

```powershell
py -3.12 -m mvr.cli build-asr-features `
  --dataset-root "D:\datasets\aic" `
  --output "artifacts\features_aic\asr" `
  --pre-seconds 3 --post-seconds 3 `
  --device cuda --allow-download
```

No model fallback is provided. `khanhld/chunkformer-ctc-large-vie` is loaded
through `chunkformer==1.2.2` at pinned model revision
`14a68262fa5fc6d497cb107c18bc5a9308ef21ca`; unavailable weights or an
incompatible runtime fail clearly. The model's CC-BY-NC license must be
reviewed before non-research use.

## Docker CPU runtime

The CPU image keeps ASR dependencies isolated from the host and uses the same
worker and output schema as CUDA:

```powershell
docker build -f docker/asr/Dockerfile.cpu -t mvr-asr:cpu .
docker run --rm `
  -e MVR_DOCKER_IMAGE_TAG=mvr-asr:cpu `
  -v "D:\datasets\aic:/data:ro" `
  -v "D:\models\mvr:/models" `
  -v "D:\outputs\aic:/output" `
  mvr-asr:cpu build-asr-features `
  --dataset-root /data --output /output/asr `
  --videos L21_V010 --device cpu --allow-download
```

After the first download, omit `--allow-download` to verify local-only model
loading. Input is read-only; caches and outputs use separate volumes. Model
weights and generated artifacts must not be committed.

## Tests

Default unit tests use fake VAD/ASR providers and never access the network.
The real runtime test is opt-in and also defaults to local model files:

```powershell
$env:MVR_RUN_ASR_RUNTIME_TESTS="1"
$env:MVR_ASR_TEST_DATASET_ROOT="D:\datasets\aic"
$env:MVR_ASR_TEST_VIDEO_ID="L21_V010"
py -3.12 -m pytest -q -s tests\integration\test_asr_runtime.py
```

This smoke test validates execution and artifacts; it does not claim ASR
quality or word-error-rate performance.

Current verification status: the CPU Docker image builds successfully and a
real five-keyframe smoke run on `L21_V001` completed with the pinned Silero and
ChunkFormer providers. It produced five canonical JSON artifacts: four with
speech, one without speech, and no failures. The run made four ASR calls and
took 69.8 seconds on CPU after model initialization/download. This verifies
container startup, model load, real CPU inference, and artifact generation; it
does not establish ASR quality or word-error rate. GPU inference remains
unverified. A second cached-model run over keyframes 6-10 produced five more
valid speech artifacts (eight segments, zero failures) in 73.2 seconds. That
run also regression-verified organizer mappings encoded as UTF-8 with BOM. A
third cached-model run over keyframes 11-20 produced ten speech artifacts (14
segments, zero failures) in 88.0 seconds. In total, 20 real keyframes from
`L21_V001` have been runtime-verified.
