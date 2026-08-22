# OCR Production Checkpoint Replacement Report

Date: 2026-08-22

## 1. Branch and HEAD

- Branch: `fix/ocr-production-checkpoints`
- Starting and current HEAD: `2159148`
- No commit, merge, or push was performed.

## 2. Files changed

Production and operator path:

- `docker/ocr/model_sources.lock.json`
- `src/mvr/enrichment/ocr/model_integrity.py`
- `src/mvr/enrichment/ocr/docker_ocr.py`
- `docker/ocr/deepsolo.Dockerfile`
- `docker/ocr/parseq_vi.Dockerfile`
- `docker/ocr/worker/deepsolo_worker.py`
- `docker/ocr/worker/parseq_worker.py`
- `docker/ocr/worker/verify_deepsolo.py`
- `docker/ocr/worker/verify_parseq.py`
- `scripts/ocr/fetch_ocr_models.ps1`
- `scripts/ocr/run_ocr_docker.ps1`
- `scripts/ocr/verify_ocr_docker.ps1`
- `docker/ocr/README.md`

Tests and focused documentation:

- `tests/unit/enrichment/ocr/test_model_integrity.py`
- `tests/unit/enrichment/ocr/test_weight_validation.py`
- `tests/unit/enrichment/ocr/test_docker_ocr.py`
- `docs/LAB_REUSE_MAP.md`
- `docs/CODEX_PROGRESS.md`
- this report

## 3. Old ambiguous sources removed

Active production references were removed for:

- the ambiguous `ViTAE-Transformer/DeepSolo` Hugging Face mirror object named
  `deepsolo_res50_finetune_synth.pth`;
- the mutable-main `Bao-Minh/PARSeq-Vietnamese` source named `parseq_vi.ckpt`.

Neither remains an active fallback. The previous ignored DeepSolo file remains
locally present under its old name because repository rules prohibit deleting a
user's ignored artifact merely because it is no longer used. No runtime path
references it.

## 4. DeepSolo source and exact checkpoint

- Logical model: DeepSolo ResNet-50
- Purpose: scene-text localization
- Target: Total-Text fine-tuned model
- Training context: `Synth150K + MLT17 + IC13 + IC15 + TextOCR`
- Official repository: <https://github.com/ViTAE-Transformer/DeepSolo>
- Official model table at the pinned revision: <https://github.com/ViTAE-Transformer/DeepSolo/blob/dbadae995035246bad3376c7a44c015c69e9b313/DeepSolo/README.md>
- Official checkpoint share: <https://1drv.ms/u/s!AimBgYV7JjTlgcd138p8HaXViFk-tw?download=1>
- Checkpoint filename returned by the official source:
  `tt_res50_finetune_synth-tt-mlt-13-15-textocr.pth`
- Runtime config:
  `/opt/DeepSolo/DeepSolo/configs/R_50/TotalText/finetune_150k_tt_mlt_13_15_textocr.yaml`

The official table row, returned artifact filename, repository config, and
runtime identity now refer to the same fine-tuned checkpoint. No generic or
synthetic-only checkpoint is renamed into this identity.

## 5. DeepSolo revision and SHA256

- Source revision: `dbadae995035246bad3376c7a44c015c69e9b313`
- Artifact size: 171,032,379 bytes
- SHA256:
  `6342945f34781ea90340d11b089c091327115d381fceb62c87b5cd2a82efeed3`
- Trust-root method: downloaded from the official model-table OneDrive share,
  computed locally, then independently matched against the existing ignored
  local bytes.

The rebuilt Docker image reports the same source commit.

## 6. Vietnamese PARSeq public source

- Logical model: PARSeq, Vietnamese-finetuned
- Purpose: scene-text transcription
- Public repository:
  <https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition>
- Pinned upstream download script:
  <https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition/blob/76cc5f3cc6268457aac764653400fdff681f8271/download_models.sh>
- Deterministic public bundle: Google Drive file ID
  `1Az9psFV6C1qiqFCL2s6DIBYRKr3Bkjd7`, distributed as `weights.zip`
- Exact checkpoint member: `weights/rec/best-parseq.ckpt`
- Pinned upstream loader:
  <https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition/blob/76cc5f3cc6268457aac764653400fdff681f8271/src/parseq/load_model.py>

The upstream README states that its PARSeq model was fine-tuned on Vietnamese
data, and its loader consumes the same exact archive member selected here.

## 7. PARSeq revision and SHA256

- Source revision: `76cc5f3cc6268457aac764653400fdff681f8271`
- `weights.zip` size: 840,594,836 bytes
- Bundle SHA256:
  `8d180494052108f20e8e68935fd5dc81061c59fad00c18e7bade64e5ecad24ab`
- Checkpoint size: 287,344,161 bytes
- Checkpoint SHA256:
  `8089b13c5ad115a96a608c6401eaab36b081393ea0a8323537b29a2dc80168f5`
- Trust-root method: downloaded the exact bundle identified by the pinned
  upstream script, hashed the bundle, streamed and hashed the exact checkpoint
  member, then independently matched it against the existing ignored local
  checkpoint.

The rebuilt Docker image reports the same source commit.

## 8. License notes

- DeepSolo's repository license permits non-commercial use and requires a
  separate license for commercial use. Competition/deployment use must be
  checked against those terms.
- The `lynguyenminh` repository is MIT-licensed. Incorporated PARSeq source
  files retain Apache-2.0 notices. The repository does not publish a separate
  checkpoint-specific license, so downstream use should retain both notices
  and confirm the intended deployment context.

## 9. Final lock entries

The schema-versioned production lock now records, for both entries:

- model, variant, and purpose;
- exact checkpoint identity;
- public source repository and immutable source commit;
- exact deterministic download source;
- license note;
- directory, runtime filename, runtime path, and runtime config;
- required lowercase SHA256;
- locally established trust-root note;
- direct-file or ZIP-member acquisition contract.

DeepSolo uses the exact upstream filename. PARSeq records both its bundle SHA256
and inner checkpoint SHA256.

## 10. Fetch integrity behavior

`fetch_ocr_models.ps1` now delegates to the shared production integrity module:

```text
validate strict lock
-> download to a same-directory temporary file
-> verify bundle hash when applicable
-> extract only the locked ZIP member when applicable
-> verify checkpoint SHA256
-> atomically publish with os.replace only after verification
```

All temporary paths are removed in `finally` cleanup. Missing/malformed hashes,
bad archives, missing members, zero-byte/HTML/placeholder responses, and digest
mismatches fail non-zero without publishing the downloaded bytes.

## 11. Existing-file behavior

Every existing final file is hashed before reuse:

- match: reuse without network access;
- mismatch: fail closed and leave the existing file untouched;
- missing: execute the verified temporary-download/publication path.

The real fetch run downloaded and published the newly canonical DeepSolo path,
then hash-verified and reused the existing PARSeq file. No temporary files were
left behind.

## 12. Runtime preflight

Before frame extraction or inference, the host loads the same strict lock and
verifies both files. Each model worker receives the locked digest and repeats
the shared verification immediately before model loading. Worker model mounts
are read-only.

Runtime fails before inference for missing files, zero-byte files,
placeholder/HTML responses, SHA256 mismatch, malformed/missing hashes, missing
production entries, and lock/runtime path disagreement. Explicit synthetic mock
mode remains isolated behind `MVR_OCR_ALLOW_MOCK=1` and never loads a model.

## 13. Runtime filename and path alignment

| Model | Lock/host filename | Docker/runtime path | Loader |
|---|---|---|---|
| DeepSolo | `deepsolo/tt_res50_finetune_synth-tt-mlt-13-15-textocr.pth` | `/models/deepsolo/tt_res50_finetune_synth-tt-mlt-13-15-textocr.pth` | DeepSolo worker with the pinned Total-Text config |
| PARSeq | `parseq_vi/best-parseq.ckpt` | `/models/parseq_vi/best-parseq.ckpt` | pinned `lynguyenminh` `load_from_checkpoint` runtime |

The lock validator derives the expected runtime path from `dir + filename` and
rejects disagreement.

## 14. Integrity test results

Focused OCR suite:

```text
py -3.12 -m pytest -q tests/unit/enrichment/ocr
52 passed in 1.76s
```

Coverage includes correct/wrong digests, verified existing-file reuse, corrupt
existing-file rejection, missing hash rejection, failure cleanup, verified-only
atomic publication, bundle/member verification, wrong runtime bytes, production
lock validation, manifest provenance, worker placeholders, canonical identity,
empty detections, multiple boxes, and temporal consolidation.

Relevant OCR/ingestion/integration regression:

```text
80 passed, 9 skipped
```

The full suite reached `760 passed, 12 skipped` with one unrelated failure:
`tests/unit/indexing/test_task05a_image_path.py` imports
`tests.unit.indexing.test_task05a`, but `tests.unit` is not importable in this
environment. The failure reproduces independently and does not execute an OCR
path.

## 15. Model-load smoke result

PASS for both real assets in the rebuilt pinned production Docker images:

- DeepSolo: SHA preflight passed; CUDA available; intended config loaded;
  checkpoint and predictor initialized; real GPU inference completed.
- PARSeq: SHA preflight passed; CUDA available; Lightning checkpoint loaded
  without a shape/key mismatch; real GPU inference completed.

The PARSeq runtime reported an in-memory Lightning 1.6.5-to-1.9.5 metadata
upgrade notice. Its model mount was read-only and the checkpoint was not
modified.

## 16. End-to-end subset OCR result

PASS on five canonical keyframes from `artifacts/fasttest/ingest`, using the two
local source videos under `artifacts/real_task04_test/videos`:

- five canonical keyframes;
- 15 extracted F-1/F/F+1 sampled frames;
- DeepSolo -> crop/rectify -> Vietnamese PARSeq -> temporal consolidation;
- five canonical output JSON files;
- 41 consolidated text records;
- zero identity/schema validation errors;
- all five outputs exercised multiple-box handling;
- 24 recognized strings contained non-ASCII characters and all files decoded
  as UTF-8;
- empty-detection serialization passed focused synthetic coverage;
- no full-corpus run was started.

This is functional validation only, not an OCR quality evaluation.

## 17. Manifest provenance result

PASS. The generated manifest records:

- detector/recognizer model, variant, and purpose;
- exact checkpoint identity and SHA256;
- source repository and immutable source revision;
- runtime filename, mount path, and runtime config;
- OCR pipeline version `docker-ocr-2.1`;
- unchanged temporal offsets and consolidation settings.

## 18. Model-binary hygiene

- No `.pth`, `.pt`, `.ckpt`, `.bin`, `weights.zip`, cache, or generated OCR
  artifact is tracked or staged.
- Real checkpoints remain ignored. The retained subset evidence remains under
  the untracked `artifacts/_ocr_runtime/` path; none of it is staged.
- `git ls-files -- '*.pth' '*.pt' '*.ckpt' '*.bin' 'weights.zip'` returned no
  files.

## 19. Evaluation boundary

No comparison with the old OCR source was performed. No accuracy, recall,
latency superiority, or quality-improvement claim is made. DeepSolo + PARSeq,
crop/rectify, and F-1/F/F+1 temporal consolidation remain the intentional
production architecture.

## 20. Final verdict

`PASS_WITH_ENVIRONMENT_LIMITATIONS`

All OCR-Fix-02 production, integrity, real-model smoke, and five-keyframe
functional requirements passed. The limitation is the independently
reproducible unrelated Task05A full-suite import failure documented in section
14; all OCR tests and relevant regressions pass.
