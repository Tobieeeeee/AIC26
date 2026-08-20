# Real Three-Provider Fasttest Report

Status: **BLOCKED** — two real providers passed, but the exact official BEiT-3
checkpoint and tokenizer are unavailable. Task05A, index creation, online text
retrieval, and RRF were intentionally not started after that provider blocker.

## Repository state

- Branch: `feat/rebuild-embedding-models` (not `main`).
- Original worktree: clean.
- Structural baseline before this run: `429 passed, 9 skipped`; `git diff --check`
  passed.
- Real outputs are under ignored `artifacts/fasttest_real_3models`; the original
  `artifacts/fasttest` was not modified.

## Original fasttest inventory

The recursive inventory contained 99 files (6,904,923 bytes): 74 JSON, 20 JPEG,
3 CSV, and 2 NPY files. The five canonical images are real 1280x720 RGB frames.
The existing 512-dimensional CLIP vectors are real legacy OpenCLIP outputs, while
the Task14 K1-K5 ablation vectors/rankings are synthetic injected fixtures. No
BEiT-3, MetaCLIP-2, Perception Encoder, or provider-specific FAISS artifact was
present before this run.

Authoritative mapping:
`artifacts/fasttest/ingest/keyframe_mapping.csv`, SHA256
`fb3148fe1f1f6d98a879d4d26146615caffe04e0939f456c434524bad4691c51`.

## Selected canonical keyframes

| keyframe_id | video_id | frame_id | shot_id | timestamp | image_path |
|---|---|---:|---:|---:|---|
| L21_V010_S0007_K01 | L21_V010 | 578 | 7 | 23.12 | keyframes/L21_V010/L21_V010_S0007_K01.jpg |
| L21_V010_S0014_K01 | L21_V010 | 822 | 14 | 32.88 | keyframes/L21_V010/L21_V010_S0014_K01.jpg |
| L21_V010_S0034_K01 | L21_V010 | 2290 | 34 | 91.6 | keyframes/L21_V010/L21_V010_S0034_K01.jpg |
| L21_V025_S0011_K01 | L21_V025 | 771 | 11 | 30.84 | keyframes/L21_V025/L21_V025_S0011_K01.jpg |
| L21_V025_S0187_K01 | L21_V025 | 17274 | 187 | 690.96 | keyframes/L21_V025/L21_V025_S0187_K01.jpg |

Validation: 5 rows, 5 unique IDs, 5 existing/readable real images, two videos,
zero missing images, and zero duplicates.

## Runtime isolation and hardware

Host physical RAM is 33,945,935,872 bytes. No NVIDIA runtime was detected;
both completed providers used CPU inference. The combined ASR image was not
modified.

| Provider | Runtime | Python / Torch | Provider packages | `pip check` |
|---|---|---|---|---|
| MetaCLIP-2 | external `.venv` plus `mvr-embedding-metaclip2:real-fasttest` (`sha256:356b2ed6…`, 334,031,603 bytes) | 3.12.14 / 2.5.1+cpu | torchvision 0.20.1+cpu, transformers 4.57.1, NumPy 2.1.2, Pillow 11.0.0, sentencepiece 0.2.0 | PASS |
| Perception Encoder | `mvr-embedding-perception-encoder:real-fasttest` (`sha256:ce6270e…`, 303,915,265 bytes) | 3.12.14 / 2.5.1+cpu | torchvision 0.20.1+cpu, timm 1.0.15, einops 0.8.1, NumPy 2.1.2, Pillow 11.0.0 | PASS |
| BEiT-3 | not created because the asset gate failed first | N/A | N/A | NOT RUN |

## Exact models and real offline evidence

The task prompt's `768768` BEiT-3 dimension is a typo. The authoritative provider
configuration and pinned Microsoft source both specify 768.

### MetaCLIP-2 — PASS

- Model: `facebook/metaclip-2-worldwide-b16` at revision
  `8cc8591f32b43a809f3325192d3b77aa56968ddb` (CC-BY-NC-4.0, public access
  succeeded). Model weights SHA256:
  `5ba4b60a28a3f563b0d2cf1f3de14298b6592ca9cbe53b2694936ff8bb7b5eba`.
- Executed adapter: `MetaClip2ImageEncoder.encode_images`, using
  `AutoProcessor` and `AutoModel.get_image_features` from the pinned repository.
- Real command used the production `scripts/run_feature_extraction.py`, explicit
  `--allow-model-downloads`, `--device cpu`, batch size 5, and only the
  `metaclip2` provider.
- Wall time including first model download: 195.6 seconds.
- Artifact: two shards `(3, 512)` and `(2, 512)`, float32, all finite/nonzero,
  all five rows unique, norms 0.99999994-1.0.
- Shard SHA256: `L21_V010.npy`
  `613077117bfb1946f70fad11458a67e0a68989441882943a1221eb13afe3e31c`;
  `L21_V025.npy`
  `4027772b59da612a13a7ab03c0e1433dc1bd55fd424fa4379e7ef6e81f873c9d`.
- Audit: expected/valid 5/5; missing/extra/duplicates/invalid 0/0/0/0.
  Manifest SHA256:
  `95ebd374134705ced3ee22f4b5f440a444cbabc6c35ad8c8fca265e3f242a938`.

### Perception Encoder — PASS

- Model: `facebook/PE-Core-L14-336:PE-Core-L14-336.pt`, revision
  `ee1fdb467ab42497fa2e60e94a79e9d7383c1cbd`, source revision
  `65a4a17a587b6a89eaa0dc08f231c2cb1c285f9f`.
- Checkpoint: 2,684,747,432 bytes; SHA256
  `0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8`
  (matches the lock).
- Executed adapter: `PerceptionEncoderImageEncoder.encode_images`, official
  `CLIP.from_config("PE-Core-L14-336", checkpoint_path=...)`, and official
  `get_image_transform`. Load reported no missing or unexpected keys.
- Real command used the production feature runner with `--device cpu`, batch
  size 5, explicit local checkpoint, and only `perception_encoder`.
- Wall time: 62.3 seconds.
- Artifact: two shards `(3, 1024)` and `(2, 1024)`, float32, all
  finite/nonzero, all five rows unique, norms exactly 1.0.
- Shard SHA256: `L21_V010.npy`
  `d9c6dc7da0d27f652a28cbb38d165bb75794c18e52411fd41fc5e251d37ba8ce`;
  `L21_V025.npy`
  `8dc861c02d7f6198602e46c8932899e6b6a808b586fc6ef409c50713cf0529aa`.
- Audit: expected/valid 5/5; missing/extra/duplicates/invalid 0/0/0/0.
  Manifest SHA256:
  `b72848b695b6d9dc420476fe7299d66888cd03cb738f926786b1f99f4d05f16f`.

### BEiT-3 — BLOCKED

- Required checkpoint: `beit3_base_itc_patch16_224.pth`; required tokenizer:
  official `beit3.spm`; source revision
  `69961945a2fd283f2beed74d1f3c543265c9930c`; expected dimension 768.
- The exact official Microsoft Azure URLs return HTTP 409 because public access
  is disabled. A previously published SAS URL returns HTTP 403
  `AuthenticationFailed`. No matching local asset exists under `C:\Code_Save`,
  Downloads, or the read-only source labs.
- Third-party reuploads were rejected because Microsoft publishes no immutable
  checkpoint hash, so byte identity cannot be established. No substitute,
  mock, random vector, or fallback provider was used.

## Downstream gates

Task05A multi-provider validation: NOT RUN. Provider-specific FAISS indexes:
NOT BUILT. Online text encoders and queries: NOT RUN. Cross-provider routing
proof: NOT RUN. RRF(k=60): NOT RUN. This is deliberate: the failure policy
requires stopping before downstream fusion when any provider fails.

The reserved create-only namespaces are:
`dev__l21_fast5_real__beit3`, `dev__l21_fast5_real__metaclip2`, and
`dev__l21_fast5_real__perception_encoder`. No existing index was deleted or
overwritten.

## Tests and Git hygiene

- Pre-run full suite: `429 passed, 9 skipped` — PASS.
- MetaCLIP real artifact audit — PASS.
- Perception Encoder real artifact audit — PASS.
- Provider runtime `pip check` — PASS for the two created runtimes.
- Post-change full suite in the isolated CI container: `429 passed, 9 skipped`
  in 18.07 seconds — PASS; container `pip check` — PASS. The general suite
  keeps opt-in tests skipped until their provider-specific environment variables
  are supplied.
- Activated MetaCLIP-2 opt-in real test: `1 passed` in 138.32 seconds — PASS.
- Activated Perception Encoder opt-in real test: `1 passed` in 77.14 seconds —
  PASS.
- Activated BEiT-3 opt-in real test: NOT RUN/BLOCKED by the missing official
  checkpoint and tokenizer.
- `audit_multi_embedding_release.py`: structural gate PASS, operational gate
  BLOCKED. Its static blocker list does not discover the separate real artifact
  root, so this report and the two provider audit JSON files are the real-runtime
  evidence for MetaCLIP-2 and PE.
- `git diff --check`: PASS. Release audit found no staged/tracked banned artifact,
  oversized file, or suspicious secret. The output root is explicitly ignored.
- Full-corpus dry run: NOT RUN because the prompt permits it only after the real
  three-provider five-keyframe acceptance gate passes.

Generated checkpoints, caches, vectors, images, reports, and indexes remain
untracked. Tracked changes are the provider Dockerfiles, real-fasttest config,
and this report.

## Required unblock action

Provide the exact official Microsoft files
`beit3_base_itc_patch16_224.pth` and `beit3.spm` (or a renewed official download
URL) and place them at the paths in `configs/multi_embedding_real_fasttest.json`.
The checkpoint SHA256 must be captured before execution. Then resume at the
BEiT-3 runtime/model-load gate; do not rerun or replace the two valid provider
artifacts unless their inputs change.

Recommendation: **not ready for the full-corpus dry run acceptance gate** until
BEiT-3 real image/text inference, Task05A, three indexes, three query paths, and
RRF all pass on these same five IDs.
