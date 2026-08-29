# Data Readiness Audit Run

- Audit version: `1.0`
- Run ID: `post-multimodel`
- Branch: `integration/post-multimodel`
- HEAD: `cc50d240781856f03517c8c4b9685e71e1b56b1b`
- Git dirty at audit start: `True`
- Overall status: **BLOCKED**

## Audited paths

- `dataset_root`: `None`
- `artifacts_root`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\artifacts\fasttest`
- `mapping`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\artifacts\fasttest\ingest\keyframe_mapping.csv`
- `videos`: `None`
- `keyframes`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\artifacts\fasttest\ingest`
- `media_info`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\artifacts\fasttest\ingest`
- `ocr`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\artifacts\fasttest\features\text`
- `asr`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\artifacts\fasttest\features\asr`
- `objects`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\artifacts\fasttest\features\objects`
- `embeddings`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\artifacts\fasttest\features`
- `processed`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\artifacts\fasttest\processed`
- `indexes`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\artifacts\fasttest\indexes`
- `output_dir`: `D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\SkillPixel\mvr-competition-starter\reports\data_readiness\post-multimodel`

## Section status

| Section | Status | Key counts |
|---|---|---|
| `canonical_mapping` | `READY` |  |
| `videos` | `BLOCKED` |  |
| `keyframes` | `BLOCKED` | expected_canonical_keyframes=5, present_keyframes=0, missing_keyframes=5 |
| `media_info` | `READY` |  |
| `ocr` | `READY` | coverage=1.0, valid_artifacts=5, missing_canonical=0 |
| `asr` | `READY` | coverage=1.0, valid_artifacts=5, missing_canonical=0 |
| `objects` | `READY` | coverage=1.0, valid_artifacts=5, missing_canonical=0 |
| `embeddings` | `BLOCKED` | pending_providers=['beit3', 'metaclip2', 'perception_encoder'] |
| `processed_documents` | `READY` | document_files=5 |
| `index_manifests` | `READY` |  |

## Issues

- Errors/blockers: `10`
- Warnings: `5`
- Pending/info findings: `0`

Only filesystem reads and report writes were performed. No corpus repair, extraction, model inference, download, or index mutation is performed by this audit.

Machine-readable outputs in this run directory: `manifest.json`, `summary.json`, `issues.jsonl`, and `coverage.csv`.
