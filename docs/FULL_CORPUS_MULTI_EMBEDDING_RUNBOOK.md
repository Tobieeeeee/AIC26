# Full-corpus multi-embedding runbook

## Current audited dry-run

Run:

```powershell
python scripts/operations/run_full_corpus.py `
  --config configs/full_corpus_build.json `
  --dry-run
```

The 2026-08-21 dry-run read only the supplied ZIP archives. It found 29 L21
videos and 7,800 matching unique `(video_id, n)` identities, with zero
missing, extra, or duplicate identities. Archive SHA256 values were:

- mapping: `982e176fdfa5b324cb3841cb76636ea0379d50575c94028533dc3d595cfe90d5`
- keyframes: `30412ab4c3c5f3bbd7cef0feb59e7670a6683d2d8d4a90b196cea9688d554221`

No archive was extracted and no model/index was loaded or created.

## Required order

The state machine permits only this order and checkpoints every passing stage:

```text
canonical full mapping
→ BEiT-3 extraction
→ BEiT-3 identity/artifact audit
→ MetaCLIP-2 extraction
→ MetaCLIP-2 identity/artifact audit
→ Perception Encoder extraction
→ Perception Encoder identity/artifact audit
→ Task05A multi-embedding validation
→ three provider-specific indexes
→ online smoke
→ benchmark
```

An audit must report `valid=7800` and missing/extra/duplicates/invalid all zero
before the next provider starts. The audit CLI writes failed IDs outside Git:

```powershell
python scripts/operations/audit_provider_artifact.py `
  --provider-root <provider-artifact-root> `
  --canonical-mapping <canonical-keyframe-mapping.csv> `
  --report <outside-git-output>/failed_ids/<provider>.json
```

## Current blockers before approval

- The L21 keyframe archive has not been extracted to the configured full root.
- A canonical CSV with globally unique `keyframe_id` and authoritative
  `video_id`, `frame_id`, `shot_id` (nullable only if the approved schema
  supports it), and timestamp has not been created. Never invent `shot_id`.
- BEiT-3 checkpoint plus `beit3.spm`, pinned MetaCLIP-2 assets, and the pinned
  PE checkpoint are not all present locally.
- Provider-specific five-keyframe real-weight GPU throughput/VRAM has not been
  measured, so the full time/VRAM estimate remains preliminary.
- Commands containing `<...>` in `configs/full_corpus_build.json` must be
  replaced with exact resolved arguments. Dry-run must then show
  `execution_ready: true`.

## Resource and destructive-operation gate

Current preliminary allocation is at least 15 GB free disk and 16 GB host RAM,
one provider at a time, with potentially multiple GPU-hours. These are planning
figures, not measured requirements. Confirm actual GPU VRAM, throughput, model
access, disk, output target, and backup policy before approval.

The configured output is outside Git and relative to the repository root:
`../dataset_aic/output/multi_embedding_full`.
Index policy is `create`; switching to `rebuild` is deliberately rejected by
the runner until a separate destructive-target approval is implemented.

Only after an updated dry-run contains exact commands, no blockers, verified
resources, and explicit owner approval may an operator run:

```powershell
python scripts/operations/run_full_corpus.py `
  --config configs/full_corpus_build.json `
  --approved-expensive-run
```

Do not run this command in the current state.
