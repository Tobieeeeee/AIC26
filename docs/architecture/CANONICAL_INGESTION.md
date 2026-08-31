# CANONICAL INGESTION V2

Shot detector: TransNetV2 @ 0.5

Keyframe selector: `hybrid_temporal_content`

Status: FROZEN FOR BATCH1 PRODUCTION

Full Batch1 execution: deferred

## Runtime contract

The following entrypoints resolve to the same profile when no ingestion
strategy overrides are supplied:

- `python -m mvr.cli ingest <input> --output <output>`
- `python scripts/run_ingestion.py --input <input> --output <output>`
- `mvr.ingestion.pipeline.run_pipeline(input_path, output_root)`
- `python scripts/operations/run_pipeline_local.py ...`
- `python scripts/operations/run_pipeline_cloud.py ...`
- `configs/default.yaml`

The resolved detector list is only `transnetv2`, the primary detector is
`transnetv2`, its threshold is `0.5`, and the selector mode is
`hybrid_temporal_content`. The selector uses a 1-second candidate grid, a
maximum repaired temporal gap of 8 seconds, MetaCLIP-2 local novelty, and
grayscale/Canny change signals.

TransNetV2 initialization or inference errors are surfaced to the caller. No
code path silently changes the detector to PySceneDetect. MetaCLIP-2 identity,
revision, preprocessing, normalization, candidate cache, and artifact hashes
are validated fail-closed; a missing model/cache is not replaced by periodic
selection.

## Frozen semantics

The hybrid implementation uses these defaults:

- candidate step: `1.0` second;
- local MetaCLIP-2 neighbor window: `4.0` seconds with `k=3`;
- novelty threshold: `0.15`, rare-event threshold: `0.25`;
- edge threshold: `max(0.08, median + 3 * MAD)`;
- minimum content-event gap: `2.0` seconds;
- every shot retains its exact floor midpoint;
- temporal repair adds the best candidate needed to keep all gaps `<= 8.0`
  seconds;
- selected frame indices are bounded, sorted, and deduplicated;
- shot/keyframe ordinals, IDs, timestamps, inclusive shot bounds, image paths,
  and `keyframe_mapping.csv` semantics are unchanged.

The selector was chosen over `semantic_dedup` on the committed four-video,
seven-query retrieval benchmark: Recall@1 `0.583` vs `0.333`, MRR `0.667` vs
`0.417`. Periodic and midpoint remain reproducibility controls.

## Explicit non-default modes

These remain available for controls or experiments only:

- detector: `pyscenedetect`;
- selectors: `periodic`, `midpoint`, and `semantic_dedup`;
- periodic preserves the previous midpoint + `duration >= 12s` / every `8s`
  behavior byte-for-byte.

Selecting a non-default mode must be explicit. This freeze changes defaults,
not the algorithms of those modes.

## Operations gate

The freeze authorizes configuration and verification only. It does not execute
full Batch1 ingestion. A full Batch1 run remains a separate, explicitly
authorized operation after the production environment has passed its runtime
preflight. Model downloads remain disabled unless the operator explicitly
passes `--allow-model-downloads`.
