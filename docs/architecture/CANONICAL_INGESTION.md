# CANONICAL INGESTION V1

Shot detector: TransNetV2 @ 0.5

Keyframe selector: periodic midpoint + 12s/8s

Status: FROZEN FOR BATCH1 PRODUCTION

Full Batch1 execution: deferred

## Runtime contract

The following entrypoints resolve to the same profile when no ingestion
strategy overrides are supplied:

- `python -m mvr.cli ingest <input> --output <output>`
- `python scripts/run_ingestion.py --input <input> --output <output>`
- `mvr.ingestion.pipeline.run_pipeline(input_path, output_root)`
- `configs/default.yaml`

The resolved detector list is only `transnetv2`, the primary detector is
`transnetv2`, its threshold is `0.5`, the selector mode is `periodic`, the
long-shot threshold is `12.0` seconds, and the periodic interval is `8.0`
seconds.

TransNetV2 initialization or inference errors are surfaced to the caller. No
code path silently changes the detector to PySceneDetect. Because periodic is
resolved before the content-aware branch, default ingestion does not construct
MetaCLIP-2, Perception Encoder, or BEiT-3 selector encoders.

## Frozen semantics

The periodic implementation remains unchanged:

- every shot retains its floor midpoint;
- a shot is long when its duration is greater than or equal to `12.0` seconds;
- long shots additionally sample at `8.0`-second offsets from the shot start;
- selected frame indices are bounded, sorted, and deduplicated by the existing
  implementation;
- shot/keyframe ordinals, IDs, timestamps, inclusive shot bounds, image paths,
  and `keyframe_mapping.csv` semantics are unchanged.

## Explicit non-default modes

These remain available for controls or experiments only:

- detector: `pyscenedetect`;
- selectors: `midpoint` and `hybrid_temporal_content`;
- MetaCLIP-2, Perception Encoder, and BEiT-3 selector experiments under the
  experiment tooling.

Selecting a non-default mode must be explicit. This freeze changes defaults,
not the algorithms of those modes.

## Operations gate

The freeze authorizes configuration and verification only. It does not execute
full Batch1 ingestion. A full Batch1 run remains a separate, explicitly
authorized operation after the production environment has passed its runtime
preflight.
