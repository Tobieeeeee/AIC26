# Multi-embedding artifact contract

Each offline image-embedding provider owns an independent directory:

```text
features/embeddings/<model_id>/
|-- <video_id>.npy
|-- feature_mapping.csv
`-- manifest.json
```

`manifest.json` is the discovery source of truth. It records provider identity,
checkpoint/revision provenance, the provider-specific dimension and dtype,
normalization state, mapping checksum, keyframe count, and creation timestamp.
Provider directories are discovered in deterministic `model_id` order without
model-name branches.

`feature_mapping.csv` must include `keyframe_id`, `video_id`, `feature_file`,
and zero-based `feature_index`. Optional source identity columns such as
`frame_id` (or legacy `frame_idx`), `shot_id`, and `timestamp` are retained.
All joins use `keyframe_id`; row position is never an identity key.

The shared loader rejects unsafe identifiers, unsupported dtypes, non-positive
dimensions, directory/manifest mismatches, mapping paths outside the provider
root, duplicate identities, and (when requested) SHA256 mismatches. Different
providers may and normally will have different embedding dimensions.

The unified feature CLI now accepts a comma-separated provider contract:

```powershell
python scripts/run_feature_extraction.py `
  --mapping-csv <mapping.csv> `
  --keyframe-root <keyframes> `
  --output <features> `
  --embedding-providers clip
```

Task 04 defines only the contract. Provider inference and provider-specific
indexes are introduced by later tasks; selecting an unavailable provider fails
explicitly instead of falling back to another embedding space.
