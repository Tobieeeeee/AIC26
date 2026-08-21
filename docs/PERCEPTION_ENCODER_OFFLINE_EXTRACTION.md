# Perception Encoder offline extraction

The provider is locked to `PE-Core-L14-336.pt` at Hugging Face revision
`ee1fdb467ab42497fa2e60e94a79e9d7383c1cbd`, with published SHA256
`0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8`.
The adapter uses Meta's `CLIP.from_config`, `get_image_transform`, and
`encode_image(normalize=True)` implementations from the pinned source commit.

Keep the 2.68 GB checkpoint and source checkout outside Git. Put the source
checkout on `PYTHONPATH`, then run PE independently:

```powershell
.\scripts\run_all_features.ps1 `
  -MappingCsv "<canonical-mapping.csv>" `
  -KeyframeRoot "<keyframes>" `
  -OutputRoot "<features>" `
  -EmbeddingProviders "perception_encoder" `
  -PerceptionEncoderCheckpoint "<PE-Core-L14-336.pt>" `
  -Device "cuda"
```

The same wrapper supports all locked providers together:

```powershell
.\scripts\run_all_features.ps1 `
  -MappingCsv "..." `
  -KeyframeRoot "..." `
  -VideoMetadataCsv "..." `
  -VideoRoot "..." `
  -OutputRoot "..." `
  -EmbeddingProviders "beit3,metaclip2,perception_encoder" `
  -Beit3Checkpoint "..." `
  -MetaClip2Model "..." `
  -PerceptionEncoderCheckpoint "..."
```

`VideoMetadataCsv` and `VideoRoot` are accepted as stable full-pipeline
parameters but are not consumed by image embedding extraction. Each provider
is run independently and publishes only its own directory. The PE checkpoint
hash is verified before inference; mismatch fails closed.

The external five-keyframe target requires `MVR_PE_FASTTEST_MAPPING`,
`MVR_PE_FASTTEST_KEYFRAMES`, and `MVR_PE_CHECKPOINT`. It never downloads a
checkpoint.
