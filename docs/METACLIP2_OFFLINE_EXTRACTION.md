# MetaCLIP-2 offline extraction

This provider uses only the locked `facebook/metaclip-2-worldwide-b16`
repository at revision `8cc8591f32b43a809f3325192d3b77aa56968ddb`.
The model and processor are loaded from the same revision. The adapter defaults
to `local_files_only=True`; it never silently downloads or substitutes a model.

Prepare the pinned Hugging Face snapshot outside Git, then run:

```powershell
python scripts/run_feature_extraction.py `
  --mapping-csv <canonical-keyframe-mapping.csv> `
  --keyframe-root <canonical-keyframes> `
  --output <feature-root> `
  --embedding-providers metaclip2 `
  --metaclip2-model <pinned-local-snapshot> `
  --embedding-batch-size 32 `
  --device cuda
```

An operator may pass `--allow-model-downloads` only as an explicit decision;
the pinned revision remains mandatory. Generated output lives at
`<feature-root>/embeddings/metaclip2/` and follows the same atomic mapping,
manifest, identity, checksum, and vector audit used by BEiT-3 while retaining
MetaCLIP-2's own preprocessing and dimension.

The model is CC-BY-NC-4.0. Preserve attribution and verify that the intended
competition/deployment use complies with the non-commercial restriction.

The real five-keyframe test requires explicit local inputs:

```powershell
$env:MVR_METACLIP2_FASTTEST_MAPPING = "<five-row-mapping.csv>"
$env:MVR_METACLIP2_FASTTEST_KEYFRAMES = "<keyframe-root>"
$env:MVR_METACLIP2_MODEL = "<pinned-local-snapshot>"
python -m pytest -q tests/integration/test_metaclip2_fasttest.py
```
