# BEiT-3 offline extraction

The BEiT-3 provider uses the locked `beit3_base_itc_patch16_224.pth`
checkpoint and Microsoft UniLM source revision recorded in
`configs/embedding_providers.json`. It never downloads weights implicitly.

Prerequisites:

1. Create an isolated BEiT-3 environment; do not downgrade the main project
   environment. The optional `beit3` dependency group declares the adapter's
   Python packages, while the official legacy environment and remaining
   upstream pinning risks are recorded in the provider spec.
2. Check out `microsoft/unilm` at the locked source revision and put its
   `beit3` directory on `PYTHONPATH`.
3. Download the exact official base ITC checkpoint and tokenizer outside the
   repository. Model weights must remain ignored and must not be committed.

Example offline image extraction:

```powershell
python scripts/run_feature_extraction.py `
  --mapping-csv <canonical-keyframe-mapping.csv> `
  --keyframe-root <canonical-keyframes> `
  --output <feature-root> `
  --embedding-providers beit3 `
  --beit3-checkpoint <beit3_base_itc_patch16_224.pth> `
  --embedding-batch-size 16 `
  --device cuda
```

The provider output is `<feature-root>/embeddings/beit3/`. Extraction validates
all identities and images before inference, emits one `.npy` shard per video,
writes the mapping in canonical input order, normalizes/validates every vector,
and publishes the directory only after its final manifest is complete. Existing
provider output is never overwritten.

The manifest records the actual local checkpoint SHA256, the locked source
revision, mapping SHA256, dimension, dtype, normalization state, and keyframe
count. Run `audit_embedding_artifact()` to reproduce the identity/vector gate.

The real five-keyframe pytest target is disabled unless all three environment
variables are explicitly supplied:

```powershell
$env:MVR_BEIT3_FASTTEST_MAPPING = "<five-row-mapping.csv>"
$env:MVR_BEIT3_FASTTEST_KEYFRAMES = "<keyframe-root>"
$env:MVR_BEIT3_CHECKPOINT = "<checkpoint.pth>"
python -m pytest -q tests/integration/test_beit3_fasttest.py
```

No retrieval index or online text encoder is created in Task 06.
