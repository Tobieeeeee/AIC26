# Multi-embedding fasttest and ablations

Run the committed structural fixture:

```powershell
python scripts/verification/run_multi_embedding_ablation.py `
  --config configs/multi_embedding_fasttest.json
```

Generated `results.json` and `report.md` are written under the ignored
`artifacts/verification/multi_embedding_fasttest/` directory. The configuration
contains exactly seven provider sets: three single-model runs, all three
two-model RRF pairs, and the three-model RRF run.

The committed fixture verifies orchestration and deterministic reporting; it
does not contain real model vectors or representative ground truth. Therefore
all quality metrics are `not evaluated`, and the report makes no claim that any
combination improves retrieval quality. Real runs must replace the synthetic
provenance status with actual provider artifact/index manifest SHA256 values.

The integration test `test_multi_embedding_fasttest_e2e.py` executes the full
five-keyframe structural chain using deterministic injected encoders/backends:

```text
5 canonical mappings
→ 3 offline provider artifacts
→ Task05A independent validation
→ 3 provider-specific indexes
→ 3 matching online query vectors/searches
→ semantic RRF
→ 5 canonical fused identities
```

Warm-up executions are excluded from measured samples. Reports separately show
per-provider encoding/search, total query, and fusion latency. Model-loading
latency is explicitly labeled as excluded after warm-up.
