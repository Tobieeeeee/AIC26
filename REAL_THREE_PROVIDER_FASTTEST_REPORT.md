# Real Three-Provider Fasttest Report

Status: **PASS for the five-keyframe real-runtime gate**. BEiT-3,
MetaCLIP-2, and Perception Encoder all completed real image inference, real text
inference, provider-specific validation/indexing, independent retrieval, and
rank-only RRF on the same five canonical keyframes.

This is a runtime/integration result, not a retrieval-quality claim. The final
full-corpus release gate remains blocked until the operator configures and
explicitly approves the full build and a representative ground-truth benchmark
is available.

## Repository and data boundary

- Branch: `feat/rebuild-embedding-models` (not `main`).
- Canonical mapping:
  `artifacts/fasttest_real_3models/ingest/keyframe_mapping.csv`.
- Mapping SHA256:
  `fb3148fe1f1f6d98a879d4d26146615caffe04e0939f456c434524bad4691c51`.
- The deterministic provider feature mappings all have SHA256
  `96169f5dd3b7d2ba8a239e049ccc260986a8a6c126bd9742175c7acab5d6ef5b`;
  this is distinct from the canonical ingest mapping hash above.
- Input: five real 1280x720 keyframes from `L21_V010` and `L21_V025`.
- Runtime outputs: ignored `artifacts/fasttest_real_3models/` only.
- Earlier generated evidence was preserved under the ignored
  `backup_pre_execution_metadata_20260821/` directory before regeneration.
- No checkpoint, tokenizer, vector, index, cache, dataset, or generated report
  was staged.

| keyframe_id | video_id | frame_id | shot_id | timestamp |
|---|---|---:|---:|---:|
| L21_V010_S0007_K01 | L21_V010 | 578 | 7 | 23.12 |
| L21_V010_S0014_K01 | L21_V010 | 822 | 14 | 32.88 |
| L21_V010_S0034_K01 | L21_V010 | 2290 | 34 | 91.60 |
| L21_V025_S0011_K01 | L21_V025 | 771 | 11 | 30.84 |
| L21_V025_S0187_K01 | L21_V025 | 17274 | 187 | 690.96 |

Input image SHA256 values used by all three providers:

| keyframe_id | Image SHA256 |
|---|---|
| L21_V010_S0007_K01 | `342f459eccc08f6e33f596ff19a340b5750f5429a1b26eaec3361d7d859dda11` |
| L21_V010_S0014_K01 | `f334d43d454451d1a7c03e622cc8c93899025fd2972639ac1309738402440c33` |
| L21_V010_S0034_K01 | `1864124dae1cd0f5acbe7dc6b1aa9ca1598fe53d5a09ff6b9fbaaa883f48f8c0` |
| L21_V025_S0011_K01 | `38621a593e1bd27657faea73c5a3c1d72d2c2fb547e7905603d1848f36a78858` |
| L21_V025_S0187_K01 | `aef3cdb439c2279c12b0f41a5dcb32e1546961ff7039366b332d230b9d394f1a` |

## Locked providers and runtime evidence

All providers ran on CPU in separate Python 3.12 / Torch 2.5.1 CPU images.
Every output manifest records the concrete adapter, requested device, batch
size, preprocessing contract, extraction latency, mapping hash, and SHA256 of
each vector shard.

| Provider | Exact model / revision | Dimension | Offline latency | Artifact audit |
|---|---|---:|---:|---|
| BEiT-3 | `beit3_base_itc_patch16_224.pth`; UniLM `69961945a2fd283f2beed74d1f3c543265c9930c` | 768 | 12,318.08 ms | PASS, 5/5 |
| MetaCLIP-2 | `facebook/metaclip-2-worldwide-b16@8cc8591f32b43a809f3325192d3b77aa56968ddb` | 512 | 17,128.56 ms | PASS, 5/5 |
| Perception Encoder | `facebook/PE-Core-L14-336@ee1fdb467ab42497fa2e60e94a79e9d7383c1cbd` | 1024 | 46,391.26 ms | PASS, 5/5 |

For every provider the audit result is:

```text
expected keyframes = 5
valid vectors      = 5
unique IDs         = 5
missing IDs        = 0
extra IDs          = 0
duplicates         = 0
invalid vectors    = 0
```

Provider manifest SHA256 values:

- BEiT-3:
  `adf7bf1070f501e48d5dcad39c066f5244219f34328309fff27ca8475055f04d`.
- MetaCLIP-2:
  `44334adc8cf13f0cf38389f311f317a45b0549ba9d719b54a0d5a4bfd6e13064`.
- Perception Encoder:
  `d9c48ca79b5c7f7ca826a467dff743a70a7f855d3623c387aff2cfe1a52ee85e`.

### BEiT-3 source resolution

The current official Microsoft UniLM BEiT-3 page links working GitHub Release
URLs for the exact base ITC checkpoint and `beit3.spm` tokenizer. The
runtime uses the pinned UniLM source revision and the official checkpoint pair;
it does not use a third-party model substitute.

The UniLM code is MIT licensed. The linked checkpoint/tokenizer assets are not
redistributed by this repository; their separate license terms must be
confirmed before any redistribution.

- Checkpoint size: 445,674,625 bytes.
- Checkpoint SHA256:
  `d483a41d92b9aea7ef87d17551413b8c22faae927110666b01ac3aa7a199a25b`.
- Tokenizer size: 1,356,293 bytes.
- Tokenizer SHA256:
  `6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4`.
- Container image digest:
  `sha256:b6670d30d49bd6b45c6af83d9bc5c9b4014eaa48311cd372f62eeb006c16fece`.
- The minimal `torch._six.inf` compatibility shim is limited to the single
  legacy symbol imported by the pinned UniLM source.

## Task05A and provider-specific indexes

Task05A generated five documents. Every document has:

```json
{
  "indexability": {
    "vector": true,
    "embeddings": {
      "beit3": true,
      "metaclip2": true,
      "perception_encoder": true
    }
  }
}
```

Each provider reports vectors expected/valid/invalid `5/5/0`, with zero
mapping mismatch, missing ID, extra ID, or duplicate ID. Three independent
FAISS `IndexFlatIP` collections were created using normalized float32 vectors:

- `dev__l21_fast5_real__beit3`: dimension 768, indexed 5, failed 0.
- `dev__l21_fast5_real__metaclip2`: dimension 512, indexed 5, failed 0.
- `dev__l21_fast5_real__perception_encoder`: dimension 1024, indexed 5,
  failed 0.

The production routing guard rejected a BEiT-3 query vector sent to the
MetaCLIP-2 index: `query provider beit3 cannot search metaclip2 index`.

## Real online queries and RRF

All three provider-specific text encoders generated finite, normalized vectors
with their own dimensions. Each provider returned an independent 1-based ranked
list before fusion. RRF used only ranks, with `k = 60`; raw cross-model
similarity scores were not fused.

Recorded query/index evidence is replayed only after validating provider
identity, model/source revision, adapter, tokenizer contract, dimension, dtype,
normalization, query-vector SHA256, index contract, and source embedding
manifest SHA256:

```powershell
python scripts/verification/run_real_multi_embedding_retrieval.py `
  --provider-specs configs/embedding_providers.json `
  --query-artifacts artifacts/fasttest_real_3models/reports/query_embeddings `
  --indexes-root artifacts/fasttest_real_3models/indexes/semantic `
  --embeddings-root artifacts/fasttest_real_3models/features/embeddings `
  --output artifacts/fasttest_real_3models/reports/verified_retrieval_rrf.json
```

| Query | RRF rank 1 | Original frame | Total measured latency |
|---|---|---:|---:|
| Q1: women receiving pension payments at an office | `L21_V010_S0007_K01` | 578 | 73,193.25 ms |
| Q2: a damaged car and rescuer at a traffic accident | `L21_V010_S0014_K01` | 822 | 543.67 ms |
| Q3: a white police van parked on a street | `L21_V025_S0011_K01` | 771 | 487.22 ms |

Q1 includes first-load time for all three model runtimes. Q2 and Q3 are warm
measurements. These five images were selected for engineering validation, not
representative accuracy evaluation, so no provider or three-model superiority
claim is made.

## Verification

- Offline/provider unit tests: `29 passed`.
- Real BEiT-3 opt-in integration: `1 passed`.
- Three provider identity/artifact audits: PASS.
- Task05A provider validation: PASS.
- Three provider-specific FAISS indexes: PASS.
- Three real text-query paths plus independent retrieval: PASS.
- Cross-provider rejection and RRF `k=60`: PASS.
- Current full regression after provenance hardening: `438 passed, 9 skipped`
  in 15.62 seconds. All three opt-in real-provider tests were also activated
  separately and passed.
- `git diff --check`: PASS.
- Structural multi-embedding release audit: PASS.
- Full-corpus dry run: PASS as a dry run; source archive audit found 7,800
  expected/unique keyframes across 29 videos with zero missing/extra/duplicate.

## Remaining release blockers

The fasttest gate is complete. A non-dry-run full-corpus build was not launched
because it requires explicit approval and resolved commands/paths. The static
release audit therefore correctly keeps the operational release gate blocked
for full-corpus extraction, per-provider audits, full indexes, online smoke,
and representative-ground-truth evaluation. In particular, this report does
not convert the five-keyframe smoke result into a competition-quality claim.
