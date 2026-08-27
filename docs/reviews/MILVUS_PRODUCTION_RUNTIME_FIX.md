# Milvus Production Runtime Fix

## Result

TASK-FIX-01 is implemented on `fix/milvus-production-runtime`. FastAPI now
constructs `MilvusSemanticService` by default, and FAISS remains available only
through the explicit `faiss_reference` backend selection. There is no automatic
Milvus-to-FAISS fallback.

Full Batch1 ingestion and indexing were deliberately not run.

## Git and runtime environment

| Item | Evidence |
|---|---|
| Branch | `fix/milvus-production-runtime` |
| Base commit | `67b4684799ed0e8ded38c5d0fce7effc5de09510` (audit head, directly above Task05 commit `d47ade0`) |
| Implementation commit | `f6bb249` (`fix: make Milvus the production semantic backend`) |
| Runtime interpreter | `.venv/Scripts/python.exe`, Python 3.12.10 |
| PyMilvus | 2.4.15; import and previously dependency-skipped failure test verified in the same project interpreter |
| Milvus server | v2.4.13, image `milvusdb/milvus:v2.4.13` |
| Docker services | `milvus-standalone`, `milvus-etcd`, and `milvus-minio` all `Up (healthy)` |
| Ports | Milvus gRPC `19530`; HTTP health `9091` |
| Push | Successful to `origin/fix/milvus-production-runtime`; upstream tracking configured |

PyMilvus was absent from every available project/runtime interpreter at the
start of the task, so the compatible `pymilvus==2.4.15` dependency was installed
in the project `.venv` and recorded as the `milvus` optional extra. The bare
Anaconda Python 3.9 and system Python 3.12 were not used for repository tests.

## Configuration and backend resolution

Before the fix:

```text
configs/default.yaml and configs/production.yaml
  semantic_backend: milvus

FastAPI _default_semantic_service()
  -> LazyFaissSemanticService
```

After the fix:

```text
MVR_SEMANTIC_BACKEND unset or "milvus"
  -> MilvusSemanticService.from_env()

MVR_SEMANTIC_BACKEND="faiss_reference"
  -> LazyFaissSemanticService

unknown backend or Milvus construction/runtime failure
  -> explicit error; never FAISS fallback
```

`configs/default.yaml`, `configs/production.yaml`, the FastAPI default, and the
Milvus operator documentation now agree on `milvus`. Both YAML files record
the collection, FLAT/IP contract, RRF `k=60`, and provider-to-field mapping.

Hybrid retrieval no longer constructs its own legacy Elasticsearch/OpenCLIP
semantic branch. It receives the same shared semantic service as the semantic,
KIS, and Q&A routes, then combines semantic and lexical identities with the
existing hybrid RRF utility.

## Collection and live fixture

The repository's existing Docker Compose stack was used without connecting
Milvus object storage to R2. The live collection status was:

```text
collection: aic26_keyframes
primary key: keyframe_id
entities: 5
loaded: true

beit3_vector:    dimension 768,  FLAT/IP
metaclip2_vector: dimension 512, FLAT/IP
pe_vector:        dimension 1024, FLAT/IP
```

The deterministic fixture contains five canonical entities. Every entity has
`keyframe_id`, `video_id`, numeric `frame_id`, `timestamp`, and all three
normalized vector fields. An explicit flush was added before index creation;
without it, the first real fixture run exposed a zero-entity count even though
the upsert had returned primary keys.

Each provider was queried only through its matching vector field. With the
fixture query, all three provider lists returned:

```text
kf-0, kf-1, kf-2, kf-3, kf-4
```

The fused first result was `fixture-video/kf-0`, `frame_id=100`,
`timestamp=0.0`, with provider ranks `{beit3: 1, metaclip2: 1,
perception_encoder: 1}` and RRF score `3 / 61 = 0.04918032786885246`.

## FAISS parity

The live integration test built a real FAISS 1.15.0 `IndexFlatIP` for each
provider from the identical vectors. For all three fields, Milvus FLAT and
FAISS returned the same top-five identities in the same order. The existing
mock/pure-math parity and provider-routing assertions also passed unchanged.
No parity assertion was weakened.

## FastAPI smoke and readiness

The default `create_app()` health request against the local collection returned
HTTP 200:

```json
{
  "status": "ok",
  "api": "ready",
  "semantic": {
    "configured": true,
    "available": true,
    "loaded": true,
    "backend": "milvus",
    "reachable": true,
    "collection_ready": true,
    "collection": "aic26_keyframes",
    "entities": 5
  },
  "elasticsearch": {"configured": false, "available": false}
}
```

The live `POST /search/semantic` request used the injected deterministic query
encoders and the real Milvus service/backend path. Its first result preserved:

```text
keyframe_id=kf-0
video_id=fixture-video
frame_id=100
timestamp=0.0
```

No FAISS object participates in that API smoke path. A separate failure-path
test makes Milvus unreachable and verifies that the API remains alive with
degraded semantic readiness, semantic search returns HTTP 503, no Milvus search
is attempted, and no FAISS fallback is constructed.

## Compatibility regressions

| Area | Result |
|---|---|
| KIS | 24 passed; canonical Milvus `keyframe_id` is preferred while numeric `frame_id`/`frame_idx` remains preserved and resolver-checked |
| TRAKE | 78 passed; per-event retrieval, same-video dense alignment, and ATLAS/Lucifer/U-CESE logic were not redesigned |
| Q&A | 75 passed; retrieval, temporal evidence expansion, and `AnswerReasoner` packaging remain unchanged apart from canonical hit identity handling |
| Hybrid / semantic RRF / production defaults | 56 passed; shared semantic service, rank-only fusion, and default resolution verified |
| Milvus schema/search/parity/failure/runtime | 78 passed, 0 skipped |
| Live Milvus integration | 1 passed, 0 skipped, 1 upstream `pkg_resources` deprecation warning |

The audit's historical Milvus baseline said 72 collected tests. On this exact
checkout the four original Milvus files collect 68 tests; all 68 pass with no
dependency skip. Ten new runtime/API tests bring the focused Milvus total to
78. The task-focused aggregate is 243 passed.

TRAKE's dense scoring contract remains intentionally unchanged and still needs
a reconstructable single-vector-space service (`.text_encoder` plus `.index`).
Therefore the Milvus semantic service is used by the shared top-k semantic
contract, while TRAKE continues to require explicitly selecting the retained
FAISS reference backend until a separately approved dense Milvus scoring
design exists. No silent per-route fallback was added.

## Broader verification and hygiene

```text
python compileall: PASS
git diff --check: PASS
full pytest: 1,108 passed, 14 skipped, 4 failed
```

The four full-suite failures are outside this task:

- two untracked ingestion-audit tests expect the older PySceneDetect default;
- one untracked R2 access-flow test supplies a detector missing the current
  `model_family` contract;
- tracked `test_task05a_image_path.py` has the already-documented
  `tests.unit` cross-import bug listed as unrelated in TASK-FIX-01.

No data, model weights, Milvus/MinIO/etcd volumes, credentials, FAISS indexes,
or other generated runtime artifacts were staged. The local Docker fixture
remains only in ignored Docker volumes. Ruff was not available in the project
environment, so no Ruff result is claimed.

## Remaining blockers and deferred work

- Full Batch1 embedding generation, Milvus ingestion, and indexing are
  deferred by task scope.
- Real three-provider production query inference still requires the pinned
  local model/checkpoint/tokenizer assets; the live runtime test correctly
  used deterministic encoders and did not download models.
- Production TRAKE on Milvus needs a separately approved dense per-video
  scoring contract, as described above; the current algorithm and explicit
  FAISS reference path remain intact.
- Elasticsearch live hybrid indexing/smoke, RF-DETR pinning, `/submit`, OCR
  runtime availability, and frontend work remain separate tasks.
