# R2 Offline Access & Cost Evidence Audit

Date: 2026-08-22
Authoritative task: `agent_task/DATA02R_R2_ACCESS_PATTERN_COST_EVIDENCE_AUDIT.md`
Final verdict: **PASS_WITH_UNMEASURED_FULL_BATCH_VARIABLES**

## 1. Repository HEAD and audit boundary

| Item | Value |
|---|---|
| Repository HEAD | `37696aa fix: pin production OCR checkpoints` |
| Branch | `fix/ocr-production-checkpoints` |
| Production modifications | None |
| Audit-only additions | `tests/benchmark/test_r2_access_call_flow.py`; `scripts/audit/run_r2_access_canary.py`; this report |
| Full Batch1 run | Not run |
| R2 bucket mutation | One temporary canary prefix, removed after measurement |
| Merge / push | Not performed |

The working tree already contained unrelated untracked user files and directories
before this audit; none were removed or altered. The untracked
`AIC26_SHARED_DATA_ARCHITECTURE_V1.md` is planning material, not an execution
instruction for this audit. The task file named above and the user request were
the authority for scope and stop conditions.

## 2. Test environment and verification gate

The audit and canary ran on the repository's Windows PowerShell host with
`py -3.12`, AWS CLI, FFmpeg/FFprobe, Pillow, and HTTPS access to Cloudflare.
The final focused suite passed:

```text
122 passed, 1 skipped in 14.30s
```

The suite included the four audit-only checks in
`tests/benchmark/test_r2_access_call_flow.py` and the relevant existing
ingestion, ASR, OCR, object, and three-provider embedding tests. The one skip
is the existing optional TransNetV2 runtime gate.

The requested verification gate was performed without printing, logging, or
persisting any credential value:

| Required capability | Non-secret evidence | Result |
|---|---|---|
| Required environment variables | Cloudflare account/token, R2 key pair, AWS-compatible key pair, and endpoint variables were all non-empty | **PASS** |
| Bucket `aic-test` write/read/list | Wrote a unique text canary, read and content-checked it, listed only its dedicated prefix, then deleted it | **PASS** |
| Cloudflare GraphQL authentication | Authenticated account-scoped query returned the requested account | **PASS** |
| `r2OperationsAdaptiveGroups` | Minimal bucket-scoped query completed without GraphQL errors | **PASS** |
| `r2StorageAdaptiveGroups` | Minimal bucket-scoped query completed without GraphQL errors | **PASS** |

No credentials were written to `.env`, a report, an artifact, command output,
or Git. AWS debug logging was deliberately not enabled because it can expose
signed request material.

## 3. R2 Standard pricing source

This audit uses Cloudflare R2 **Standard** pricing as documented on
2026-08-22: [R2 pricing](https://developers.cloudflare.com/r2/pricing/).

| Dimension | Included monthly usage | Standard overage |
|---|---:|---:|
| Storage | 10 GB-month | $0.015 / GB-month |
| Class A operations | 1,000,000 | $4.50 / million |
| Class B operations | 10,000,000 | $0.36 / million |
| Internet egress from R2 | — | $0 |

Cloudflare rounds each usage dimension up to its billing unit. Storage is the
average **daily peak** over the billing period, expressed in GB-month; it is
not simply all bytes ever written. The current analytics documentation confirms
that `r2OperationsAdaptiveGroups` is the operation authority and
`r2StorageAdaptiveGroups` is the storage authority:
[R2 metrics and analytics](https://developers.cloudflare.com/r2/platform/metrics-analytics/).

## 4. Local media inventory: actual values and limits

### 4.1 Raw video bytes

No authoritative Batch1 raw-video root is present. `data/raw/` contains no
videos and the configured `../dataset_aic/videos/` root is missing. The only
locally measurable raw-video canary is an existing two-video ASR fixture:

| Local source | Videos | Bytes | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| `artifacts/real_task04_test/videos/` | 2 | **279,403,661** | 134,782,223 | 139,701,830.5 | 144,621,438 |

The files are `L21_V010.mp4` and `L21_V025.mp4`. This is an exact local
fixture measurement only; it is **not** the Batch1 raw-video count or total.

### 4.2 Regenerated keyframe bytes

No authoritative regenerated Batch1 mapping or full keyframe root is present.
Two exact local, non-full fixtures are available:

| Local source | Keyframes | Bytes | Min | Median | p95 | Max | Mapping evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| `artifacts/ingest_public_test/keyframes/` | 1,053 | **106,068,695** | 8,435 | 91,093 | 195,956 | 415,413 | 1,053 unique `(video_id,keyframe_id)` rows across 250 video IDs |
| `artifacts/keyframes/phase02-smoke/keyframes/` | 5 | **31,235** | 926 | 3,864 | 16,143 | 16,143 | five-keyframe smoke only |

The 2-video raw fixture and 1,053-keyframe fixture are from different local
test sources and must not be added together as one corpus.

**Authoritative Batch1 regenerated keyframe count and bytes: unavailable.**
The planning figures `~900` videos and `~200,000` keyframes are deliberately
not used in a cost calculation.

## 5. Static access audit and scheduling proof

`VIDEO_NATIVE` means one video is opened/read once to generate multiple
outputs. `KEYFRAME_NATIVE` means each keyframe image is an independent input.
`MIXED` means both video-level and per-keyframe access occur in the current
call flow.

| Stage | Execution entrypoint | Input / file-open unit | Grouping proved by call flow | Classification | Video jobs | Collection jobs | Output unit |
|---|---|---|---|---|---|---|---|
| Ingestion | `scripts/run_ingestion.py` -> `mvr.ingestion.pipeline.run_pipeline` | Raw video | `run_pipeline` discovers then iterates `videos`; selected keyframes are regrouped with `by_video` before `extract_keyframes` | **VIDEO_NATIVE** | YES | WITH_CHANGE — no collection identifier exists in the runtime contract | Keyframes and corpus-level CSV/JSON |
| OCR, EasyOCR path | `mvr.enrichment.ocr.scene_text.process_dataset_ocr` | Raw video, then temporary JPEGs | Mapping rows are put in `by_video`; one `process_video_ocr` call receives all rows for a video; it calls `extract_keyframes` once | **VIDEO_NATIVE** | YES | WITH_CHANGE — partition the mapping without splitting one video | One JSON per keyframe |
| OCR, production Docker path | `scripts/ocr/run_ocr_docker.ps1` -> `process_dataset_docker_ocr` | Raw video, then local temporary JPEGs | F-1/F/F+1 requests are grouped in `by_video`; one `extract_keyframes(video_path, v_frames, ...)` call per video; workers consume the temporary frames | **VIDEO_NATIVE** | YES | WITH_CHANGE — no collection field/CLI partition exists | One canonical JSON per keyframe |
| ASR | `mvr.cli build-asr-features` -> `build_asr_features` -> `FfmpegAudioWindowProvider` | Raw video audio window per keyframe | `audio_state` memoizes `has_audio_stream` by `video_path`, but the source loop calls `process_keyframe`, which invokes `decode` for every non-skipped keyframe | **MIXED** (video probe + keyframe-window reads) | YES — existing `videos` filter | WITH_CHANGE — no collection identifier | One JSON per keyframe |
| BEiT-3 | `scripts/run_feature_extraction.py` -> `extract_embedding_artifacts` | JPEG batches | The generic writer groups rows by video only to write one `.npy` shard; each image is still handed to the provider adapter and opened | **KEYFRAME_NATIVE** | YES — safe filtered mapping / per-video shard workflow | WITH_CHANGE | Per-video `.npy`, shared mapping/manifest |
| MetaCLIP-2 | Same | JPEG batches | Same as BEiT-3 | **KEYFRAME_NATIVE** | YES — safe filtered mapping / per-video shard workflow | WITH_CHANGE | Per-video `.npy`, shared mapping/manifest |
| Perception Encoder | Same | JPEG batches | Same as BEiT-3 | **KEYFRAME_NATIVE** | YES — safe filtered mapping / per-video shard workflow | WITH_CHANGE | Per-video `.npy`, shared mapping/manifest |
| Objects | `scripts/run_feature_extraction.py` -> `extract_objects` -> `FasterRCNN.detect` | One JPEG per loop iteration | `extract_objects` loops every mapping row; `detect(image_path)` opens that JPEG | **KEYFRAME_NATIVE** | WITH_CHANGE — caller must provide a filtered mapping | WITH_CHANGE | One JSON per keyframe plus per-shot summary |
| Task05A / provider index build | `scripts/indexing/build_index_documents.py`, `build_provider_indexes.py` | JSON, CSV, manifests, and `.npy` vector shards | Uses `image_path` as metadata; source reads feature artifacts, not raw video or JPEG pixels | **MIXED artifact I/O** | WITH_CHANGE | WITH_CHANGE | Documents and indexes |

### Proven facts required by the task

1. **OCR does not reopen one raw video per keyframe.** Both OCR implementations
   group all rows/samples by `video_id` before extraction. The EasyOCR path is
   `scene_text.py:425-456` then `:291-345`; the Docker path is
   `docker_ocr.py:423-456`. Both invoke the shared
   `ingestion.extract.extract_keyframes`, whose decoder command opens the
   video once and scans sequentially (`extract.py:95-180`). Neighbor samples
   are F-1/F/F+1 frame requests inside that one per-video extraction call.

2. **ASR does reopen the same raw video for each processed keyframe window.**
   `pipeline.py:196-240` loops sources. It memoizes only the audio-stream
   check at `:211-218`; `process_keyframe` calls `audio_provider.decode` at
   `:79-100` for each source. `audio.py:61-92` launches a new FFmpeg command
   with `-ss`, `-t`, and `-i <video>` for every such decode. Resume can remove
   calls for already-valid artifacts, but it does not reuse an open video
   handle for new keyframes.

3. **The three visual providers reread the JPEG independently.**
   The selected-provider runner invokes provider callbacks serially without a
   shared image object cache (`embeddings/offline.py:34-42` and
   `scripts/run_feature_extraction.py:75-145`). Each provider's actual adapter
   loops `image_paths` and calls `PIL.Image.open`: BEiT-3 `beit3.py:105-121`,
   MetaCLIP-2 `metaclip2.py:88-105`, and Perception Encoder
   `perception_encoder.py:78-96`.

4. **Object extraction rereads JPEGs independently of visual extraction.**
   It is called after selected embedding providers finish
   (`scripts/run_feature_extraction.py:155-166`). `extract_objects` loops
   frames and invokes `detector.detect(img_path)`
   (`feature_extraction.py:130-157`); `FasterRCNN.detect` calls
   `Image.open(image_path)` (`faster_rcnn.py:27-34`). No image object or
   decoded tensor is passed from any embedding provider.

The existence of per-video embedding `.npy` output is therefore not evidence
of a single input-image read per video.

## 6. Access call graphs

### Ingestion

```text
run_ingestion.py
  -> run_pipeline
    -> discover_videos / probe_video                    [each video]
    -> detector.detect(video)                           [each video, each selected detector]
    -> make_keyframes
    -> by_video[video_id]
    -> extract_keyframes(video, all selected frames)    [one FFmpeg decode per video]
       -> ffprobe resolution
       -> FFmpeg decoder scan -> selected JPEG encoder
```

### OCR

```text
process_dataset_ocr / process_dataset_docker_ocr
  -> load_keyframes(mapping)
  -> group samples by video_id
  -> extract_keyframes(video, all F-1/F/F+1 samples)    [one decoder invocation per video]
  -> local temporary JPEGs
  -> EasyOCR loop OR DeepSolo -> PARSeq Docker workers
  -> canonical JSON per anchor keyframe
```

### ASR

```text
build_asr_features
  -> load_keyframe_audio_sources
  -> for each keyframe source
       -> cached has_audio_stream(video)                [at most once per Path]
       -> process_keyframe
          -> FFmpeg decode(video, t-3s, t+3s)           [once per keyframe]
          -> VAD -> ChunkFormer -> keyframe JSON
```

### Visual embeddings

```text
run_feature_extraction.py
  -> run_selected_embedding_providers
    -> BEiT-3 extract_embedding_artifacts
         -> group rows by video -> batches -> PIL.Image.open(each JPEG)
    -> MetaCLIP-2 extract_embedding_artifacts
         -> group rows by video -> batches -> PIL.Image.open(each JPEG)
    -> Perception Encoder extract_embedding_artifacts
         -> group rows by video -> batches -> PIL.Image.open(each JPEG)
```

### Objects

```text
run_feature_extraction.py (after visual providers)
  -> extract_objects
    -> for each keyframe row
       -> FasterRCNN.detect(JPEG)
          -> PIL.Image.open(JPEG)
       -> JSON per keyframe
```

## 7. Non-invasive executable call-flow instrumentation

`tests/benchmark/test_r2_access_call_flow.py` is audit-only and makes no
production logging, caching, or output-semantic change.

| Test | Measured repository-level result |
|---|---|
| `test_asr_probes_one_video_once_but_decodes_it_once_per_keyframe` | Two keyframes on one video: 1 `has_audio_stream` call, 2 `decode` calls |
| `test_scene_text_groups_mapping_rows_and_calls_the_video_worker_once_per_video` | Three rows over two videos: worker calls `V1(K1,K2)`, then `V2(K3)` |
| `test_real_visual_adapters_open_the_same_image_independently` | Real BEiT-3, MetaCLIP-2, and PE adapter code each opened the same JPEG once using fake inference runtimes |
| `test_objects_dispatches_one_detector_call_per_keyframe` | Two keyframes: two independent detector-path dispatches |

These static logical-open counts are not converted by assumption. Section 8
reconciles them against the real canary, where one direct acquisition produced
a `HeadObject` plus a `GetObject`.

## 8. R2 canary and authoritative Cloudflare metrics

### 8.1 Deterministic fixture and execution boundary

The canary used one real repository fixture, not Batch1:

| Item | Exact value |
|---|---|
| Source video | Existing `artifacts/real_task04_test/videos/L21_V010.mp4` |
| Derived canary video | First 12 seconds, stream copy, video plus optional first audio stream |
| Canary video | 936,098 bytes; SHA-256 `8dcd05dd54bd971a64d2f59db29ba4746508f9bcbeca3d5e9c2226f1af4e6134` |
| Real keyframe | Frame 125 at 5.0 seconds; 160,490 bytes; SHA-256 `c7a8ed05cbe01bae8eff1063307c6bfbbb80143ee6e19f6016057334d346e229` |
| R2 location | Bucket `aic-test`, unique prefix `data02r/canary-20260822T142016Z` |
| Cleanup | Prefix removed; a post-cleanup prefix list returned zero objects |

`scripts/audit/run_r2_access_canary.py` ran the production video probe and
exact keyframe extractor, OCR F-1/F/F+1 sampling/extraction, ASR audio-stream
probe and six-second FFmpeg decode, and one real Pillow decode for each of
BEiT-3, MetaCLIP-2, Perception Encoder, and Objects. Heavy inference was
**mocked**. Therefore the measurements prove object-access and media-decode
behavior, not model throughput or GPU idle time.

### 8.2 Cloudflare aggregate delta

Cloudflare's adaptive dataset coalesced canary events into earlier time
buckets: narrow per-second queries returned incomplete rows even after the
aggregate had settled. The audit rejects those narrow rows and uses captured
pre-run and settled post-run `r2OperationsAdaptiveGroups` snapshots, both
grouped by `actionType`. The non-secret reconciliation remains in ignored
runtime evidence at
`artifacts/verification/r2_access_audit/cloudflare_aggregate_reconciliation.json`.

| Observed `actionType` | Before | After | Canary delta | Billing treatment |
|---|---:|---:|---:|---|
| `GetObject` | 1 | 12 | **11** | Class B |
| `HeadObject` | 1 | 10 | **9** | Class B |
| `PutObject` | 1 | 3 | **2** | Class A; fixture setup |
| `ListObjects` | 3 | 8 | **5** | Class A; sync/list/cleanup control traffic |
| `ListMultipartUploads` | 0 | 2 | **2** | Class A; AWS CLI control traffic |
| `HeadBucket` | 0 | 5 | **5** | Class B; AWS CLI control traffic |
| `DeleteObject` | 1 | 3 | **2** | Free; cleanup |
| `GetBucketSippyConfiguration` | 0 | 1 | **1** | Observed but not listed in published A/B/free tables |
| `CreateMultipartUpload` | 0 | 0 | **0** | Class A |
| `UploadPart` | 0 | 0 | **0** | Class A |
| `CompleteMultipartUpload` | 0 | 0 | **0** | Class A |
| `CopyObject` | 0 | 0 | **0** | Class A |
| Other multipart actions | 0 | 0 | **0** | None observed |

The classified canary total is **9 Class A** and **25 Class B** operations.
`DeleteObject` is free. The unlisted Sippy-configuration lookup is reported
separately instead of being silently assigned to a billing class.

The two uploaded objects total 1,096,588 bytes and are both below 100 MiB, so
the observed two `PutObject` actions and zero multipart upload actions agree
with Cloudflare's documented small-object upload behavior.

### 8.3 Storage dataset

`r2StorageAdaptiveGroups` was queryable, but both immediate before and
after-upload snapshots returned the same hourly `2026-08-22T14:00:00Z` row:
`objectCount=0`, `uploadCount=0`, `payloadSize=0`, and `metadataSize=0`. The
short-lived objects were removed before a later storage snapshot represented
them. Storage-dataset access is verified, but no storage delta is claimed.
The exact uploaded payload bytes above come from the hashed local fixture.

### 8.4 Direct per-stage read allocation

The aggregate read delta reconciles exactly with the client acquisition plan:

```text
GetObject:  A full sync 2 + B prefetch 2 + C direct 7 = 11
HeadObject: A full sync 0 + B prefetch 2 + C direct 7 = 9
```

Each C stage used the same fresh `aws s3 cp` mechanism once, so its deterministic
allocation is one `HeadObject` plus one `GetObject`. No stage wrote output to R2.

| Direct C stage | `GetObject` | `HeadObject` | Put/List/multipart/copy |
|---|---:|---:|---:|
| Ingestion | 1 | 1 | 0 |
| OCR | 1 | 1 | 0 |
| ASR | 1 | 1 | 0 |
| BEiT-3 | 1 | 1 | 0 |
| MetaCLIP-2 | 1 | 1 | 0 |
| Perception Encoder | 1 | 1 | 0 |
| Objects | 1 | 1 | 0 |

## 9. Request ratios and A/B/C strategy comparison

All three strategies used the same two immutable fixture objects and stage
work. Strategy B is explicit unit prefetch/cache; no rclone mount or WinFsp
layer was available, so this is not a claim about rclone VFS range reads.

| Strategy | Object acquisitions | Reconciled primary object actions | Downloaded bytes | One-run wall-clock | Final local bytes | GPU idle |
|---|---:|---|---:|---:|---:|---|
| A — full pre-sync local | 2 | 2 Get; no HeadObject | 1,096,588 | 5.378s | 1,735,526 | Not measured; inference mocked |
| B — video/keyframe prefetch cache | 2 | 2 Get + 2 Head | 1,096,588 | 5.800s | 1,735,526 | Not measured; inference mocked |
| C — minimal-cache direct object access | 7 | 7 Get + 7 Head | 3,450,254 | 19.134s | 638,938 after source removal | Not measured; inference mocked |

AWS CLI also generated the aggregate `ListObjects`, `ListMultipartUploads`,
`HeadBucket`, and Sippy-configuration actions in section 8. Adaptive timestamps
did not support a defensible per-command allocation, so they are not hidden
inside the strategy rows.

C transferred **3.146x** as many fixture bytes as A/B and used **14 primary
Class B object actions** versus 2 for A and 4 for B. The single run supports
deterministic request/byte ratios, but not dispersion or a compute-latency
claim. A ran first and paid cold OS/process effects; the 0.422-second A/B
difference is not treated as a robust speed ranking.

Reconciliation with `tests/benchmark/test_r2_access_call_flow.py`:

- ASR's one video probe plus per-keyframe decode explains why uncached remote
  scheduling scales poorly with keyframes.
- OCR groups rows per video, so one prefetch serves all F-1/F/F+1 extraction.
- BEiT-3, MetaCLIP-2, PE, and Objects each decode the same JPEG. C made four
  remote acquisitions; B made one shared prefetch and four local decodes.
- One logical open was not one R2 operation: each direct acquisition produced
  both a `HeadObject` and a `GetObject`.

## 10. Full-Batch1 formula — intentionally unpopulated

Let `N_v` be the authoritative Batch1 video count and `N_k` the authoritative
regenerated keyframe count. Let the lower-case terms be ratios measured from
Cloudflare deltas, not client logs. Once available, the request model is:

```text
B_total =
  b_ingest_per_video * N_v
  + b_ocr_per_video * N_v
  + b_asr_per_keyframe * N_k
  + (b_beit3_per_keyframe
     + b_metaclip2_per_keyframe
     + b_pe_per_keyframe
     + b_objects_per_keyframe) * N_k
  + b_task05_and_index

A_total = a_ingestion_writes + a_feature_writes + a_task05_and_index
```

The canary supplies measured acquisition ratios for its AWS-CLI prefetch and
direct mechanisms, but `N_v`/`N_k`, full payload bytes, Task05/index output
operations, and the final production transfer client are unavailable. Thus
the full Batch1 request count and cost are
**NOT YET EXACTLY DETERMINABLE**. The task's rough planning quantities are not
substituted into this formula.

## 11. Exact billing-unit calculation

For the eventual measured monthly totals `S` (Standard GB-month), `A`
(Class A operations), and `B` (Class B operations), aggregate all components
first. Do not reset the account-level free tier for individual stages.

```text
rounded_storage_GB = ceil(S)
billable_storage_GB = max(0, rounded_storage_GB - 10)
storage_cost = billable_storage_GB * 0.015

rounded_A_million = ceil(A / 1,000,000)
billable_A_million = max(0, rounded_A_million - 1)
class_A_cost = billable_A_million * 4.50

rounded_B_million = ceil(B / 1,000,000)
billable_B_million = max(0, rounded_B_million - 10)
class_B_cost = billable_B_million * 0.36

R2_cost = storage_cost + class_A_cost + class_B_cost
```

R2 internet egress is `$0`. Any GPU-provider bandwidth charge is external to
R2 and unmeasured here.

### Required aggregate cost table

| Component | Measured source | Quantity | Billing class | Free tier | Rounded billable quantity | Cost |
|---|---|---:|---|---:|---:|---:|
| Raw video storage, Batch1 | Full raw corpus absent | Not measured | Storage | 10 GB-month shared | Not calculable | Not calculable |
| Regenerated keyframe storage, Batch1 | Full keyframe root/mapping absent | Not measured | Storage | 10 GB-month shared | Not calculable | Included in aggregate storage calculation |
| Ingestion reads | R2 GraphQL direct canary | 1 Get + 1 Head direct; shared video-cache candidate | Class B | 10M shared | Batch quantity unavailable | Not calculable |
| Keyframe writes | R2 GraphQL fixture setup only | 2 fixture PutObject total; production output writes unmeasured | Class A | 1M shared | Batch quantity unavailable | Not calculable |
| OCR operations | R2 GraphQL direct canary | 1 Get + 1 Head direct; zero after video prefetch | Class B | Shared | Batch quantity unavailable | Not calculable |
| ASR operations | R2 GraphQL plus call-flow test | 1 Get + 1 Head for one-keyframe canary; decode remains per keyframe | Class B | Shared | Batch quantity unavailable | Not calculable |
| BEiT-3 operations | R2 GraphQL direct canary | 1 Get + 1 Head direct; zero after shared prefetch | Class B | Shared | Batch quantity unavailable | Not calculable |
| MetaCLIP-2 operations | R2 GraphQL direct canary | Same | Class B | Shared | Batch quantity unavailable | Not calculable |
| Perception Encoder operations | R2 GraphQL direct canary | Same | Class B | Shared | Batch quantity unavailable | Not calculable |
| Object operations | R2 GraphQL direct canary | Same | Class B | Shared | Batch quantity unavailable | Not calculable |

## 12. Storage billing timelines

Let `R`, `K`, and `F` be the actual final raw-video, keyframe, and feature
bytes stored in R2, respectively. Convert bytes to decimal GB before applying
the 30-day average-peak rule. These are exact scenario formulas, not a
full-Batch numerical quote:

| Scenario | Standard GB-month before rounding |
|---|---|
| S1 — all artifacts present for 30 days | `(R + K + F) / 1,000,000,000` |
| S2 — raw present 30 days; keyframes/features produced halfway through | `(R + 0.5 * (K + F)) / 1,000,000,000` |
| S3 — raw present 30 days; staging retained for 24 hours then deleted | `(R + (K_staging + F_staging) / 30) / 1,000,000,000` |

Apply `ceil(S)` only after calculating the appropriate scenario's
GB-month total. The locally measured fixtures above cannot populate these
Batch1 scenarios because neither is the authoritative full corpus.

## 13. Scheduling/cache recommendation

The measured repository recommendation is **HYBRID**:

```text
Ingestion           -> video-level cache
OCR                 -> video-level cache
ASR                 -> video-level cache; consider audio-window cache later
BEiT-3              -> shared keyframe prefetch/cache
MetaCLIP-2          -> shared keyframe prefetch/cache
Perception Encoder  -> shared keyframe prefetch/cache
Objects             -> reuse the same keyframe cache
```

| Stage | Best measured strategy | Evidence |
|---|---|---|
| Ingestion | Video-level cache | Production extraction needs a local path; one cached video preserves its one-video execution unit without reacquisition |
| OCR | Video-level cache | Existing code groups mapping rows and F-1/F/F+1 samples before one extraction call per video |
| ASR | Video-level cache | Existing test proves one stream probe but one FFmpeg decode per keyframe; cache prevents repeated remote acquisition |
| BEiT-3 | Keyframe prefetch/cache | One remote acquisition can feed the provider's independent local decode |
| MetaCLIP-2 | Keyframe prefetch/cache | Same |
| Perception Encoder | Keyframe prefetch/cache | Same |
| Objects | Shared keyframe cache | Reuses the JPEG already prefetched for visual providers |

Full local sync and collection-level cache are not required by this canary.
They may be convenient for a worker assigned a complete collection, but no
measured request advantage over narrow video/keyframe prefetch was shown.
Minimal-cache direct access is worst for this multi-stage fixture because it
repeats transfers and Class B actions.

## 14. Unknowns and required continuation evidence

The credential and dataset-access blockers are resolved. Remaining unknowns:

1. Authoritative regenerated Batch1 video/keyframe counts and exact bytes.
2. Production output writes for keyframes, OCR/ASR JSON, embeddings, objects,
   Task05 documents, and indexes.
3. A real rclone/WinFsp VFS or final production S3 client benchmark; this run
   measured explicit AWS-CLI prefetch and object acquisition.
4. Range-GET behavior for direct FFmpeg-over-HTTP/VFS; the canary downloaded
   complete objects before running unchanged local-path production code.
5. Repeated trials and real model inference for compute/GPU-idle statistics.
6. A retained fixture long enough for `r2StorageAdaptiveGroups` to expose an
   object/payload/metadata delta.

None authorizes substituting `~900` or `~200,000` into an "exact" bill.

## 15. Final verdict

**PASS_WITH_UNMEASURED_FULL_BATCH_VARIABLES**

The verification gate, small real-fixture A/B/C canary, authoritative
Cloudflare aggregate operation delta, call-flow reconciliation, and stage-level
cache recommendation are complete. The supported design is hybrid video-level
plus shared-keyframe caching. No production behavior changed, no credential
was persisted, and the temporary R2 prefix was removed.

The exact full-Batch1 bill remains **NOT YET EXACTLY DETERMINABLE** because the
authoritative regenerated corpus counts/bytes and complete production output
write counts do not exist. No Batch1 estimate is presented as exact.
