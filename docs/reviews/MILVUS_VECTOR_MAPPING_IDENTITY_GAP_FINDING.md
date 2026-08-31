# Finding: `keyframe_id`-only identity in the extraction-status/Milvus-ingest path

Status: **Documented, not fixed**. Discovered while planning the Milvus
`entity_id`/HNSW migration (see `CODEX_TASK_MILVUS_HNSW_AND_ENTITY_ID.md`,
`MILVUS_RUNTIME_AUDIT_AND_DECISION_REPORT.md`). Scoped out of that task —
deliberately not fixed here — because the real fix touches a shared,
actively-used status-tracking schema and its callers across OCR/ASR/embedding
extraction, well beyond the Milvus layer that task authorized.

## The problem

`EXTRACTION_STATUS_SCHEMA` (`src/mvr/operations/parquet_schemas.py:116-128`)
— the schema for `status_parquet`, the file `ingest_to_milvus()` reads to
decide which keyframes are eligible for Milvus ingest — has **no `video_id`
column**, only `keyframe_id` plus per-provider status columns. The entire
`src/mvr/operations/extraction_status.py` API (`initialize_status`,
`update_status`, `find_todo`, `find_failed`) is keyed purely by `keyframe_id`
strings; none of these functions take or track `video_id`.

This matters because `keyframe_id` is **not globally unique** across the
organizer's dataset. `task05a.py:271-280` documents a real, previously
measured instance of this: 166,550 keyframes carry only 632 distinct
`keyframe_id` strings (id "034" appears in 842 different videos), because the
team's own keyframe ids spell the video into the string
(`L21_V001_S0007_K01`) but the organizer's ids do not.

## Confirmed downstream effect (traced, not speculative)

In `ingest_to_milvus()` (`src/mvr/indexing/milvus_ingest.py:308-349`):

```python
for row in eligible_rows:          # row comes from status_parquet — no video_id key
    kf_id = row[FIELD_KEYFRAME_ID]
    ...
    entity = _build_entity_row(kf_id, row, provider_vecs)
```

`_build_entity_row()` reads `meta_row.get(FIELD_VIDEO_ID, "")` — since `row`
(sourced from `status_parquet`) never has this key, **every entity currently
written to Milvus via the `milvus ingest` CLI gets `video_id=""`**.

Separately, `_load_mapping()` (`milvus_ingest.py:122-130`) builds each
provider's `keyframe_id → row_idx` lookup as a flat dict with no `video_id`
component, loaded once and reused across the *entire* `status_parquet` (which
is not scoped to a single video). Given the same non-uniqueness as above,
this dict silently collapses rows from different videos onto the same key —
the last video's `row_idx` wins, so a different video's embedding vector can
get attached to a keyframe under a colliding id.

## Why the Milvus `entity_id` fix (this PR) does not fully solve this

The Milvus migration in this branch changes the primary key from
`keyframe_id` to `entity_id = video_id + "/" + keyframe_id`. But since
`ingest_to_milvus()` currently always resolves `video_id` to `""`, in
practice this collapses back to `entity_id = "/" + keyframe_id` for every
row — still colliding across videos, just with an extra leading slash. The
`entity_id` primary-key fix is real and correct for any caller that supplies
a proper `video_id` (e.g. `MilvusProviderBackend.insert_batch()` called
directly with correct metadata), but the production `milvus ingest` CLI path
will not benefit from it until this identity gap is closed upstream.

## What a real fix would require (not done here)

1. Add a `video_id` column to `EXTRACTION_STATUS_SCHEMA` and thread it
   through `initialize_status`/`update_status`/`find_todo`/`find_failed`.
2. Update every caller of `extraction_status.py` across the OCR/ASR/
   embedding extraction pipelines (not enumerated here — requires its own
   scoped investigation) to pass `video_id` alongside `keyframe_id`.
3. Only then can `ingest_to_milvus()` safely resolve a correct `video_id`
   per row without an ambiguous `keyframe_id`-only join against
   `keyframe_mapping_csv` (which has the same non-uniqueness problem if
   joined on `keyframe_id` alone).

This is materially larger than the Milvus task's stated scope ("không audit
lại toàn bộ repository") and should be planned as its own task.
