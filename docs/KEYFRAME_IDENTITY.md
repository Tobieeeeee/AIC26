# Keyframe identity

**A keyframe is identified by `(video_id, keyframe_id)`. Never by
`keyframe_id` alone.**

That is the whole rule. The rest of this page is why it changed, and what
went wrong while it was untrue.

## What changed

The keyframe ids this repository generates spell the video into the id:

```python
f"{video_id}_S{shot_id:04d}_K{ordinal:02d}"     # L21_V001_S0007_K01
```

They are unique across the corpus, so keying on `keyframe_id` alone was
correct — everywhere, from the first commit.

The organizer's keyframe ids are the JPEG stem: `034`. That is a **position
within a video**, not a name. Measured on the real mapping:

```
166,550 keyframes  ->  632 distinct ids     (each id reused ~263 times)
"034" appears in 842 videos
within any one video, no id repeats          (0 cases)
```

So `L21_V001/034` and `L21_V002/034` are two different frames of two
different broadcasts that happen to share a name. Adopting the organizer's
keyframes as canonical identity — which the team did, because they already
decoded 873 videos and shipped CLIP features for every keyframe — removed
the uniqueness that ~20 call sites depended on, and nobody declared it.

## How it failed

Two shapes, and only one of them is loud.

### It refuses to start

Anything that treated a repeated id as a data error rejected the corpus at
the *second video*:

```
Task05AValidationError: keyframe_id is not globally unique: 034
Task05BValidationError: duplicate Task05A keyframe IDs: ['001']
ProviderIndexError: duplicate processed keyframe_id: 034
OfflineEmbeddingError: duplicate keyframe_id: 034
ERROR: Duplicate keyframe_id in mapping: 001
```

`task05a._canonical_lookup` was the one that mattered: it made
`build_index_documents` unable to emit a single document, and the lexical
index, the provider indexes and the OCR documents all read from those.

### It quietly drops data

Worse, because nothing reports it:

- **`_id = keyframe_id`** would have had Elasticsearch accept 166,550
  documents and keep **632**, each overwritten ~263 times. The audit could
  not catch it either — `expected_ids` was built from the same id, so 632
  expected matched 632 actual and both were wrong.
- **`provider_semantic` / `semantic_rrf`** deduplicated search hits on the
  id, so two different frames became one and the loss was recorded as a
  `duplicate:` warning. A top-100 could return at most 632 rows.
- **`scene_text` shot boundaries** were keyed on a bare `shot_id`, and every
  video numbers its shots from 1, so one video's ranges clipped all the
  others. Measured: a keyframe of `L21_V002` at frame 5050 kept 0 of 3
  samples against `L21_V001`'s `(0, 99)`. With 873 videos, one wins and 872
  are OCR'd to nothing.
- **`docker_ocr` anchors** resolved to whichever video was read last, so OCR
  requests carried another video's `anchor_frame_idx` — the number the
  competition scores on — and canonical output was written to one
  `<wrong video>/034.json` for all of them.

## Why the tests did not catch it

Every fixture builds one video. `test_task05b._document` hardcodes
`video_id="L21_V010"`; the multi-embedding fixtures use `"V1"`. A collision
had nowhere to happen.

Three fixtures went further and encoded the bug as intent: they duplicated a
`keyframe_id` while leaving `video_id` different — the organizer's normal
case — and asserted the result was rejected. One test was even named
`test_bulk_action_uses_keyframe_id_as_elasticsearch_id`.

The tests added here all use two videos that share an id.

## Writing new code

- Dict keys, sets, dedup: `(video_id, keyframe_id)`.
- Elasticsearch `_id`: `f"{video_id}/{keyframe_id}"`, which reads as the
  path the document came from.
- Sorting: include both, or the order is not total and not reproducible.
- Shot boundaries: `(video_id, shot_id)` — shot numbers restart per video.
- Output paths: `<root>/<video_id>/<keyframe_id>.json` (already the layout).
- `mvr.indexing.validation.identity_key(video_id, keyframe_id)` returns the
  tuple if you want it named.

A duplicate check should still fire for a genuine duplicate — same video,
same id. Widening the key must not remove the check, only correct it.
