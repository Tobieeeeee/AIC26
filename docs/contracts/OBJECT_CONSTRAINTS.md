# Object Constraint Contract

## Status

**Object-01 specification only.**

This document freezes provider-independent object-constraint semantics for a
future unified `POST /search` implementation. It does not change the offline
detector, Task05 documents, Elasticsearch mappings, specialized object route,
frontend, hybrid fusion, or Task06A implementation.

The evidence-backed v1 subset is deliberately narrow:

```text
canonical COCO label presence/minimum count
+ same-keyframe AND
+ object-only hard filtering (existing specialized capability)
```

The repository does not currently provide online object colors, geometric
relations, per-request detector thresholds, object relevance scoring, or an
implemented typed OR/max-count evaluator.

## 1. Ownership boundary

```text
Query Processing
  -> may extract non-authoritative object hints from text
  -> may normalize language/synonyms before API validation
  -> does not execute object matching

Frontend
  -> sends typed user-selected canonical constraints
  -> does not send detector/model/class IDs
  -> does not infer support from visible controls

POST /search
  -> validates the unified envelope
  -> delegates the object block
  -> maps typed failures to the Task06A error envelope

ObjectConstraintValidator
  -> validates labels, counts, duplicate rules, logic, and maturity

ObjectConstraintEvaluator
  -> evaluates rules against one canonical keyframe's existing count evidence
  -> returns eligibility and optional match evidence
  -> does not calculate hybrid scores

ObjectRetriever
  -> generates eligible object-only candidates where an object index exists

HybridSearchService
  -> will apply mandatory object eligibility without changing Boolean semantics
     once the reserved Task06A orchestration gate is implemented
  -> owns fusion/ranking of retrieval branches

Offline detector / Task05
  -> produces and projects object evidence
  -> is never invoked during query execution
```

The FastAPI route must remain thin. It must not parse legacy free text, build
Elasticsearch filters, inspect detector output files, or perform fusion
mathematics inside the handler.

## 2. Current evidence schema

### 2.1 Task04A per-keyframe artifact

The inspected Faster R-CNN artifact has:

```json
{
  "keyframe_id": "L21_V010_S0007_K01",
  "video_id": "L21_V010",
  "shot_id": 7,
  "frame_idx": 578,
  "timestamp": 23.12,
  "image_path": "keyframes/L21_V010/L21_V010_S0007_K01.jpg",
  "person_count": 6,
  "car_count": 0,
  "objects": [
    {
      "label": "person",
      "class_id": 1,
      "score": 0.9977,
      "bbox": [151.58, 97.43, 514.39, 557.32]
    }
  ]
}
```

The accompanying manifest records Faster R-CNN, retained-score threshold
`0.5`, built-in class-aware NMS at `0.5`, and `xyxy` pixel coordinates.

| Field | Classification | Object-constraint role |
|---|---|---|
| `keyframe_id` | `CANONICAL` | Internal join/result identity |
| `video_id` | `CANONICAL` | Cross-validates identity |
| `frame_idx` | `CANONICAL` | Becomes competition-facing `frame_id` after canonical mapping |
| `shot_id`, `timestamp` | `CANONICAL/OPTIONAL` | Navigation metadata; not an object rule |
| `image_path` | `OPTIONAL` | Media reference only |
| detection `label` | `CANONICAL_EVIDENCE` | Source label counted by Task05 |
| detection `score` | `OPTIONAL_SOURCE_EVIDENCE` | Already subject to offline threshold; not kept by Task05 |
| detection `bbox` | `OPTIONAL_SOURCE_EVIDENCE` | Not kept by Task05 online representation |
| detection `class_id` | `DERIVED_PROVIDER_FIELD` | Detector-specific; never a public API field |
| `person_count`, `car_count` | `DERIVED_AUTHORITATIVE_CONVENIENCE` | Override projected counts for those two labels |
| object color | `NOT_AVAILABLE` | Cannot evaluate color constraints |
| geometric relation | `NOT_AVAILABLE_ONLINE` | Raw boxes exist, but no Task05/online relation representation exists |

### 2.2 Task05 processed representation

Task05 projects object evidence without a second threshold or NMS pass:

```json
{
  "objects": {
    "labels": ["book", "bowl", "chair", "knife", "person", "toothbrush"],
    "counts": {
      "book": 1,
      "bowl": 1,
      "chair": 9,
      "knife": 1,
      "person": 6,
      "toothbrush": 1
    }
  },
  "indexability": {
    "object": true
  }
}
```

This is the canonical online evidence boundary:

- `objects.counts[label]` is the retained detection count for the keyframe;
- `objects.labels` is the sorted set of labels with positive counts;
- a valid empty detection artifact becomes `labels: []`, `counts: {}` and
  remains object-indexable;
- a missing, invalid, duplicate, or identity-mismatched artifact leaves
  `indexability.object = false`;
- confidence, bounding boxes, class IDs, colors, and relations are not
  preserved.

No positional join is allowed. The object document must match canonical
`keyframe_id`, `video_id`, `frame_idx`, `shot_id`, and timestamp checks.

## 3. Taxonomy

The authoritative v1 request taxonomy is the existing
`mvr.retrieval.object_search.COCO_CLASSES` set. It contains the current 80
COCO labels:

```text
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush
```

Validation rules:

- trim surrounding whitespace;
- require exact canonical lower-case spelling after trimming;
- preserve multiword labels such as `traffic light` and `cell phone`;
- do not singularize/pluralize;
- do not case-fold request values;
- do not silently map aliases or synonyms;
- reject empty and unknown labels;
- reject repeated rules for the same canonical label.

Examples of unsupported aliases include `man`, `woman`, `baby`, `phone`,
`traffic`, `coffee`, `glasses`, `flower`, and `tree`. A Query Processing
component may propose `man -> person` or `phone -> cell phone`, but the
canonical API receives only the validated canonical label. This task does not
define or implement a synonym table.

## 4. Canonical typed block

The future normalized shape is:

```json
{
  "object_constraints": {
    "logic": "AND",
    "objects": [
      {
        "label": "person",
        "min_count": 1
      },
      {
        "label": "motorcycle",
        "min_count": 1
      }
    ]
  }
}
```

### 4.1 Fields

| Field | Type | Required | Maturity | Meaning |
|---|---|---:|---|---|
| `logic` | `"AND" \| "OR"` | yes | AND `SUPPORTED_V1`; OR `RESERVED` | Combines rule truth values for one keyframe |
| `objects` | non-empty `ObjectRule[]` | yes | `SUPPORTED_V1` subset | Typed rules; array order does not affect Boolean truth |
| `ObjectRule.label` | canonical COCO string | yes | `SUPPORTED_V1` | Exact taxonomy label |
| `ObjectRule.min_count` | integer `>= 1` | yes | `SUPPORTED_V1` | Minimum retained count for that label |
| `ObjectRule.max_count` | integer `>= 0` or null | no | `RESERVED` | Maximum retained count for that label |
| `total_max_objects` | integer `>= 0` or null | no | `RESERVED` | Maximum sum of all retained label counts |
| `colors` | any | no | `RESERVED` | Rejected until object-color evidence/taxonomy exists |
| confidence threshold | any | no | `OUT_OF_SCOPE` | Offline threshold is artifact provenance, not a query control |
| spatial relationships | any | no | `OUT_OF_SCOPE` | Requires a separate relation/evidence contract |

`max_objects_text` is not canonical. Neither is the Task06A-era ambiguous
single `max_objects` field.

The supported v1 executable subset is therefore:

```text
logic = AND
one or more unique canonical labels
min_count >= 1 for every rule
no max_count
no total_max_objects
no colors or relations
```

Requests using a field with `RESERVED` maturity receive a structured
unsupported-attribute error until the evaluator is implemented.

## 5. Rule evaluation

For a valid object-evidence record, define:

```text
count(label) = objects.counts[label] if present, otherwise 0
```

The absent-key-as-zero rule applies only to a **valid object artifact/document**.
It must not be used for missing or invalid evidence.

Conceptually, a bounded rule matches when:

```text
count(label) >= min_count
AND (max_count is null OR count(label) <= max_count)
```

`max_count` is defined here for future compatibility but remains `RESERVED`.

### 5.1 AND

Every rule must match the same canonical keyframe:

```text
count(person) >= 1
AND count(motorcycle) >= 1
```

Objects in different keyframes, neighboring frames, shots, or temporal events
cannot satisfy one AND block.

AND is `SUPPORTED_V1` because the current specialized object search emits all
rules as Elasticsearch `bool.filter` clauses.

### 5.2 OR

At least one rule must match the same canonical keyframe:

```text
count(car) >= 1
OR count(motorcycle) >= 1
```

Rule order has no effect on truth. Duplicate labels are invalid rather than
being merged.

OR semantics are frozen, but execution is `RESERVED`: the current object
service has no `bool.should`/minimum-should-match path and no typed evaluator.

## 6. Empty/null semantics

| Input | Meaning |
|---|---|
| missing `object_constraints` | No user-mandatory object filter |
| `object_constraints: null` | No user-mandatory object filter |
| block with `objects: []` | `EMPTY_OBJECT_CONSTRAINT` validation error |
| block missing `objects` | Schema validation error |
| `retrieval_mode: "object"` with missing/null block | `OBJECT_CONSTRAINTS_REQUIRED` |

An empty block is not silently converted to null. This prevents a client bug
from silently broadening an intended object search.

## 7. Object-only search

The specialized `POST /search/object` capability is `SUPPORTED_V1`: it
requires only `want` and `top_k` and does not require text. The unified
`POST /search` object mode is contract-ready but not implemented.

The future unified request may therefore use:

```json
{
  "task_mode": "kis",
  "query": "",
  "retrieval_mode": "object",
  "top_k": 20,
  "object_constraints": {
    "logic": "AND",
    "objects": [
      {"label": "person", "min_count": 1}
    ]
  }
}
```

This is a required future amendment to Task06A's blanket non-blank text rule:
an empty text query is valid only for `retrieval_mode: "object"` with a valid,
non-empty object constraint. Semantic, lexical, and current hybrid execution
still require non-blank text unless another contract explicitly changes them.

## 8. Hard filter, retrieval, and ranking

### Current behavior

- `/search/object`: object-only candidate generation through hard
  Elasticsearch Boolean filters;
- `/search/text`: optional `person_min` and `car_min` are hard filters on a
  lexical query;
- `/search/hybrid`: combines lexical and semantic results only; it does not
  consume object constraints;
- no current component performs object-based reranking;
- no current component calculates a meaningful object relevance score.

### Canonical ownership

User-supplied `object_constraints` are mandatory eligibility constraints:

- object-only mode filters the object-indexable corpus;
- a future hybrid implementation must apply the same Boolean eligibility
  without embedding object math in fusion; this is reserved/pending the
  Task06A orchestration implementation;
- presence of the block means mandatory; absence/null means no mandatory
  object gate;
- query-extracted object hints are not automatically promoted to mandatory
  filters.

For object-only Boolean results:

- primary ordering: `keyframe_id` ascending;
- no object relevance score is invented;
- score representation is not frozen here: current Task06A requires a finite
  `score`, so the parent-contract conflict between a nullable score and a
  separate eligibility/relevance field remains an open question.

The current specialized Elasticsearch route does not send an explicit sort and
does not guarantee this tie order. That is a compatibility gap, not evidence
for an object score.

For hybrid, the mandatory object gate is not an existing supported capability:

- applying the gate is `RESERVED`/pending Task06A implementation;
- once implemented, it should preserve the upstream/fused order among eligible
  hits and filter before final `top_k` truncation, with backend overfetch as
  needed;
- Task06A's equal-score `keyframe_id` tie-break remains applicable;
- object counts may be returned as match evidence but are not fusion scores.

Object-count-based ranking or reranking is `DEFERRED`.

## 9. Missing and invalid evidence

The evaluator must distinguish:

| Evidence state | Meaning |
|---|---|
| valid non-empty counts | Evaluates normally |
| valid empty counts | Every positive `min_count >= 1` rule is false |
| missing artifact/document | Unknown/no evidence; never interpreted as zero |
| invalid/identity-mismatched artifact | Data-integrity failure; never repaired or interpreted as empty |

Policy:

- object-only mandatory object filter: only a definitive match passes; missing
  evidence does not satisfy a positive rule;
- object-only mode with isolated missing evidence may omit that candidate;
- known corrupt/invalid indexed evidence is `OBJECT_ARTIFACT_INVALID`, not a
  no-hit success;
- a globally unavailable object service/index is `503`, not an empty result;
- hybrid without `object_constraints` must not discard semantic/lexical hits
  merely because object evidence is missing;
- hybrid with a supplied block is intended to treat it as mandatory and filter
  unknown evidence out, but that gate is reserved/pending Task06A
  implementation.

## 10. Color and spatial semantics

### Colors

Colors are `RESERVED`.

The inspected object artifacts have label, class ID, confidence, and bounding
box fields, but no object-color association. Task05 retains only label counts.
The frontend's uppercase color chips and query-decomposition `attributes`
strings are not object-color evidence and do not establish a taxonomy.

A non-empty color constraint must return:

```text
UNSUPPORTED_OBJECT_ATTRIBUTE
```

No color taxonomy, casing rule, or label-to-color association is frozen here.

### Geometric relations

Geometric object relations are `OUT_OF_SCOPE` for v1.

Raw Task04 detections have `xyxy` boxes, but Task05 discards them and no online
object service evaluates `left-of`, `right-of`, `inside`, `overlap`, `near`,
`above`, or `below`. The frontend name `spatial_filter` and experimental query
decomposition `spatial_relations` strings do not provide execution support.

A future relation feature requires its own typed entities, coordinate
normalization, relation definitions, evidence availability, and temporal
scope. It must not overload this count contract.

## 11. Canonical identity

Every eligible candidate must resolve through canonical mapping to:

```text
keyframe_id
video_id
frame_id
shot_id (nullable)
timestamp
```

Rules:

- `keyframe_id` is the object evidence/join/result identity;
- `video_id + frame_id` is competition-facing identity;
- `frame_id` is the original-video `frame_idx`;
- `shot_id` and timestamp are navigation metadata;
- object artifacts are never joined by file position, vector row, or list
  order;
- conflicting identity is an error, not a repair opportunity.

The legacy Lab 6 index calls the canonical keyframe string `frame_id`.
Unified normalization must not expose that string as Task06A's integer
competition `frame_id`.

## 12. Structured errors

Errors use the Task06A envelope.

| Code | Intended HTTP | Meaning |
|---|---:|---|
| `INVALID_OBJECT_CONSTRAINT` | `422` | Malformed typed block/rule |
| `EMPTY_OBJECT_CONSTRAINT` | `400` | Explicit block contains no rules |
| `UNKNOWN_OBJECT_LABEL` | `422` | Label is not an exact canonical COCO label |
| `INVALID_OBJECT_COUNT` | `422` | Boolean, non-integer, negative, invalid range, or unsupported zero minimum |
| `DUPLICATE_OBJECT_RULE` | `422` | Same canonical label appears more than once |
| `UNSUPPORTED_OBJECT_LOGIC` | `400` | Reserved OR execution requested |
| `UNSUPPORTED_OBJECT_ATTRIBUTE` | `400` | Color, max count, total maximum, confidence, or relation not executable |
| `OBJECT_CONSTRAINTS_REQUIRED` | `400` | Object mode has no non-empty constraint |
| `OBJECT_PROVIDER_UNAVAILABLE` | `503` | Object evaluation service unavailable |
| `OBJECT_INDEX_UNAVAILABLE` | `503` | Required object index unavailable |
| `OBJECT_ARTIFACT_INVALID` | `500` | Persisted evidence violates identity/schema invariants |

Unknown labels and unsupported attributes are never silently dropped. A
well-formed search with no matching keyframe remains HTTP `200` with an empty
result array.

## 13. Feature maturity

| Feature | Status | Evidence/decision |
|---|---|---|
| Canonical label presence | `SUPPORTED_V1` | COCO validation + count fields |
| Same-keyframe AND | `SUPPORTED_V1` | Current `bool.filter` behavior |
| Same-keyframe OR | `RESERVED` | Semantics frozen; no current execution path |
| Per-label `min_count >= 1` | `SUPPORTED_V1` | Current Elasticsearch `gte` filters |
| Per-label `max_count` | `RESERVED` | Task05 counts can evaluate; current service cannot |
| `total_max_objects` | `RESERVED` | Task05 sum is available; current service cannot |
| Specialized object-only hard filtering | `SUPPORTED_V1` | Existing `POST /search/object` applies Boolean filters without text |
| Unified `POST /search` object mode | `CONTRACT_READY_NOT_IMPLEMENTED` | Typed contract is defined; unified route is not implemented |
| Hybrid mandatory object gate | `RESERVED` | Pending Task06A orchestration implementation |
| Object relevance ranking | `DEFERRED` | No evidence-backed object score |
| Object reranking | `DEFERRED` | No current implementation |
| Colors | `RESERVED` | UI only; no object-color evidence/taxonomy |
| Geometric spatial relations | `OUT_OF_SCOPE` | Boxes discarded before online representation |
| Per-request confidence threshold | `OUT_OF_SCOPE` | Offline detector provenance only |
| Query-extracted object hints | `DEFERRED` | Experimental decomposition only; not production constraints |

## 14. Examples

### 14.1 One object

```json
{
  "logic": "AND",
  "objects": [
    {"label": "person", "min_count": 2}
  ]
}
```

Matches a valid object record with `count(person) >= 2`.

### 14.2 AND

```json
{
  "logic": "AND",
  "objects": [
    {"label": "person", "min_count": 1},
    {"label": "motorcycle", "min_count": 1}
  ]
}
```

Both rules must match the same keyframe.

### 14.3 OR (reserved execution)

```json
{
  "logic": "OR",
  "objects": [
    {"label": "car", "min_count": 1},
    {"label": "motorcycle", "min_count": 1}
  ]
}
```

The Boolean meaning is “at least one rule matches the same keyframe,” but
current v1 execution returns `UNSUPPORTED_OBJECT_LOGIC`.

### 14.4 Unknown label

```json
{
  "logic": "AND",
  "objects": [
    {"label": "dragon", "min_count": 1}
  ]
}
```

Result: `UNKNOWN_OBJECT_LABEL` (`422`).

### 14.5 Invalid count

```json
{
  "logic": "AND",
  "objects": [
    {"label": "person", "min_count": -1}
  ]
}
```

Result: `INVALID_OBJECT_COUNT` (`422`).

### 14.6 Empty constraint

```json
{
  "logic": "AND",
  "objects": []
}
```

Result: `EMPTY_OBJECT_CONSTRAINT` (`400`).

### 14.7 Missing evidence

For a canonical keyframe with no valid object artifact:

```text
person min_count=1 -> UNKNOWN, does not pass a mandatory hard filter
```

It is not interpreted as `person = 0`.

### 14.8 Object-only

```json
{
  "task_mode": "kis",
  "query": "",
  "retrieval_mode": "object",
  "top_k": 20,
  "object_constraints": {
    "logic": "AND",
    "objects": [
      {"label": "person", "min_count": 1}
    ]
  }
}
```

The specialized object-only capability exists today, while the unified
`POST /search` object mode is contract-ready but not implemented. This example
therefore requires the proposed Task06A query-validation amendment before it
can execute through the unified route.

### 14.9 Unsupported color

```json
{
  "logic": "AND",
  "objects": [
    {"label": "car", "min_count": 1}
  ],
  "colors": ["red"]
}
```

Result: `UNSUPPORTED_OBJECT_ATTRIBUTE` (`400`).

## 15. Explicit non-goals

This contract does not:

- rerun or change object detection;
- add a taxonomy or synonym table;
- build/rebuild Task05 or Elasticsearch artifacts;
- implement typed validators/evaluators/routes;
- implement OR, maximum counts, colors, or relations;
- add object scoring/reranking;
- modify frontend controls;
- modify Task06A or hybrid fusion;
- change multi-embedding or Query Processing work.
