# Object-01 Object Constraint Specification Report

## Final verdict

**PASS_WITH_OPEN_QUESTIONS**

The current object evidence, Task05 projection, specialized backend route,
frontend fields, tests, and documentation were audited without changing
production code or rerunning detection. The contract freezes the supported
same-keyframe AND/minimum-count subset, typed canonical identities,
specialized object-only hard filtering, missing-evidence policy, deterministic
future ordering, structured errors, and explicit maturity for unsupported or
pending features. Hybrid mandatory object gating remains reserved pending
Task06A orchestration work.

Colors are `RESERVED`; geometric spatial relations are `OUT_OF_SCOPE`.
Task06A requires future amendments, but its files were not edited.

## Branch and HEAD

- Branch: `docs/object01-object-constraint-contract`
- Integration base: `integration/post-multimodel`
- HEAD/base commit: `cc50d24`
- Base commit subject: `Merge remote-tracking branch 'origin/feat/rebuild-embedding-models' into integration/post-multimodel`
- The base contains merged multi-model work, Task06A/06B specifications, and
  Data-01.
- No merge and no push were performed.

## Files changed

Documentation/specification only:

- `docs/contracts/OBJECT_CONSTRAINTS.md`
- `docs/reviews/OBJECT01_SPEC_REPORT.md`

No frontend/backend production code, detector, Task05, Query Processing,
multi-embedding, hybrid fusion, test, config, dependency, dataset, or artifact
file was changed.

## Sources inspected

Required guidance and prior contracts:

- `AGENTS.md`
- `agent_task/object1/OBJECT01_OBJECT_CONSTRAINT_SPEC.md`
- `docs/contracts/UNIFIED_SEARCH_API.md`
- `docs/reviews/TASK06A_SPEC_REPORT.md`
- `docs/contracts/SUBMISSION_API.md`
- `docs/reviews/TASK06B_SPEC_REPORT.md`
- `docs/reviews/DATA_READINESS_AUDIT_REPORT.md`

Object extraction/evidence:

- `src/mvr/enrichment/objects/faster_rcnn.py`
- `src/mvr/enrichment/objects/yolo.py`
- `src/mvr/enrichment/feature_extraction.py`
- actual fasttest `features/objects/manifest.json`
- actual fasttest per-keyframe object JSON

Task05/index representation:

- `src/mvr/indexing/filtering.py`
- `src/mvr/indexing/task05a.py`
- `src/mvr/indexing/validation.py`
- `src/mvr/indexing/documents.py`
- `src/mvr/indexing/elasticsearch_index.py`
- actual fasttest Task05 processed document JSON

Backend object path:

- `src/mvr/retrieval/object_search.py`
- `src/mvr/retrieval/lexical.py`
- `src/mvr/api/app.py`
- `src/mvr/api/schemas.py`

Frontend:

- `frontend/src/api/searchApi.ts`
- `frontend/src/types/competition.ts`
- `frontend/src/stores/useSearchStore.ts`
- `frontend/src/components/Search/SpatialPanel.tsx`
- `frontend/src/components/Search/QueryBar.tsx`
- `frontend/src/components/Search/SearchWorkspace.tsx`
- current frontend review/hardening/test reports

Tests and supporting documentation:

- `tests/unit/retrieval/test_object_search.py`
- `tests/unit/api/test_app.py`
- `tests/integration/test_lab6_runtime.py`
- `tests/unit/enrichment/test_feature_extraction.py`
- `tests/unit/enrichment/objects/test_yolo.py`
- `tests/unit/indexing/test_task05a.py`
- `tests/unit/indexing/test_elasticsearch_index.py`
- `docs/ARCHITECTURE.md`
- `docs/LAB_REUSE_MAP.md`
- `docs/CODEX_PROGRESS.md`
- `docs/AIC_KIS_IMPLEMENTATION_NOTES.md`
- `docs/SOURCE_FILE_ANALYSIS.md`
- experimental query-decomposition models/tests, used only to classify
  extracted object/attribute/relation hints as non-production

## Current object artifact schema

The current Task04A/Faster R-CNN branch stores one JSON object per canonical
keyframe with:

- canonical identity: `keyframe_id`, `video_id`, `shot_id`, `frame_idx`,
  timestamp;
- media reference: `image_path`;
- retained detections: `label`, detector `class_id`, confidence `score`, and
  `xyxy` bounding box;
- derived explicit `person_count` and `car_count`.

The manifest records threshold/NMS/model provenance. It contains no color
attribute or precomputed geometric relation.

Task05 is the online-ready evidence boundary. It cross-validates canonical
identity and projects:

```json
{
  "objects": {
    "labels": ["car", "chair", "person"],
    "counts": {
      "car": 1,
      "chair": 1,
      "person": 2
    }
  },
  "indexability": {
    "object": true
  }
}
```

Task05 does not rerun thresholding/NMS and does not retain score, bounding box,
class ID, color, or relations. A valid empty artifact remains object-indexable
with empty labels/counts. Invalid or mismatched evidence makes the object
branch unavailable for that canonical document.

## Current online behavior

The specialized `POST /search/object` route:

- exists only when Elasticsearch was configured, reachable at application
  startup, and the configured legacy multimodal index exists;
- accepts `ObjectSearchRequest { want: string, top_k: 1..100 }`;
- parses comma-separated exact `COCO-class:non-negative-count` conditions;
- validates against the existing 80-label `COCO_CLASSES` set;
- rejects unknown, empty, malformed, or duplicate labels;
- emits every condition as an Elasticsearch `range gte` clause inside one
  `bool.filter` array;
- therefore implements same-keyframe AND and per-label minimum counts only;
- requires no text query, so the specialized object-only capability is real;
- returns a route-specific `SearchResponse`, not the Task06A envelope;
- has no explicit sort and no evidence-backed object ranking score;
- returns parser failures as string-detail HTTP `400` and runtime failures as
  HTTP `503`.

The current lexical route separately supports `person_min` and `car_min` as
hard filters. The current hybrid route fuses lexical and semantic branches and
does not consume object constraints. No object reranker exists.

The legacy Elasticsearch mapping explicitly declares `person` and `car` below
`objects`; other COCO labels rely on dynamically indexed object fields.
Task05B's lexical-only index does not contain object fields. A future unified
object service therefore needs an explicit Task05 object index/evaluator
ownership decision.

## Canonical contract decision

The contract replaces flat/free-text fields with:

```json
{
  "object_constraints": {
    "logic": "AND",
    "objects": [
      {"label": "person", "min_count": 1},
      {"label": "motorcycle", "min_count": 1}
    ]
  }
}
```

Supported executable v1 subset:

- exact canonical COCO labels;
- one or more unique rules;
- required integer `min_count >= 1`;
- same-keyframe `AND`;
- object-only candidate generation through existing hard filtering;
- hybrid mandatory object gating is not included in the supported subset and
  remains reserved/pending Task06A implementation.

Frozen but reserved:

- same-keyframe `OR`;
- per-label `max_count`;
- `total_max_objects`;
- colors.

Out of scope:

- per-request confidence thresholds;
- geometric/spatial relationships.

`max_objects_text` and ambiguous `max_objects` are not canonical.

## Boolean and count semantics

For a valid object document:

```text
count(label) = stored count, or 0 when the label key is absent
```

This absent-key rule never applies to a missing/invalid object document.

- AND: every rule matches the same canonical keyframe.
- OR: at least one rule matches the same canonical keyframe; semantics frozen,
  execution reserved.
- `min_count`: per-label lower bound, supported for values `>= 1`.
- `max_count`: per-label upper bound, reserved.
- `total_max_objects`: upper bound on the sum of all retained counts, reserved.
- duplicate label rules: validation error, never merged.

The legacy parser accepts `label:0`, but an Elasticsearch range on a missing
dynamic field does not express an unconstrained minimum reliably. The typed
supported subset therefore starts at `min_count = 1`; zero-count/negative
constraints require future `max_count` semantics and valid empty evidence.

## Empty and object-only decisions

- missing/null block: no mandatory object constraint;
- explicit `objects: []`: `EMPTY_OBJECT_CONSTRAINT`;
- object retrieval mode with missing/null block:
  `OBJECT_CONSTRAINTS_REQUIRED`;
- empty text query is valid for object-only mode when a non-empty supported
  object block is present.

The last rule differs from Task06A's blanket non-blank textual-query rule and
is recorded as a proposed amendment. The specialized `/search/object`
capability exists today; unified `POST /search` object mode is contract-ready
but not implemented.

## Filter/rerank and ordering decision

Object constraints define Boolean eligibility, not relevance score:

- object-only search filters the object-indexable corpus;
- supplied constraints in hybrid are intended to be mandatory gates, but that
  gate is reserved/pending Task06A implementation;
- no supplied block means missing object evidence must not delete
  semantic/lexical hits;
- query-extracted hints are not silently promoted into mandatory constraints;
- hybrid fusion/ranking remains owned by HybridSearchService;
- object count does not become a fusion score.

Future object-only deterministic ordering:

```text
keyframe_id ASC
```

The current filter-only Elasticsearch route has no explicit deterministic
sort. The contract does not invent an object score. Task06A currently requires
a finite `score`, so the parent-contract conflict between a nullable score and
a separate eligibility/relevance field remains an open question; this
specification does not freeze `score: null`.

For hybrid:

- the mandatory object gate is reserved/pending Task06A implementation;
- once implemented, hard filtering should preserve the upstream/fused order
  among eligible hits and occur before final `top_k` truncation, with backend
  overfetch as needed;
- Task06A's equal-score `keyframe_id` tie-break remains applicable;
- object counts may be returned as match evidence but are not fusion scores.

## Missing artifact policy

The contract distinguishes:

1. valid counts, including a valid empty count map;
2. missing object evidence;
3. invalid/identity-mismatched evidence.

A valid empty record definitively fails positive minimum-count rules. Missing
evidence is unknown and cannot satisfy a mandatory filter; it is never treated
as count zero. Known corrupt evidence is `OBJECT_ARTIFACT_INVALID`. Global
service/index unavailability is `503`, not a successful empty result.

## Taxonomy and unknown labels

The current `COCO_CLASSES` set is authoritative. Labels are exact lower-case
canonical spellings; multiword values are preserved. The API does not
case-fold, pluralize, or accept aliases.

Unknown labels, empty labels, unsupported aliases, and duplicate rules are
structured validation errors. Query Processing may normalize a user phrase
before the API boundary, but no synonym taxonomy is invented here.

The frontend currently exposes several non-canonical labels (`man`, `woman`,
`baby`, `music`, `tennis`, `glasses`, `coffee`, `phone`, `traffic`, `flower`,
and `tree`). These cannot be passed through unchanged.

## Color and spatial status

Colors: **`RESERVED`**

- frontend color chips are presentation/state only;
- no object-color association exists in Task04/Task05 evidence;
- no backend color taxonomy/evaluator exists;
- experimental query attributes do not establish object-color support.

Geometric spatial relations: **`OUT_OF_SCOPE`**

- raw Task04 boxes exist;
- Task05 discards boxes;
- current online object route evaluates counts only;
- no source evaluates left/right/inside/overlap/near/above/below.

The legacy field name `spatial_filter` does not establish geometric support.

## Frontend compatibility

| Current field/behavior | Current meaning | Canonical mapping | Compatible? | Required future action |
|---|---|---|---:|---|
| `spatial_filter.enabled` | UI toggle | null vs non-null `object_constraints` | partial | Disabled -> null; enabled requires non-empty supported rules |
| `spatial_filter.logic` | UI AND/OR selector | `object_constraints.logic` | partial | AND usable; OR rejected while reserved |
| `selectedObjects` / `objects` | Flat UI IDs | typed `ObjectRule[]` | partial | Validate taxonomy and add typed counts |
| `selectedColors` / `colors` | Uppercase UI color chips | reserved | no | Do not send until evidence/taxonomy exists |
| `maxObjectsText` / `max_objects_text` | Free text suggesting per-label maxima | none | no | Remove; use typed numeric rule fields later |
| scene-2 object panel | “next scene” state | temporal/event constraints | no | Keep outside same-keyframe object block |
| object retrieval selector | Capability choice | `retrieval_mode: object` | yes | Allow constraint-only submission with blank text |
| search validation | Requires text or image | object-only permits constraints | no | Add object-only validation path |
| UI object IDs such as `man`, `phone` | Display shortcuts | canonical COCO labels | no | Explicit adapter/validation; no silent aliasing |
| result `objects?: string[]` | Display list | count map | no | Accept/map `Record<label,count>` without dropping counts |

The current response adapter only accepts object arrays. The backend returns an
object-count dictionary, so current counts are discarded instead of preserved.

## Backend compatibility

| Current backend behavior | Canonical contract | Gap |
|---|---|---|
| `want: "person:2,car:1"` | typed rules | Replace free-text parser at unified boundary |
| Exact 80-label COCO validation | same taxonomy | Compatible source of truth |
| Non-negative legacy count, including zero | supported typed `min_count >= 1` | Zero legacy semantics must not leak |
| `bool.filter` list | same-keyframe AND | Compatible |
| No OR path | frozen OR semantics | Implementation reserved |
| Only `gte` filters | min count | Max/total maximum reserved |
| No text required | object-only request | Compatible; Task06A validation change needed |
| Conditional route registration | stable unified route + structured `503` | Current absence appears as `404` |
| Filter context/no explicit sort | deterministic `keyframe_id ASC` | Sorting required later |
| Route-specific `SearchResponse` | Task06A envelope | Normalization required |
| Legacy string `frame_id` is keyframe identity | explicit keyframe ID + integer frame ID | Canonical identity adapter required |
| Numeric object count map in response | optional canonical counts | Compatible backend evidence |
| Hybrid ignores object constraints | mandatory hard gate when supplied | Future orchestration work |
| Task05B index excludes objects | dedicated Task05 object source/index | Ownership/index migration unresolved |

## Proposed Task06A amendments

Do not edit Task06A in this task. A future amendment should:

1. replace `objects: string[]`, `colors`, and ambiguous `max_objects` with
   typed `ObjectRule[]`, explicit `max_count`, and
   `total_max_objects`;
2. mark only AND + exact label + `min_count >= 1` as supported initially;
3. mark OR/max-count/total-max execution reserved until implemented;
4. mark non-empty colors reserved and geometric relations out of scope;
5. allow empty `query` only for object mode with a valid non-empty object
   block;
6. define a supplied object block as a mandatory hard filter in object mode;
   specify the hybrid mandatory gate as reserved/pending Task06A
   orchestration implementation;
7. distinguish valid empty, missing, and invalid object evidence;
8. add the Object-01 structured error codes;
9. require deterministic object-only ordering by `keyframe_id`;
10. resolve the finite-score parent-contract conflict: either permit a
    nullable score for Boolean-only results or add a separate
    eligibility/relevance field; do not invent an object relevance score before
    that decision;
11. require canonical `keyframe_id`, `video_id`, integer `frame_id`, nullable
    `shot_id`, and numeric timestamp normalization;
12. update object maturity wording so support means the documented subset, not
    colors/OR/maxima;
13. clarify that current specialized `/search/object` is evidence/service
    infrastructure, while unified `POST /search` remains unimplemented.

## Maturity matrix

| Capability | Status |
|---|---|
| Canonical object-label presence | `SUPPORTED_V1` |
| Same-keyframe AND | `SUPPORTED_V1` |
| Same-keyframe OR | `RESERVED` |
| Per-label `min_count >= 1` | `SUPPORTED_V1` |
| Per-label `max_count` | `RESERVED` |
| `total_max_objects` | `RESERVED` |
| Specialized object-only hard filtering | `SUPPORTED_V1` |
| Unified `POST /search` object mode | `CONTRACT_READY_NOT_IMPLEMENTED` |
| Hybrid mandatory object gate | `RESERVED` |
| Object relevance ranking | `DEFERRED` |
| Object reranking | `DEFERRED` |
| Colors | `RESERVED` |
| Geometric spatial relations | `OUT_OF_SCOPE` |
| Per-request confidence threshold | `OUT_OF_SCOPE` |
| Query-extracted object hints | `DEFERRED` |

## Open questions

1. Should the first unified implementation add a dedicated Task05 object
   index, evaluate processed documents directly, or extend another canonical
   metadata index?
2. Should OR/max-count/total-max be implemented in the first unified release
   or remain reserved for a later contract version?
3. How will the canonical COCO taxonomy version be published to clients
   without making frontend UI labels authoritative?
4. Should zero/absence rules such as `max_count: 0` be enabled once the
   evaluator can prove that a valid empty artifact exists?
5. At which orchestration stage should mandatory object filtering overfetch
   hybrid candidates so final `top_k` is not unnecessarily starved?
6. Should optional object match evidence expose only counts, or also
   matched-rule diagnostics?
7. How should legacy dynamic Elasticsearch object fields be migrated to an
   explicit all-label mapping?
8. Should pure Boolean result `score` become nullable globally, or should
   Task06A define a separate relevance/eligibility field?
9. What production Query Processing policy, if any, will map synonyms to exact
   COCO labels?

## Scope confirmation

- Only the two required Object-01 documentation files changed.
- No production frontend/backend source changed.
- Object detection was not rerun.
- Task05 documents/indexes were not rebuilt or modified.
- Multi-model embedding, Query Processing, Task06A implementation, and hybrid
  fusion were not modified.
- No merge or push was performed.
- Final verdict: **PASS_WITH_OPEN_QUESTIONS**.
