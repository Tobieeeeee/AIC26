# Hybrid Fusion Contract

## Status and scope

**Hybrid-01 specification only.**

This document freezes a provider-independent contract for the future unified
`POST /search` `retrieval_mode: "hybrid"` path. It does not implement that
route, change Task05C, rebuild an index, change BM25, add object gating, or
modify Task06A.

The contract separates two different fusion layers:

```text
Layer A: semantic providers -> one semantic ranking
Layer B: semantic ranking + lexical ranking -> final hybrid ranking
```

The current specialized services are evidence for compatibility, not proof
that the unified route already exists.

## 1. Ownership boundary

```text
Query Processing
  -> may normalize a query and emit non-authoritative hints

SemanticSearchService / Task05C
  -> calls backend-owned semantic providers
  -> performs provider-level semantic RRF
  -> returns one canonical semantic ranking

LexicalSearchService / Task05B
  -> returns one canonical lexical ranking

HybridFusionService
  -> consumes semantic and lexical rankings only
  -> performs cross-branch fusion
  -> validates identity, deduplicates, orders, and assigns final ranks
  -> does not call BEiT-3, MetaCLIP-2, or Perception Encoder directly

ObjectConstraintEvaluator
  -> supplies Boolean eligibility evidence when a future object gate exists
  -> does not contribute an object score

UnifiedSearchService
  -> owns branch selection, dependency errors, and the final response envelope
```

Provider names, model names, RRF constants, pool sizes, and branch weights are
backend-owned. They are not stable frontend request fields.

## 2. Fusion layers

### 2.1 Layer A: semantic provider fusion

Task05C owns this layer:

```text
BEiT-3 ranking
MetaCLIP-2 ranking
Perception Encoder ranking
        |
        v
semantic provider-level RRF
        |
        v
one semantic ranking
```

`HybridFusionService` receives the resulting semantic ranking and its
provenance. It must not re-run provider fusion or make provider availability a
frontend concern.

### 2.2 Layer B: cross-branch hybrid fusion

The unified hybrid path consumes:

```text
semantic ranking + lexical/BM25 ranking
        |
        v
cross-branch fusion
        |
        v
canonical final ranking
```

Object constraints are not a scoring branch. Object evidence remains Boolean
eligibility under Object-01; the mandatory hybrid object gate is
`RESERVED`/pending Task06A implementation.

## 3. Capability-level branch semantics

| `retrieval_mode` | Branches | Contract status |
|---|---|---|
| `semantic` | semantic service only | Existing provider service; unified route not implemented |
| `lexical` | lexical/BM25 service only | Existing Task05B service; unified route not implemented |
| `object` | object service only | Object-01 specialized capability exists; unified route not implemented |
| `hybrid` | semantic + lexical | Specialized route exists; unified contract is ready but not implemented |

An explicit `object_constraints` block in hybrid is intended to become a
mandatory eligibility gate after fusion and before final truncation. It is not
an implicit score and is not a supported hybrid execution capability yet.

## 4. Canonical identity and final deduplication

The only fusion and deduplication key is:

```text
keyframe_id
```

Every branch candidate must normalize and validate:

```text
keyframe_id  : non-empty canonical internal identity
video_id     : non-empty competition video identity
frame_id     : non-negative integer original-video frame identity
timestamp    : finite numeric seconds
shot_id      : optional nullable navigation metadata
```

The identity rules are inherited from Task06A and Object-01:

1. `keyframe_id` is the internal retrieval/join identity.
2. `video_id + frame_id` is the competition-facing identity.
3. `frame_id` is not a positional, filename, shot-relative, or organizer
   ordinal substitute.
4. `shot_id` and `timestamp` do not replace `keyframe_id`.
5. Missing, non-numeric, or ambiguous identity is an error; no fallback ID is
   generated.
6. Conflicting metadata for one `keyframe_id` is a contract error.

If a candidate occurs in both branches, emit one final hit and aggregate its
branch ranks. Never deduplicate by array position, image path, `video_id`
alone, or `frame_id` alone.

The current specialized route uses the legacy source field `frame_id` as a
string keyframe identity and does not expose `keyframe_id`. A migration
adapter must resolve this legacy shape before unified fusion.

## 5. Raw score non-comparability

These values are not assumed to share a scale:

```text
BM25 score
semantic similarity score
semantic provider RRF score
object counts
```

The unified hybrid contract must not add raw BM25 and semantic scores. Raw
branch scores may remain in backend/debug provenance, but they do not enter
the v1 cross-branch formula.

## 6. Baseline fusion method

The source-backed v1 baseline is equal Reciprocal Rank Fusion (RRF):

```text
RRF(d) = sum over branches b of 1 / (k + rank_b(d))
```

Rules:

- branch ranks are positive and 1-based;
- a candidate absent from a branch contributes zero for that branch;
- raw branch scores are ignored by the RRF calculation;
- every branch participates with equal weight;
- `score` is the finite hybrid RRF score for this response;
- the result is mode-local and not comparable across queries, branch sets, or
  retrieval modes;
- `rank`, not score thresholds, is the client ordering authority.

This choice is source-backed by the existing `rrf_fuse` implementation, its
tests, the specialized `/search/hybrid` route, the documented Lab-6 reuse
behavior, and the roadmap's rank-fusion baseline. It is not a retrieval
quality claim; no representative ground truth is available.

Weighted/min-max fusion exists as a utility, but is not the current
specialized hybrid route and has no evidence-backed tuning result. It is
`DEFERRED`.

## 7. RRF constant ownership

The two RRF layers have separate backend settings even though both currently
use the same source-backed contract baseline/default:

| Setting | Current evidence | Target ownership |
|---|---|---|
| `semantic_provider_rrf_k` | `semantic_rrf.py` default and Task05C fixtures use `60` | SemanticSearchService/Task05C |
| `hybrid_branch_rrf_k` | `hybrid.py`, `HybridSearchRequest`, `default.yaml`, and route behavior use `60` | HybridFusionService |

The v1 contract baseline/default for `hybrid_branch_rrf_k` is **60**. A backend
deployment configuration may override it without changing the frontend
contract. The unified frontend never sends this value. The current specialized
route accepts a request `rrf_k` (`1..1000`), but that is a legacy diagnostic
surface, not the target UI contract.

The values are independently configurable backend settings. Changing either
requires an explicit backend configuration/review change. Future representative
ground-truth evaluation may justify changing the contract default from `60`;
until then, `60` is the unambiguous source-backed default rather than an
unresolved value.

## 8. Candidate pooling and overfetch

`top_k` controls the final response only. Internal branch pools may be larger:

```text
semantic_pool_k >= top_k
lexical_pool_k  >= top_k
final results   <= top_k
```

Current source behavior:

- specialized `/search/hybrid` uses `TOP_N_EACH = 30` for both branches,
  regardless of requested `top_k`;
- the semantic provider retriever requests `effective_top_k *
  candidate_multiplier`, with a default multiplier of `2`, then deduplicates
  and truncates its provider result;
- KIS orchestration separately uses `max(top_k * 3, 30)` when it calls the
  specialized hybrid service.

These current depths must not be silently presented as the unified policy.
The target service owns branch pool sizes and must ensure each pool is at
least the final `top_k`. Exact multipliers beyond the source-backed values
above remain an `OPEN_QUESTION`, especially for `top_k=100`.

Provider-specific pool sizes and overfetch multipliers are backend metadata,
never frontend request fields.

When the reserved hybrid object gate is implemented, the backend must fuse an
overfetched pool, apply eligibility, and truncate afterward. Arbitrary
object-gate multipliers are not frozen here.

## 9. Cross-branch algorithm and final rank

The target operation order is:

```text
1. execute enabled semantic and lexical branches
2. obtain one semantic ranking from Task05C
3. validate canonical identity for every candidate
4. compute equal cross-branch RRF by 1-based branch rank
5. merge duplicate keyframe identities and retain branch provenance
6. apply the reserved object eligibility gate, when implemented
7. sort deterministically
8. truncate to final top_k
9. assign contiguous ranks 1..N
```

Deterministic ordering:

```text
1. hybrid RRF score descending
2. keyframe_id ascending
```

Backend iteration order is never a tie-breaker. Branch-local ranks remain
provenance only. Final `rank` is assigned after all fusion, gating,
deduplication, and truncation operations.

## 10. Hybrid score and provenance

For the v1 target:

```text
score = finite cross-branch hybrid RRF score
```

The score is deterministic for one response and opaque across modes/queries.
No raw BM25, semantic similarity, provider RRF, or object count is added
directly.

Optional debug provenance may include:

```json
{
  "fusion_method": "rrf",
  "hybrid_branch_rrf_k": 60,
  "contributing_branches": ["semantic", "lexical"],
  "branch_ranks": {"semantic": 2, "lexical": 1},
  "branch_scores": {"semantic": 0.84, "lexical": 7.1},
  "semantic_provider_provenance": null
}
```

This metadata is diagnostic only. Provider/model names, raw scores, and
fusion constants are not required frontend fields.

## 11. Failure and degradation semantics

### 11.1 Branch state distinctions

The service must distinguish:

| State | Meaning | Hybrid result |
|---|---|---|
| successful non-empty | branch ran and returned candidates | candidates participate |
| successful empty | branch ran and returned `[]` | no candidates/contribution; not an error |
| unavailable | dependency/index/service could not run | dependency error |
| invalid | malformed identity/index/artifact or fusion invariant failure | structured failure |

An empty ranking is never converted into an unavailable error.

### 11.2 Semantic provider failures

Task05C owns provider-level behavior:

- `IndependentSemanticRetriever` supports `fail_closed` and `degraded`;
- provider search uses backend-owned provider contracts and reports coverage;
- `fuse_semantic_rankings` ignores raw provider scores;
- missing required providers fail under `fail_closed`;
- degraded provider fusion reports missing providers and warnings rather than
  silently substituting another model.

HybridFusionService consumes the resulting semantic service response. It does
not invent a new provider fallback policy.

### 11.3 Lexical failures

The current lexical/hybrid API path catches retrieval exceptions and returns
HTTP `503`. The target lexical service must expose a structured branch failure
to the orchestration layer rather than an empty success.

### 11.4 v1 hybrid policy

The specialized `/search/hybrid` route has `SUPPORTED_CURRENT` fail-closed
behavior: any branch/fusion exception becomes an all-or-nothing HTTP `503`.
It does not emit the unified structured codes below.

The unified v1 structured fail-closed policy is
`CONTRACT_READY_NOT_IMPLEMENTED` for an unavailable required hybrid branch:

- semantic unavailable -> `503` with `SEMANTIC_BRANCH_UNAVAILABLE`;
- lexical unavailable -> `503` with `LEXICAL_BRANCH_UNAVAILABLE`;
- both unavailable -> `503` with `HYBRID_NO_AVAILABLE_BRANCH`;
- identity conflict or fusion invariant failure -> structured
  `HYBRID_IDENTITY_CONFLICT` or `HYBRID_FUSION_ERROR`;
- no silent switch to semantic-only, lexical-only, fixtures, or another
  provider.

Whether a future explicitly opted-in degraded hybrid mode is allowed remains
an `OPEN_QUESTION`. If it is approved, the response must be visibly marked,
for example:

```json
{
  "degraded": true,
  "warnings": [
    {"code": "LEXICAL_BRANCH_UNAVAILABLE"}
  ]
}
```

No degraded response metadata is frozen into Task06A here; it is a proposed
amendment.

## 12. Successful empty results

When all required enabled branches execute successfully and contribute no
candidate, the unified response is a successful empty result:

```json
{
  "status": "ok",
  "total_results": 0,
  "results": []
}
```

Examples:

- semantic returns five, lexical returns zero -> semantic candidates may be
  fused/returned;
- semantic returns zero, lexical returns ten -> lexical candidates may be
  fused/returned;
- both return zero successfully -> HTTP `200` with an empty result.

An unavailable branch is not represented as this successful empty response.

## 13. Object constraint interaction

Object-01 remains authoritative:

```text
object evidence = Boolean eligibility
object counts   != fusion score
```

Current maturity:

- specialized object-only hard filtering exists;
- unified object mode is contract-ready but not implemented;
- hybrid mandatory object gating is `RESERVED`/pending Task06A;
- missing object evidence must not remove a hybrid hit when no object block is
  supplied;
- with a future mandatory object block, missing evidence cannot satisfy a
  positive rule and must not become a fusion penalty.

Preferred future stage, not current implementation:

```text
semantic + lexical
      |
      v
fused overfetched pool
      |
      v
mandatory object eligibility gate
      |
      v
final top_k
```

Object labels, counts, colors, and object-provider fields are not hybrid
fusion inputs. Query-extracted object hints are not silently promoted to
mandatory constraints.

## 14. Structured errors

The unified Task06A error envelope should carry stable codes. Hybrid-specific
codes are:

| Code | Directional HTTP class | Meaning |
|---|---:|---|
| `SEMANTIC_BRANCH_UNAVAILABLE` | `503` | Semantic service/index unavailable |
| `LEXICAL_BRANCH_UNAVAILABLE` | `503` | Lexical service/index unavailable |
| `HYBRID_NO_AVAILABLE_BRANCH` | `503` | No required branch could execute |
| `HYBRID_IDENTITY_CONFLICT` | `500` | One keyframe identity has conflicting metadata |
| `HYBRID_FUSION_ERROR` | `500` | Fusion invariant or malformed branch result |
| `OBJECT_GATE_UNAVAILABLE` | `503` | Reserved object gate cannot execute once enabled |

Validation errors such as blank query, invalid `top_k`, or unsupported
retrieval mode continue to use the parent Task06A codes and status mapping.

## 15. Feature maturity

| Capability | Status | Evidence/decision |
|---|---|---|
| Task05C semantic provider-level RRF | `SUPPORTED_CURRENT` | `semantic_rrf.py` implements rank-only RRF |
| Semantic ranking service boundary | `SUPPORTED_CURRENT` | Independent provider service and fusion response exist |
| Specialized lexical/BM25 branch | `SUPPORTED_CURRENT` | Task05B and Lab-6 lexical services exist |
| Specialized cross-branch RRF route | `SUPPORTED_CURRENT` | `POST /search/hybrid` uses `rrf_fuse` |
| Unified cross-branch hybrid contract | `CONTRACT_READY_NOT_IMPLEMENTED` | Target semantics are defined; `POST /search` is absent |
| Canonical keyframe dedup | `CONTRACT_READY_NOT_IMPLEMENTED` | Provider semantic path validates it; specialized route still uses legacy frame identity |
| Deterministic hybrid tie-breaking | `CONTRACT_READY_NOT_IMPLEMENTED` | Utility has deterministic ID ties; unified normalization is absent |
| Finite hybrid RRF score | `CONTRACT_READY_NOT_IMPLEMENTED` | Specialized route emits RRF score; unified response not implemented |
| Successful empty branch semantics | `CONTRACT_READY_NOT_IMPLEMENTED` | Source behavior distinguishes empty responses; unified envelope absent |
| Specialized `/search/hybrid` fail-closed behavior | `SUPPORTED_CURRENT` | Any branch/fusion exception returns all-or-nothing HTTP `503` |
| Unified structured fail-closed behavior | `CONTRACT_READY_NOT_IMPLEMENTED` | Stable branch-specific codes and Task06A envelope are specified but absent |
| Explicit degraded hybrid mode | `RESERVED` | Must be visible; v1 allowance remains open |
| Hybrid degradation metadata | `RESERVED` | Proposed Task06A response amendment |
| Hybrid mandatory object gate | `RESERVED` | Pending Task06A implementation |
| Weighted fusion | `DEFERRED` | Utility exists, no evidence-backed v1 choice |
| Score calibration/normalization | `DEFERRED` | No approved cross-score calibration |
| Provider/model frontend fields | `OUT_OF_SCOPE` | Backend-owned by Task05C/semantic service |
| Retrieval quality improvement claim | `OUT_OF_SCOPE` | No representative ground truth |

## 16. Examples

### 16.1 Candidate in both branches

```text
semantic: K1(rank 2), K2(rank 1)
lexical:  K1(rank 1), K3(rank 2)
```

With `k=60`:

```text
K1 = 1/62 + 1/61
K2 = 1/61
K3 = 1/62
```

`K1` is emitted once, with both branch ranks in optional provenance.

### 16.2 Candidate in one branch

```text
semantic: K1(rank 1)
lexical:  []
```

The lexical branch contributes zero. If it completed successfully, the
semantic candidate is eligible for a successful response.

### 16.3 Equal-score tie

```text
K2 score = K1 score
```

The final order is `K1` then `K2` because `keyframe_id` is ascending.

### 16.4 Successful empty branch

```text
semantic = [K1]
lexical = []
```

This is not a lexical failure. The final result may contain `K1`.

### 16.5 One unavailable branch

```text
semantic = [K1]
lexical = unavailable
```

The v1 fail-closed policy returns `503` with
`LEXICAL_BRANCH_UNAVAILABLE`; it does not silently return semantic-only hits.

### 16.6 Both unavailable

```text
semantic = unavailable
lexical = unavailable
```

Return `503` with `HYBRID_NO_AVAILABLE_BRANCH`.

### 16.7 Reserved object gate

```text
semantic + lexical -> fused overfetch -> reserved object eligibility gate
```

The gate is not implemented in this task and object counts never alter the
RRF score.

### 16.8 Final top-k and ranks

Fusion, identity validation, deduplication, and any future object gate happen
before truncation. The response then contains at most `top_k` hits with ranks
`1..N`.

### 16.9 Debug provenance

```json
{
  "provenance": {
    "fusion_method": "rrf",
    "contributing_branches": ["semantic", "lexical"],
    "branch_ranks": {"semantic": 2, "lexical": 1}
  }
}
```

The frontend must render correctly if provenance is null or omitted.

## 17. Proposed Task06A amendments

Task06A files are intentionally not edited. A future amendment should:

1. define hybrid `score` as finite cross-branch RRF and preserve rank authority;
2. state that provider-level semantic RRF is internal to Task05C;
3. add canonical hybrid provenance as optional debug metadata;
4. add structured branch-unavailable and identity-conflict error codes;
5. state that successful empty branches differ from unavailable branches;
6. document the v1 fail-closed policy and decide whether degraded mode is
   allowed later;
7. define any `degraded`/`warnings` response fields only after that decision;
8. replace the current generic hybrid object-gate maturity with
   `RESERVED`/pending Task06A implementation;
9. preserve Object-01's blank-query amendment for object-only mode;
10. keep provider/model/RRF tuning out of frontend request fields.

## 18. Open questions

1. What exact branch pool multiplier or adaptive overfetch policy is needed for
   `top_k=100` after canonical mapping and deduplication?
2. Should degraded hybrid ever be allowed in v1, or remain fail-closed?
3. If degradation is allowed, which stable Task06A response fields carry the
   warning and branch coverage?
4. Should the reserved object gate always run after fusion, or can a
   source-backed branch-level gate be proven safe later?
5. How much object-gate overfetch is required to avoid starving final top-k?
6. How should Task05C provider-level fail-closed/degraded policy propagate into
   the cross-branch error envelope?
7. Should optional provenance expose raw branch scores, provider ranks, both,
   or only branch names?
8. Is equal RRF sufficient for the first contract, or should weighted fusion be
   separately specified after ground-truth evaluation?
9. How will legacy specialized `frame_id` values be migrated to canonical
    `keyframe_id` and integer competition `frame_id` without positional joins?

## Explicit non-goals

This document does not:

- implement `POST /search` or a new `HybridFusionService`;
- modify Task05C, semantic providers, BM25, indexes, or model artifacts;
- modify Object-01, Task06A, Task06B, Query Processing, or frontend code;
- define organizer submission behavior;
- claim retrieval quality improvement without ground truth.
