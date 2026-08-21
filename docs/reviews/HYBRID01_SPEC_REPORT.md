# Hybrid-01 Hybrid Fusion Specification Report

## Final verdict

**PASS_WITH_OPEN_QUESTIONS**

The current Task05C semantic provider fusion and specialized lexical+semantic
route were audited separately. The provider-independent hybrid contract
freezes rank-only cross-branch RRF, canonical `keyframe_id` deduplication,
deterministic ordering, final-rank timing, backend-owned pooling, finite
mode-local score semantics, successful-empty behavior, and structured
fail-closed errors. Degraded hybrid behavior, exact future pool sizing, and
Task06A response amendments remain open.

No production code, index, model artifact, BM25 implementation, Task05C,
Object-01 implementation, Task06A implementation, or frontend file changed.

## Branch and HEAD

- Branch: `docs/hybrid01-fusion-contract`
- HEAD/base: `7df9f91 docs: define object constraint contract`
- Base includes the merged multi-model/Task05C work, Data-01, Task06A/06B
  specifications, and Object-01 documentation.
- No merge and no push were performed.

## Files changed

Documentation only:

- `docs/contracts/HYBRID_FUSION.md`
- `docs/reviews/HYBRID01_SPEC_REPORT.md`

## Sources inspected

Required contracts and reports:

- `AGENTS.md`
- `docs/contracts/UNIFIED_SEARCH_API.md`
- `docs/reviews/TASK06A_SPEC_REPORT.md`
- `docs/contracts/OBJECT_CONSTRAINTS.md`
- `docs/reviews/OBJECT01_SPEC_REPORT.md`

Hybrid and semantic implementation:

- `src/mvr/retrieval/hybrid.py`
- `src/mvr/retrieval/semantic_rrf.py`
- `src/mvr/retrieval/provider_semantic.py`
- `src/mvr/retrieval/lexical.py`
- `src/mvr/retrieval/lexical_bm25.py`
- `src/mvr/api/app.py`
- `src/mvr/api/schemas.py`
- `src/mvr/retrieval/kis.py`

Task05C/index/config evidence:

- `src/mvr/indexing/provider_indexes.py`
- `configs/default.yaml`
- `configs/multi_embedding_fasttest.json`
- `docs/SEMANTIC_MODEL_RRF.md`
- `docs/MULTI_EMBEDDING_FASTTEST.md`
- `docs/MULTI_EMBEDDING_FINAL_RELEASE_REPORT.md`
- `docs/LAB_REUSE_MAP.md`
- `docs/AIC_KIS_FINAL_REPORT.md`
- `docs/AIC_KIS_IMPLEMENTATION_NOTES.md`
- `docs/CODEX_PROGRESS.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`

Tests:

- `tests/unit/retrieval/test_semantic_rrf.py`
- `tests/unit/retrieval/test_provider_semantic.py`
- `tests/unit/test_hybrid.py`
- `tests/unit/api/test_app.py`
- `tests/unit/retrieval/test_kis.py`
- `tests/integration/test_multi_embedding_fasttest_e2e.py`
- `tests/integration/test_lab6_runtime.py`

## Current Task05C semantic provider RRF

Task05C has a distinct provider fusion layer:

- `IndependentSemanticRetriever` routes each provider to a matching
  provider-specific index contract;
- provider search requests `effective_top_k * candidate_multiplier`, with
  default `candidate_multiplier=2`, then keeps the best canonical identity
  and returns a provider-local rank;
- required metadata includes `keyframe_id`, `video_id`, `frame_id`, `shot_id`,
  and `timestamp`;
- non-finite scores and malformed metadata are excluded at provider mapping;
- `fuse_semantic_rankings` uses positive 1-based ranks and
  `1 / (rrf_k + rank)`;
- default/source-backed semantic `rrf_k` is `60`;
- raw provider similarity scores do not affect RRF;
- duplicate IDs within a provider keep the best rank and emit a deterministic
  warning;
- cross-provider identity metadata conflicts for one `keyframe_id` raise
  `SemanticRrfError`;
- provider order is sorted, and ties use RRF score, best provider rank,
  provider occurrence count, then `keyframe_id`;
- full provider union is scored before semantic `top_k` truncation;
- output includes provider ranks, participating providers, missing providers,
  and warnings.

Failure behavior is explicitly provider-owned. `IndependentSemanticRetriever`
supports `fail_closed` and `degraded`; required-provider failures raise under
fail-closed, while degraded responses report coverage/errors. The semantic
fusion layer does not silently substitute another provider.

Hybrid-01 consumes this semantic ranking and does not reimplement any of the
above provider logic.

## Current specialized hybrid route

`POST /search/hybrid` in `src/mvr/api/app.py` currently:

- accepts `{query, top_k, rrf_k}` with non-blank query, `top_k` in `1..100`,
  and `rrf_k` in `1..1000`;
- calls the lexical branch with `TOP_N_EACH = 30`;
- calls the Elasticsearch semantic kNN branch with `TOP_N_EACH = 30` and
  `num_candidates=200`;
- applies `mvr.retrieval.hybrid.rrf_fuse` to two ranked lists using equal
  rank contribution and the requested `rrf_k`;
- uses the legacy `_source.frame_id` string as the candidate key;
- merges repeated candidates into one payload using a dictionary, but does
  not validate canonical `keyframe_id` or identity conflicts;
- emits `rank_lexical` and `rank_semantic` metadata;
- emits finite fused RRF scores and a route-specific `rrf_k` field;
- has deterministic `(-score, id)` ordering through `rrf_fuse`;
- catches any branch/fusion exception and returns HTTP `503`;
- is registered only when Elasticsearch is configured, reachable, and the
  configured index exists;
- returns the specialized `HybridSearchResponse`, not the Task06A unified
  envelope.

This route is real service infrastructure, but it is not the future unified
contract. In particular, it does not expose canonical `keyframe_id`,
does not support multi-provider semantic orchestration, has no explicit
partial-degradation metadata, and does not apply Object-01's reserved hybrid
object gate.

The source/configuration audit also found:

- `hybrid.py` default `rrf_k=60`;
- `HybridSearchRequest` default `rrf_k=60`;
- `configs/default.yaml` records `retrieval.rrf_k: 60`;
- the same config records `hybrid_method: weighted`, but the specialized API
  route explicitly uses RRF, so the config label must not be treated as the
  current route's executed method;
- `weighted_fuse(alpha=0.25)` exists as a separate utility but is not used by
  this specialized endpoint and has no quality-backed tuning result.

## Semantic-vs-hybrid boundary

| Layer | Owner | Input | Output |
|---|---|---|---|
| Semantic provider fusion | Task05C/SemanticSearchService | provider rankings | one semantic ranking |
| Cross-branch hybrid fusion | future HybridFusionService | semantic + lexical rankings | final canonical ranking |
| Object eligibility | Object-01/Task06A future gate | object evidence + typed constraints | Boolean eligibility |

Hybrid-01 does not expose BEiT-3, MetaCLIP-2, Perception Encoder, or provider
RRF controls to the frontend.

## Target fusion contract

The v1 contract selects equal cross-branch RRF because it is source-backed,
rank-only, deterministic, and avoids unsupported raw-score calibration:

```text
RRF(d) = sum_b 1 / (hybrid_branch_rrf_k + rank_b(d))
```

The backend owns `semantic_provider_rrf_k` and `hybrid_branch_rrf_k` as
separate settings. The source-backed contract baseline/default for
`hybrid_branch_rrf_k` is `60`; backend configuration may override it, while the
frontend never sends it. Future representative ground-truth evaluation may
justify changing that default. This is not a quality claim.

The target operation order is:

```text
branch execution
-> semantic provider fusion
-> identity validation
-> cross-branch RRF
-> canonical dedup
-> reserved object gate, if later implemented
-> deterministic sort
-> final top_k
-> contiguous rank assignment
```

Raw BM25, semantic similarity, provider RRF scores, and object counts are not
added directly. Hybrid score is finite, mode-local, and rank is authoritative.

## Identity, deduplication, tie-breaking, and rank

- Dedup key: canonical `keyframe_id`.
- Required normalized identity: `keyframe_id`, `video_id`, integer `frame_id`,
  numeric timestamp; nullable `shot_id`.
- Conflicting metadata for one keyframe is a structured fusion error.
- Equal fused scores sort by `keyframe_id ASC`.
- Final ranks are assigned only after fusion, any future object gate,
  deduplication, and truncation.
- Final ranks are contiguous `1..N`, with `N <= top_k`.

The specialized route's legacy use of source `frame_id` is recorded as a
migration gap, not silently accepted as the unified identity.

## Candidate pooling and overfetch

The target contract requires backend-owned branch pools at least as large as
final `top_k`. Exact future multipliers remain open because current evidence
comes from different layers:

- specialized hybrid branch depth: `30` per branch;
- Task05C provider candidate multiplier: `2`;
- KIS orchestration overfetch: `max(top_k * 3, 30)`.

The target must not expose provider pool sizes to the frontend. A future
reserved object gate should filter an overfetched fused pool before final
truncation; its multiplier is not frozen.

## Failure, empty, and degradation policy

Successful `[]` is a valid branch result. One successful empty branch may
still allow the other successful branch to produce candidates; both successful
empty branches produce HTTP `200` with an empty result.

The specialized `/search/hybrid` route has `SUPPORTED_CURRENT` fail-closed
behavior: any branch/fusion exception returns an all-or-nothing HTTP `503`.
The unified structured fail-closed behavior below is
`CONTRACT_READY_NOT_IMPLEMENTED`:

- semantic unavailable: `SEMANTIC_BRANCH_UNAVAILABLE`;
- lexical unavailable: `LEXICAL_BRANCH_UNAVAILABLE`;
- both unavailable: `HYBRID_NO_AVAILABLE_BRANCH`;
- identity/fusion invariant failure: `HYBRID_IDENTITY_CONFLICT` or
  `HYBRID_FUSION_ERROR`;
- directional dependency failures map to HTTP `503`; invariant failures map to
  a server-side structured error.

The current route supports only all-or-nothing `503` behavior. Task05C has a
separate provider-level degraded policy, but Hybrid-01 does not silently
promote that into cross-branch partial results. Whether an explicit degraded
hybrid mode is allowed remains an open question; if approved, `degraded` and
warning/coverage metadata must be mandatory and visible.

## Object interaction

Object-01 is preserved exactly:

- object evidence is Boolean eligibility, never a score;
- specialized object-only hard filtering exists;
- unified object mode is contract-ready but unimplemented;
- hybrid mandatory object gating is `RESERVED`/pending Task06A;
- without an object block, missing object evidence does not remove a semantic
  or lexical hit;
- with a future mandatory block, missing evidence cannot satisfy a positive
  rule and is not converted into a score penalty.

The preferred future stage is after fused overfetch and before final `top_k`,
but implementation remains outside Hybrid-01.

## Proposed Task06A amendments

Task06A is not edited. Proposed future amendments are:

1. define cross-branch hybrid RRF separately from Task05C provider RRF;
2. add optional fusion provenance and branch coverage metadata;
3. add stable hybrid branch-unavailable, no-available-branch, identity
   conflict, and fusion error codes;
4. distinguish successful empty branches from dependency failures;
5. document fail-closed v1 behavior and decide whether degraded mode may be
   enabled later;
6. state that finite hybrid RRF score is mode-local and rank is authoritative;
7. mark hybrid mandatory object gating `RESERVED`/pending Task06A;
8. preserve Object-01 canonical identity and blank-query amendment;
9. keep provider/model/RRF constants out of frontend requests.

## Maturity matrix

| Capability | Status |
|---|---|
| Task05C semantic provider RRF | `SUPPORTED_CURRENT` |
| Semantic ranking service | `SUPPORTED_CURRENT` |
| Lexical ranking service | `SUPPORTED_CURRENT` |
| Specialized cross-branch RRF | `SUPPORTED_CURRENT` |
| Unified cross-branch hybrid contract | `CONTRACT_READY_NOT_IMPLEMENTED` |
| Canonical hybrid dedup | `CONTRACT_READY_NOT_IMPLEMENTED` |
| Deterministic hybrid tie-breaking | `CONTRACT_READY_NOT_IMPLEMENTED` |
| Finite hybrid score semantics | `CONTRACT_READY_NOT_IMPLEMENTED` |
| Successful empty branch behavior | `CONTRACT_READY_NOT_IMPLEMENTED` |
| Specialized `/search/hybrid` fail-closed behavior | `SUPPORTED_CURRENT` |
| Unified structured fail-closed behavior | `CONTRACT_READY_NOT_IMPLEMENTED` |
| Explicit degraded hybrid mode | `RESERVED` |
| Degradation metadata | `RESERVED` |
| Hybrid mandatory object gate | `RESERVED` |
| Weighted fusion | `DEFERRED` |
| Score calibration | `DEFERRED` |
| Provider/model frontend fields | `OUT_OF_SCOPE` |
| Retrieval quality claims | `OUT_OF_SCOPE` |

## Open questions

1. What exact adaptive or fixed branch pool policy is sufficient for
   `top_k=100`?
2. Is degraded cross-branch hybrid allowed in v1, or must it remain
   fail-closed?
3. Which stable response fields should carry degradation warnings and branch
   coverage if degradation is approved?
4. How should Task05C provider-level fail-closed/degraded status propagate into
   the unified hybrid error envelope?
5. At what stage and with what overfetch should the future object gate run?
6. Should provenance expose raw branch scores, ranks, or only branch names?
7. When ground truth exists, should weighted/calibrated fusion be specified?

## Verification and scope confirmation

- `git diff --check`: PASS.
- `git status --short`: only
  `docs/contracts/HYBRID_FUSION.md` and
  `docs/reviews/HYBRID01_SPEC_REPORT.md` are present as untracked changes.
- No production code, Task05C code, model artifact, index, BM25,
  Query Processing, Object-01, Task06A, or frontend file was changed.
- No retrieval quality improvement is claimed.
- Final verdict: **PASS_WITH_OPEN_QUESTIONS**.
