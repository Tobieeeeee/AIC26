# AIC TRAKE Implementation Notes

Companion to `docs/AIC_KIS_IMPLEMENTATION_NOTES.md`, written for the TRAKE
task pack (`agent_task/trake_competition_ready/`, local-only). Records the
exact existing interfaces later TRAKE phases must reuse, and the two
behavior changes made in Phase 01.

## Phase 01 Changes

### 1. Resolver bug fix: `_normalize_ref()` now raises `MissingFrameMappingError`

`src/mvr/ingestion/aic_dataset.py::_normalize_ref()` previously raised a
bare `ValueError` for a non-numeric keyframe reference. `AicDatasetRegistry
.resolve_frame()` calls it before its own `except KeyError ->
MissingFrameMappingError` block, so the `ValueError` propagated uncaught.
`src/mvr/retrieval/kis.py::resolve_original_frame_id()` only catches
`MissingFrameMappingError`, so any candidate whose `keyframe_id` was this
repo's own pipeline format (non-numeric, e.g. `"L01_V001_S0002_K01"`)
crashed the entire `search_kis()` call instead of being dropped.

Fixed at the root: `_normalize_ref()` now raises `MissingFrameMappingError`
for a non-numeric reference. `kis.py` was not touched — its existing
`except MissingFrameMappingError: return None` now handles this path
correctly.

**Blast radius found beyond the reported bug:** `_normalize_ref()` has a
second internal caller, `keyframe_path()`'s directory scan, which had its
own `except ValueError: continue` guarding against malformed `.jpg`
filenames. Changing the raised type would have broken that guard silently
(a single non-numeric-stem file in a real keyframes directory would have
crashed `keyframe_path()` entirely). Updated that `except` clause to
`MissingFrameMappingError` in the same commit so this defensive behavior is
preserved, not just the originally-reported path.

`tests/unit/test_tkis_resolver_regression.py` (not ours) had a
`MockAicRegistry` that hard-coded its own `raise ValueError(...)`,
independent of the real implementation — fixing `aic_dataset.py` alone
would not have made its test fail, silently leaving a stale assertion.
Both the mock and the assertion in `test_pipeline_resolver_missing_mapping`
were corrected to match real behavior (`MissingFrameMappingError` in,
`None` out, no raise).

### 2. `search_kis()`'s `top_k` ceiling raised from 100 to 500

`src/mvr/retrieval/kis.py::search_kis()` validated `1 <= top_k <= 100`.
TRAKE's video-shortlisting step (Phase 05) needs a wider per-event
candidate pool (mirroring the course lab's own `candidate_videos()` helper,
which scans each event's top-500 hits to build a video shortlist, not
top-100) so a valid video is less likely to be missed entirely before
alignment runs.

Changed the validation to `1 <= top_k <= 500`. Nothing else in the function
assumed `top_k<=100` — the overfetch formula `fetch_count = max(top_k * 3,
30)` and the final truncation loop both already scale correctly to the new
range; verified via `test_search_kis_accepts_top_k_above_100_for_trake_shortlisting`
(300 requested, 300 returned).

**Not changed:** `src/mvr/api/schemas.py`'s `KisSearchRequest.top_k` stays
`le=100` — the public `POST /search/kis` HTTP contract is unaffected. This
is a ceiling increase internal to `search_kis()`'s Python call surface
only, used by TRAKE's own shortlisting call path (Phase 05), not a change
to any existing caller's behavior at `top_k<=100`.

Both changes are also recorded in `docs/LAB_REUSE_MAP.md`.

## Audit — Existing Interfaces TRAKE Must Reuse

### `search_kis()` (`src/mvr/retrieval/kis.py`)

```python
def search_kis(
    query: str,
    *,
    top_k: int,                                          # now 1..500
    semantic_service: _SemanticService | None,
    multimodal_service: _MultimodalService | None,
    pipeline_resolver: KeyframeFrameIndexResolver | None = None,
    aic_registry: Any | None = None,
    overfetch: int | None = None,
) -> list[KisCandidate]:
```

Raises `ValueError` for blank query or `top_k` out of range; `RuntimeError`
if neither backend is configured. Deduplicates on `(video_id, frame_id)`.
Never raises on a single unresolvable candidate — drops it and continues
(this was the class of bug fixed above).

### `KisCandidate` (`src/mvr/retrieval/kis.py`)

```python
@dataclass(frozen=True)
class KisCandidate:
    rank: int
    video_id: str
    frame_id: int
    score: float                        # original coarse retrieval score, never overwritten
    keyframe_id: str
    timestamp: float | None = None
    refinement_score: float | None = None   # separate space, added in PR#2 hardening

    def to_answer(self) -> KisAnswer: ...   # lazy import of evaluation.aic_kis.KisAnswer inside the method -- follow this exact pattern for any retrieval type that needs an evaluation type (see Phase 04's "Import direction" note)
```

### `AicDatasetRegistry` (`src/mvr/ingestion/aic_dataset.py`)

Public methods: `video_path(video_id)`, `keyframe_path(video_id, ref)`,
`resolve_frame(video_id, keyframe_ref) -> AicFrameMapping` (raises
`MissingFrameMappingError`, never guesses), `video_info(video_id) ->
VideoInfo` (raw-video-first probe, works without organizer support
assets), `media_info(video_id) -> dict | None` (descriptive only, never a
frame-mapping source).

`AicFrameMapping` fields: `video_id, keyframe_ref, frame_idx, timestamp,
fps`.

### `evaluate_kis()` / `KisGroundTruth` / `KisAnswer` / `KIS_CUTOFFS` (`src/mvr/evaluation/aic_kis.py`)

`KIS_CUTOFFS = (1, 5, 20, 50, 100)`. `KisGroundTruth(video_id, start_frame,
end_frame)` — an interval, binary correctness. `KisAnswer(video_id,
frame_id)`. `KisScore(recall_at_k: dict[int, float], final_score: float)`.

**Not reusable as-is for TRAKE:** TRAKE's ground truth is one target frame
per event (with a frame-count tolerance), not an interval, and TRAKE
correctness is partial-per-chain, not binary. TRAKE's evaluator
(`src/mvr/evaluation/trake.py`, Phase 02) is new code with the same
`recall_at_k`/`final_score` *shape* for consistency, but different
correctness math — do not attempt to reuse `kis_r_score()` or
`evaluate_kis()` directly.

### `KeyframeFrameIndexResolver` (`src/mvr/retrieval/kis.py`)

```python
class KeyframeFrameIndexResolver:
    def __init__(self, mapping: Mapping[str, int]) -> None: ...
    @classmethod
    def from_csv(cls, path: Path | str) -> "KeyframeFrameIndexResolver": ...
    def resolve(self, keyframe_id: str) -> int | None: ...
```

Backed by this repo's own `keyframe_mapping.csv` (`keyframe_id ->
frame_idx`). Tried first in `resolve_original_frame_id()`, before falling
back to `aic_registry`.

### `configs/default.yaml` `kis:` section

```yaml
kis:
  top_k_default: 100
  top_k_max: 100
  benchmark_top_k: 100
  overfetch_multiplier: 3
  overfetch_floor: 30
  refinement_enabled: false
  refinement_top_n: 3
  refinement_frame_radius: 45
  local_export_format: local_verification_export
```

Unchanged by this phase. A `trake:` section will be added in Phase 06 once
the benchmark selects a default alignment strategy — not before, per the
task pack's evidence-based-choice principle.

### No TRAKE files exist yet

Confirmed: no `src/mvr/retrieval/trake*.py` or `src/mvr/evaluation/trake.py`
exist as of this commit. Later phases create them.

### `FaissIndex` (`src/mvr/indexing/faiss_index.py`) — dense scoring dependency

Confirmed built as `faiss.IndexFlatIP` (`FaissIndex.build()`), storing
every L2-normalized vector verbatim. The underlying `faiss.IndexFlatIP`
object (accessible as `FaissIndex.index`) natively supports
`.reconstruct(i)` and `.reconstruct_n(start, num)`, verified directly:
reconstructing a stored vector returns values numerically identical to the
original input vector (`np.allclose`, no precision loss beyond float32).
`FaissIndex.metadata` is a parallel `list[dict]` (one entry per FAISS
internal id, same order), each entry carrying `video_id`, `timestamp`,
`frame_id`, `image_path` (enforced by `FaissIndex.load()`'s
`required_metadata` check). This is what Phase 04's dense per-video
scoring (`build_video_score_matrix()`) will reconstruct from — confirmed
real, not assumed.

### `SemanticRetriever` (`src/mvr/retrieval/semantic.py`) — attribute path for Phase 05

```python
class SemanticRetriever:
    def __init__(self, text_encoder, index) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...
```

`.text_encoder.encode_text([...])` and `.index` (a `FaissIndex` instance,
same object type as above) are both directly accessible attributes on a
`SemanticRetriever` instance. Phase 05's orchestration should receive (or
construct) one `SemanticRetriever`-shaped dependency and pull both the text
encoder and the FAISS index from it, rather than requiring two separately
threaded parameters where one already composes the other.

Note: this dense-scoring path only works against the semantic-only FAISS
branch. `ElasticsearchSemanticRetriever` (same file) has no local vector
store to reconstruct from — its `search()` returns a raw ES response, not
reconstructable vectors. TRAKE's dense scoring (Phase 04) is therefore
scoped to the FAISS/semantic-only retrieval backend; this is consistent
with KIS's own production default (semantic-only, refinement off) recorded
in `docs/AIC_KIS_FINAL_REPORT.md`.
