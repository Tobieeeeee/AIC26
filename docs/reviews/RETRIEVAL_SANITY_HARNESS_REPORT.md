# Eval-01 Retrieval Sanity & Regression Harness Report

## Final verdict

**PASS**

Eval-01 is implemented as a provider-independent, ground-truth-free harness for retrieval contract validation, runtime stability, latency profiling, determinism, and ranking-change regression detection. The real Task05B Elasticsearch/BM25 provider was available and passed the live sanity run.

## Branch and HEAD

- Branch: `feat/retrieval-sanity-harness`
- Clean integration base: `main`
- HEAD: `2e11a636ae4fd28f37f720e817335ab635e90a57`
- No merge or push was performed.

## Files changed

- `.gitignore`
- `evaluation/queries/sanity_queries.json`
- `scripts/evaluation/run_retrieval_sanity.py`
- `src/mvr/evaluation/sanity_contracts.py`
- `src/mvr/evaluation/sanity_adapters.py`
- `src/mvr/evaluation/sanity_validators.py`
- `src/mvr/evaluation/sanity_determinism.py`
- `src/mvr/evaluation/sanity_regression.py`
- `src/mvr/evaluation/sanity_runner.py`
- `src/mvr/evaluation/sanity_reporting.py`
- `tests/unit/evaluation/test_sanity_validators.py`
- `tests/unit/evaluation/test_sanity_determinism.py`
- `tests/unit/evaluation/test_sanity_regression.py`
- `tests/unit/evaluation/test_sanity_runner_reporting.py`
- `tests/unit/evaluation/test_sanity_bm25_adapter.py`
- `docs/reviews/RETRIEVAL_SANITY_HARNESS_REPORT.md`

No frontend, API, indexing, retrieval-ranking, semantic-provider, model, or dataset implementation was modified.

## Evaluation architecture

The harness is separated into small evaluation-domain modules:

```text
GT-free query suite
    ↓
RetrieverAdapter
    ↓
normalized RetrievalHit records
    ↓
contract/integrity validators
    ↓
latency + determinism aggregation
    ↓
regression snapshot comparison
    ↓
JSON/JSONL/Markdown reports
```

`SanityQuery`, `RetrievalHit`, `ValidationResult`, `QueryRunResult`, `DeterminismResult`, and `SanityRun` are serializable, provider-independent contracts. They contain no relevance labels, expected keyframes, or expected ranks.

## Query-suite summary

`evaluation/queries/sanity_queries.json` contains 24 stable queries:

- 22 valid queries and 2 explicitly invalid queries;
- 10 Vietnamese, 9 English, and 5 mixed/null-language cases;
- short and long descriptions;
- person/action, vehicle, object-heavy, indoor, outdoor, rare, and ambiguous concepts;
- OCR-like and ASR-like text;
- numeric, punctuation, and Unicode cases;
- empty and whitespace-only negative cases.

The suite contains no ground-truth keyframes or relevance judgments.

## Provider interface

`RetrieverAdapter` requires:

```python
provider_id: str
search(query: str, top_k: int) -> list[RetrievalHit]
provenance() -> dict
```

The evaluator has no Elasticsearch, vector, embedding-model, or fusion assumptions. Future BEiT-3, MetaCLIP-2, Perception Encoder, and semantic-RRF adapters can implement this interface without changing the validator, runner, determinism, regression, or reporting layers.

Provider failures propagate into structured run records. There is no silent fallback between providers.

## BM25 adapter status

`BM25Adapter` directly calls the existing:

```text
mvr.retrieval.lexical_bm25.search_task05b_lexical
```

It does not duplicate BM25 request construction, Elasticsearch ranking, field weighting, score processing, or deterministic tie-breaking. It only:

- checks Elasticsearch/index availability;
- invokes the existing Task05B retrieval service;
- maps `frame_idx` to the evaluator's generic `frame_id` field;
- preserves `keyframe_id`, `video_id`, `shot_id`, timestamp, rank, and score;
- records Elasticsearch/index/Task05B-manifest provenance.

Live provider:

```text
provider_id: bm25-task05b
Elasticsearch: 8.11.4
index: mvr-fasttest-lexical
documents: 5
Task05B manifest SHA-256:
ddd3361677d0ec9eb93c199e7e1f628ec7c433e9dd60d8965a72e0f127ea9ad9
```

## Validation checks

Each returned ranking is checked for:

- nonblank canonical `keyframe_id`;
- nonblank `video_id`;
- valid integer/string `frame_id`;
- returned count not exceeding requested Top-K;
- integer, unique, increasing ranks starting at 1;
- numeric finite scores;
- duplicate canonical keyframes;
- conflicting `video_id`/`frame_id` values for repeated canonical IDs.

Raw score scales are never compared across providers.

## Latency methodology

Provider availability, index validation, and provenance collection occur before query timers start. Query latency is measured with `time.perf_counter()` for every attempt, including failures. Errors retain their measured latency and are never converted to zero.

The summary records:

- per-query/per-repetition latency;
- mean;
- median/P50;
- P95;
- minimum;
- maximum;
- per-repetition aggregates so first-pass and later-pass behavior remain distinguishable.

The five-document fasttest corpus is suitable only for runtime verification. No broad performance claim is made from these latency measurements.

## Determinism methodology

Default execution performs three repetitions for the same:

```text
query + provider + index/configuration + Top-K
```

The checker compares ordered `(rank, keyframe_id)` pairs. Score drift is reported separately with a configurable finite tolerance; score identity is not required unless the configured tolerance requires it. Erroring queries are marked non-comparable instead of being labeled deterministic.

## Regression snapshot format

Snapshots include:

- schema version and timestamp;
- Git commit and dirty state;
- provider/config/index provenance;
- Task05B manifest content/hash when supplied;
- query-suite path/hash;
- Top-K;
- ordered keyframe IDs for each successful query/provider ranking.

Comparison produces machine-readable:

- exact-order match;
- changed ranking count;
- added and removed keyframes;
- shared-keyframe rank displacement;
- overlap and Jaccard ranking-change statistics.

These statistics detect change only. They do not classify a changed ranking as better or worse.

## Report and output format

The CLI writes:

```text
reports/retrieval_sanity/<run_id>/
    manifest.json
    results.jsonl
    summary.json
    regression_diff.json  # when a baseline is supplied
    report.md
```

Generated run directories and baselines under `reports/retrieval_sanity/` are ignored by Git. The tracked query suite is the only deterministic evaluation fixture added.

CLI behavior was verified:

- valid live run exits `0`;
- unavailable provider/index exits `3`;
- invalid configuration exits `2`;
- help text is available;
- no index creation, mutation, or rebuild occurs.

## Tests and verification

Focused evaluator tests:

```text
25 passed
```

Coverage includes:

- query loading and duplicate query IDs;
- canonical identity validation;
- Top-K validation;
- duplicate keyframe detection;
- rank validation;
- NaN/infinity score validation;
- identity conflicts;
- determinism and score-tolerance behavior;
- snapshot save/load and comparison;
- JSON/JSONL/Markdown report serialization;
- BM25 adapter reuse and provider-unavailable behavior.

Full repository suite:

```text
321 passed, 6 skipped
```

CLI/error checks:

```text
--help                         PASS
missing index configuration   exit 2
unavailable index             exit 3
```

## Live BM25 sanity

Status: **PASS**

Command inputs:

```text
queries: evaluation/queries/sanity_queries.json
provider: bm25
index: mvr-fasttest-lexical
Top-K: 5
repetitions: 3
```

Final live run:

```text
queries: 24
attempts: 72
successful expected outcomes: 72
failed attempts: 0
expected invalid-query errors: 6
contract failures: 0
Top-K failures: 0
invalid-rank results: 0
non-finite-score results: 0
duplicate keyframes: 0
identity conflicts: 0
comparable determinism queries: 22
stable ordered rankings: 22
changed ordered rankings: 0
score-drift cases: 0
```

The six errors are the two intentional empty/whitespace negative queries repeated three times. They were expected and correctly surfaced.

Latest observed query latency over 72 attempts:

```text
mean: 8.412 ms
P50: 7.952 ms
P95: 14.908 ms
min: 0.001 ms
max: 27.177 ms
```

Baseline reload/comparison:

```text
compared rankings: 22
changed rankings: 0
all exact ordered matches: true
```

This live run validates the five-document fasttest provider only. It does not support any retrieval-quality or model-ranking conclusion.

## Environment blockers

None for the final run. Elasticsearch 8.11.4 and the existing five-document `mvr-fasttest-lexical` Task05B index were reachable.

If either dependency is unavailable in another environment, the CLI returns a provider-unavailable error and the live status must be reported as `BLOCKED_BY_ENVIRONMENT`; it never creates a fake success.

## Explicit GT-free statement

**No ground-truth retrieval quality metrics are produced by Eval-01.**

The harness reports only pipeline/contract correctness, runtime errors, latency, determinism, duplicate/identity integrity, and ranking-change signals.

## Future semantic-provider integration

Later work can add:

- `BEiT3Adapter`
- `MetaCLIP2Adapter`
- `PerceptionEncoderAdapter`
- `SemanticRRFAdapter`

without modifying evaluator contracts or validators. A later semantic-RRF adapter should preserve canonical `keyframe_id` fusion, deterministic ordering, missing-provider candidate handling, and separation of incompatible raw score spaces.

Eval-01 did not implement, copy, or modify any teammate Task05C/multi-embedding work and did not start Task06A.
