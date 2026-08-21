# Semantic model-level RRF

`fuse_semantic_rankings` combines only the independent semantic model lists
from Task05C.3. With the default `rrf_k=60`, a keyframe contributes
`1 / (60 + rank)` per provider using positive 1-based ranks.

Fusion is keyed by `keyframe_id`. The output retains `video_id`, `frame_id`,
`shot_id`, `timestamp`, and a sorted `provider_ranks` map. Conflicting identity
metadata is rejected. Duplicate IDs inside one provider keep the best rank and
produce a deterministic warning.

Raw provider similarity scores are not accepted by the scoring calculation.
They may differ in scale and meaning between models and therefore never affect
RRF order. This stage intentionally excludes BM25, object, and temporal lists.

Global ties use RRF score, best provider rank, provider occurrence count, then
`keyframe_id`. Fused Top-K is applied only after scoring the full input union.
The deterministic serializer is suitable for reproducibility checks.
