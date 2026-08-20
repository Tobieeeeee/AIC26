"""The multi-embedding path against the organizer's keyframe identity.

Every fixture in the multi-embedding suite builds one video, so a keyframe
id collision had nowhere to happen -- and the organizer's corpus is nothing
but collisions: 166,550 keyframes carry 632 distinct ids, so "034" is in 842
videos, one per video.

The failure mode differs by layer and neither is loud:

**Extraction and indexing raise.** `offline._validate_inputs`,
`artifacts.load_mapping` and `provider_indexes._load_documents` each treated
the second video as a duplicate, so nothing could be built at all.

**Retrieval discards.** `provider_semantic` and `semantic_rrf` kept the
better-scoring of two different frames and recorded the loss as a
`duplicate:` warning. A top-100 could return at most 632 rows.

Both were correct while the team generated its own ids --
`L21_V001_S0007_K01` spells the video into the id -- and became wrong when
the organizer's keyframes were adopted as canonical identity.
"""
from __future__ import annotations

import pytest

from mvr.retrieval.provider_semantic import ProviderSemanticResult
from mvr.retrieval.semantic_rrf import fuse_semantic_rankings


def _result(rank: int, model: str, video_id: str, keyframe_id: str, frame_id: int):
    return ProviderSemanticResult(
        rank=rank,
        score=1.0 / rank,
        model_id=model,
        keyframe_id=keyframe_id,
        video_id=video_id,
        frame_id=frame_id,
        shot_id=1,
        timestamp=frame_id / 25.0,
    )


PROVIDERS = ("beit3", "metaclip2")


def test_the_same_keyframe_id_in_two_videos_survives_fusion() -> None:
    """Two candidates in, two results out. Keyed on keyframe_id this
    returned one, and reported the other as `duplicate:beit3:034`."""
    ranked = {
        model: (
            _result(1, model, "L21_V001", "034", 1745),
            _result(2, model, "L21_V002", "034", 900),
        )
        for model in PROVIDERS
    }

    fused = fuse_semantic_rankings(ranked, expected_providers=PROVIDERS)

    assert [(r.video_id, r.keyframe_id, r.frame_id) for r in fused.results] == [
        ("L21_V001", "034", 1745),
        ("L21_V002", "034", 900),
    ]
    assert fused.warnings == ()


def test_a_real_duplicate_within_one_video_is_still_collapsed() -> None:
    """Widening the key must not stop the deduplication it exists to do:
    one provider returning the same keyframe twice keeps the better rank."""
    ranked = {
        "beit3": (
            _result(1, "beit3", "L21_V001", "034", 1745),
            _result(5, "beit3", "L21_V001", "034", 1745),
        )
    }

    fused = fuse_semantic_rankings(ranked, expected_providers=("beit3",))

    assert len(fused.results) == 1
    assert fused.results[0].rank == 1
    assert fused.warnings == ("duplicate:beit3:L21_V001/034",)


def test_conflicting_metadata_for_one_keyframe_still_raises() -> None:
    """Two providers disagreeing about the same keyframe's frame_id is a
    real inconsistency, and must stay an error rather than becoming
    invisible now that different videos no longer collide."""
    ranked = {
        "beit3": (_result(1, "beit3", "L21_V001", "034", 1745),),
        "metaclip2": (_result(1, "metaclip2", "L21_V001", "034", 9999),),
    }

    with pytest.raises(Exception, match="conflicting identity metadata"):
        fuse_semantic_rankings(ranked, expected_providers=PROVIDERS)


def test_fusion_ranks_by_rrf_across_videos() -> None:
    """A frame ranked well by both providers must beat one ranked well by
    only one, even when they share a keyframe id -- which is exactly the
    comparison the collapse used to prevent."""
    ranked = {
        "beit3": (
            _result(1, "beit3", "L21_V002", "034", 900),
            _result(2, "beit3", "L21_V001", "034", 1745),
        ),
        "metaclip2": (
            _result(1, "metaclip2", "L21_V002", "034", 900),
            _result(9, "metaclip2", "L21_V001", "034", 1745),
        ),
    }

    fused = fuse_semantic_rankings(ranked, expected_providers=PROVIDERS)

    assert [r.video_id for r in fused.results] == ["L21_V002", "L21_V001"]
    assert fused.results[0].rrf_score > fused.results[1].rrf_score


def test_a_hundred_videos_sharing_one_id_all_survive() -> None:
    """The scale version. Every video numbers a keyframe "034", so under the
    old key a hundred results fused down to one."""
    ranked = {
        "beit3": tuple(
            _result(index + 1, "beit3", f"L21_V{index:03d}", "034", index * 10)
            for index in range(100)
        )
    }

    fused = fuse_semantic_rankings(ranked, expected_providers=("beit3",), top_k=100)

    assert len(fused.results) == 100
    assert len({r.video_id for r in fused.results}) == 100
    assert fused.warnings == ()


# -- extraction and indexing: the layers that refused to start ---------


def test_embedding_mapping_accepts_the_same_id_in_two_videos(tmp_path) -> None:
    """`artifacts.load_mapping` raised "duplicate keyframe_id in mapping" on
    the organizer's second video, so no provider artifact could be loaded."""
    import csv

    from mvr.enrichment.embeddings.artifacts import (
        EmbeddingArtifactError,
        load_mapping,
    )

    def _write(rows):
        path = tmp_path / f"mapping_{len(list(tmp_path.glob('mapping_*')))}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["keyframe_id", "video_id", "feature_file", "feature_index"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    class _Manifest:
        def __init__(self, keyframe_count):
            self.keyframe_count = keyframe_count

    class _Artifact:
        """Only what load_mapping touches: the CSV, a path resolver it calls
        to reject locators escaping the artifact root, and the declared
        keyframe count it checks the row count against."""

        def __init__(self, mapping_path, keyframe_count=2):
            self.mapping_path = mapping_path
            self.manifest = _Manifest(keyframe_count)

        def resolve_vector_path(self, feature_file):
            return self.mapping_path.parent / feature_file

    shared = _write(
        [
            {"keyframe_id": "034", "video_id": "L21_V001", "feature_file": "a.npy", "feature_index": 0},
            {"keyframe_id": "034", "video_id": "L21_V002", "feature_file": "b.npy", "feature_index": 0},
        ]
    )
    records = load_mapping(_Artifact(shared))
    assert [(r.video_id, r.keyframe_id) for r in records] == [
        ("L21_V001", "034"),
        ("L21_V002", "034"),
    ]

    # A genuine duplicate -- same video, same id -- must still be refused.
    duplicated = _write(
        [
            {"keyframe_id": "034", "video_id": "L21_V001", "feature_file": "a.npy", "feature_index": 0},
            {"keyframe_id": "034", "video_id": "L21_V001", "feature_file": "a.npy", "feature_index": 1},
        ]
    )
    with pytest.raises(EmbeddingArtifactError, match="duplicate keyframe"):
        load_mapping(_Artifact(duplicated))
