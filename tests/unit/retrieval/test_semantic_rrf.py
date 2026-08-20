from __future__ import annotations

from itertools import permutations

import pytest

from mvr.retrieval.provider_semantic import (
    IndependentSemanticResponse,
    ProviderSemanticResult,
)
from mvr.retrieval.semantic_rrf import (
    SemanticRrfError,
    fuse_independent_response,
    fuse_semantic_rankings,
)


def _result(model_id: str, keyframe_id: str, rank: int, score: float = 0.0, **identity):
    defaults = {
        "video_id": "V1",
        "frame_id": int(keyframe_id.removeprefix("K") or 0),
        "shot_id": 1,
        "timestamp": 1.0,
    }
    defaults.update(identity)
    return ProviderSemanticResult(
        rank=rank,
        score=score,
        model_id=model_id,
        keyframe_id=keyframe_id,
        **defaults,
    )


def test_exact_hand_computed_scores_for_one_two_three_provider_occurrences() -> None:
    rankings = {
        "beit3": [_result("beit3", "K1", 1), _result("beit3", "K2", 2)],
        "metaclip2": [_result("metaclip2", "K1", 2), _result("metaclip2", "K3", 1)],
        "perception_encoder": [_result("perception_encoder", "K1", 3)],
    }
    fused = fuse_semantic_rankings(rankings)
    by_id = {item.keyframe_id: item for item in fused.results}
    assert by_id["K1"].rrf_score == pytest.approx(
        1 / 61 + 1 / 62 + 1 / 63
    )
    assert by_id["K2"].rrf_score == pytest.approx(1 / 62)
    assert by_id["K3"].rrf_score == pytest.approx(1 / 61)
    assert by_id["K1"].provider_ranks == {
        "beit3": 1,
        "metaclip2": 2,
        "perception_encoder": 3,
    }


def test_duplicate_keeps_best_rank_and_surfaces_warning() -> None:
    fused = fuse_semantic_rankings(
        {
            "beit3": [
                _result("beit3", "K1", 5),
                _result("beit3", "K1", 2),
            ]
        }
    )
    assert fused.results[0].provider_ranks == {"beit3": 2}
    assert fused.results[0].rrf_score == pytest.approx(1 / 62)
    assert fused.warnings == ("duplicate:beit3:V1/K1",)


def test_empty_and_missing_provider_policies() -> None:
    empty = fuse_semantic_rankings(
        {"beit3": []}, expected_providers=["beit3"], failure_policy="fail_closed"
    )
    assert empty.results == ()
    assert empty.missing_providers == ()
    with pytest.raises(SemanticRrfError, match="missing"):
        fuse_semantic_rankings(
            {"beit3": []},
            expected_providers=["beit3", "metaclip2"],
            failure_policy="fail_closed",
        )
    degraded = fuse_semantic_rankings(
        {"beit3": []},
        expected_providers=["beit3", "metaclip2"],
        failure_policy="degraded",
    )
    assert degraded.missing_providers == ("metaclip2",)
    assert degraded.warnings == ("missing_provider:metaclip2",)


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_invalid_k_is_rejected(rrf_k: int) -> None:
    with pytest.raises(SemanticRrfError, match="positive"):
        fuse_semantic_rankings({}, rrf_k=rrf_k)


@pytest.mark.parametrize("rank", [0, -1, 1.5, True])
def test_invalid_rank_is_rejected(rank) -> None:
    with pytest.raises(SemanticRrfError, match="invalid rank"):
        fuse_semantic_rankings({"beit3": [_result("beit3", "K1", rank)]})


def test_identity_conflict_is_rejected() -> None:
    with pytest.raises(SemanticRrfError, match="conflicting identity"):
        fuse_semantic_rankings(
            {
                "beit3": [_result("beit3", "K1", 1, frame_id=1)],
                "metaclip2": [_result("metaclip2", "K1", 1, frame_id=999)],
            }
        )


def test_deterministic_ties_and_provider_order_permutations() -> None:
    rows = {
        "beit3": [_result("beit3", "K2", 1)],
        "metaclip2": [_result("metaclip2", "K1", 1)],
        "perception_encoder": [_result("perception_encoder", "K3", 1)],
    }
    serialized = set()
    for order in permutations(rows):
        fused = fuse_semantic_rankings({key: rows[key] for key in order})
        serialized.add(fused.to_json())
        assert [item.keyframe_id for item in fused.results] == ["K1", "K2", "K3"]
    assert len(serialized) == 1


def test_raw_score_changes_do_not_change_rrf_or_serialization() -> None:
    first = fuse_semantic_rankings(
        {
            "beit3": [_result("beit3", "K1", 1, score=-999)],
            "metaclip2": [_result("metaclip2", "K2", 1, score=999)],
        }
    )
    second = fuse_semantic_rankings(
        {
            "beit3": [_result("beit3", "K1", 1, score=12345)],
            "metaclip2": [_result("metaclip2", "K2", 1, score=-12345)],
        }
    )
    assert first.to_json() == second.to_json()


def test_fused_top_k_applies_after_full_union() -> None:
    rankings = {
        "beit3": [_result("beit3", f"K{i}", i) for i in range(1, 6)],
        "metaclip2": [_result("metaclip2", "K5", 1)],
    }
    fused = fuse_semantic_rankings(rankings, top_k=1)
    assert [item.keyframe_id for item in fused.results] == ["K5"]


def test_independent_response_orchestration_hook_preserves_policy() -> None:
    response = IndependentSemanticResponse(
        ranked_lists={"beit3": (_result("beit3", "K1", 1),)},
        requested_providers=("beit3", "metaclip2"),
        successful_providers=("beit3",),
        failed_providers=("metaclip2",),
        errors={"metaclip2": "TimeoutError"},
        failure_policy="degraded",
    )
    fused = fuse_independent_response(response)
    assert fused.missing_providers == ("metaclip2",)
    assert fused.results[0].video_id == "V1"
    assert fused.results[0].frame_id == 1
