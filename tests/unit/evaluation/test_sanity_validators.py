from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mvr.evaluation.sanity_contracts import RetrievalHit
from mvr.evaluation.sanity_runner import load_sanity_queries
from mvr.evaluation.sanity_validators import validate_hits


def _hit(
    keyframe_id: str,
    *,
    rank: int = 1,
    score: float = 1.0,
    video_id: str = "L21_V001",
    frame_id: int = 10,
) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=score,
        keyframe_id=keyframe_id,
        video_id=video_id,
        frame_id=frame_id,
    )


def test_query_loader_preserves_gt_free_query_contract(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "text": "xin chào",
                    "language": "vi",
                    "tags": ["unicode"],
                },
                {
                    "query_id": "q_empty",
                    "text": "",
                    "tags": ["negative"],
                    "expect_error": True,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    queries = load_sanity_queries(path)

    assert [query.query_id for query in queries] == ["q1", "q_empty"]
    assert queries[0].tags == ("unicode",)
    assert queries[1].expect_error is True
    assert "ground_truth" not in queries[0].to_dict()


def test_query_loader_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            [{"query_id": "q1", "text": "a"}, {"query_id": "q1", "text": "b"}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate query IDs"):
        load_sanity_queries(path)


def test_validator_accepts_valid_canonical_ranked_hits() -> None:
    validation = validate_hits(
        [_hit("kf-a", rank=1), _hit("kf-b", rank=2, score=0.5)],
        top_k=2,
    )
    assert validation.valid is True
    assert validation.duplicate_keyframe_ids == ()


def test_validator_detects_duplicate_and_identity_conflict() -> None:
    validation = validate_hits(
        [
            _hit("kf-a", rank=1, video_id="video-a", frame_id=1),
            _hit("kf-a", rank=2, video_id="video-b", frame_id=2),
        ],
        top_k=5,
    )
    assert validation.valid is False
    assert validation.duplicate_keyframe_ids == ("kf-a",)
    assert validation.identity_conflicts == ("kf-a",)


@pytest.mark.parametrize(
    "hits",
    [
        [_hit("kf-a", rank=2)],
        [_hit("kf-a", rank=1), _hit("kf-b", rank=1)],
        [_hit("kf-a", rank=1), _hit("kf-b", rank=3), _hit("kf-c", rank=2)],
    ],
)
def test_validator_detects_invalid_ranks(hits: list[RetrievalHit]) -> None:
    assert validate_hits(hits, top_k=5).ranks_valid is False


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_validator_detects_non_finite_scores(score: float) -> None:
    validation = validate_hits([_hit("kf-a", score=score)], top_k=1)
    assert validation.scores_finite is False
    assert validation.valid is False


def test_validator_detects_top_k_violation() -> None:
    validation = validate_hits(
        [_hit("kf-a", rank=1), _hit("kf-b", rank=2)],
        top_k=1,
    )
    assert validation.top_k_valid is False


@pytest.mark.parametrize(
    "hit",
    [
        _hit("", rank=1),
        _hit("kf-a", rank=1, video_id=""),
        _hit("kf-a", rank=1, frame_id=-1),
        RetrievalHit(1, 1.0, 123, "video-a", 1),  # type: ignore[arg-type]
    ],
)
def test_validator_detects_invalid_canonical_identity(hit: RetrievalHit) -> None:
    assert validate_hits([hit], top_k=1).contract_valid is False
