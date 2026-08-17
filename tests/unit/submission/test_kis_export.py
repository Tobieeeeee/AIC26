from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mvr.submission.kis_export import (
    InvalidKisPredictionError,
    KisPrediction,
    export_local_verification_csv,
    export_local_verification_json,
    load_local_verification_json,
    validate_kis_predictions,
)


@dataclass(frozen=True)
class _FakeCandidate:
    video_id: str
    frame_id: int


# -- ordering -----------------------------------------------------------------


def test_validate_preserves_ranking_order() -> None:
    predictions = validate_kis_predictions(
        [_FakeCandidate("L21_V001", 300), _FakeCandidate("L21_V001", 100), _FakeCandidate("L21_V002", 50)]
    )

    assert [p.frame_id for p in predictions] == [300, 100, 50]
    assert [p.rank for p in predictions] == [1, 2, 3]


# -- max 100 --------------------------------------------------------------------


def test_validate_accepts_exactly_100_unique_predictions() -> None:
    candidates = [_FakeCandidate("L21_V001", i) for i in range(100)]

    predictions = validate_kis_predictions(candidates)

    assert len(predictions) == 100


def test_validate_rejects_more_than_100_unique_predictions() -> None:
    candidates = [_FakeCandidate("L21_V001", i) for i in range(101)]

    with pytest.raises(InvalidKisPredictionError, match="at most 100"):
        validate_kis_predictions(candidates)


def test_validate_allows_over_100_raw_items_if_duplicates_bring_it_under_the_limit() -> None:
    # 101 raw items but only 100 unique (video_id, frame_id) pairs.
    candidates = [_FakeCandidate("L21_V001", 0)] + [_FakeCandidate("L21_V001", i) for i in range(100)]

    predictions = validate_kis_predictions(candidates)

    assert len(predictions) == 100


# -- invalid IDs ------------------------------------------------------------------


def test_validate_rejects_blank_video_id() -> None:
    with pytest.raises(InvalidKisPredictionError, match="video_id"):
        validate_kis_predictions([_FakeCandidate("   ", 10)])


def test_validate_rejects_non_integer_frame_id() -> None:
    with pytest.raises(InvalidKisPredictionError, match="frame_id"):
        validate_kis_predictions([_FakeCandidate("L21_V001", 10.5)])


def test_validate_rejects_bool_frame_id() -> None:
    # bool is a subclass of int in Python; must not silently pass as a frame_id.
    with pytest.raises(InvalidKisPredictionError, match="frame_id"):
        validate_kis_predictions([_FakeCandidate("L21_V001", True)])


def test_validate_rejects_negative_frame_id() -> None:
    with pytest.raises(InvalidKisPredictionError, match="frame_id"):
        validate_kis_predictions([_FakeCandidate("L21_V001", -1)])


# -- exact duplicate handling -----------------------------------------------------


def test_validate_removes_exact_duplicates_keeping_first_rank() -> None:
    predictions = validate_kis_predictions(
        [
            _FakeCandidate("L21_V001", 261),
            _FakeCandidate("L21_V002", 45),
            _FakeCandidate("L21_V001", 261),  # exact duplicate of rank 1
        ]
    )

    assert len(predictions) == 2
    assert predictions[0] == KisPrediction(rank=1, video_id="L21_V001", frame_id=261)
    assert predictions[1] == KisPrediction(rank=2, video_id="L21_V002", frame_id=45)


def test_validate_does_not_dedupe_same_frame_id_across_different_videos() -> None:
    predictions = validate_kis_predictions(
        [_FakeCandidate("L21_V001", 100), _FakeCandidate("L21_V002", 100)]
    )

    assert len(predictions) == 2


# -- original frame ID preserved --------------------------------------------------


def test_validate_preserves_exact_original_frame_id() -> None:
    predictions = validate_kis_predictions([_FakeCandidate("L21_V001", 5760)])

    assert predictions[0].frame_id == 5760
    assert predictions[0].video_id == "L21_V001"


# -- local export round-trip -------------------------------------------------------


def test_local_json_export_round_trips(tmp_path: Path) -> None:
    predictions = validate_kis_predictions(
        [_FakeCandidate("L21_V001", 261), _FakeCandidate("L21_V002", 45)]
    )
    output_path = tmp_path / "export.json"

    export_local_verification_json(predictions, output_path, query_id="q1")
    loaded = load_local_verification_json(output_path)

    assert loaded == predictions


def test_local_json_export_is_clearly_labeled_not_official(tmp_path: Path) -> None:
    import json

    predictions = validate_kis_predictions([_FakeCandidate("L21_V001", 261)])
    output_path = tmp_path / "export.json"

    export_local_verification_json(predictions, output_path, query_id="q1")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["format"] == "local_verification_export"
    assert "not the official" in payload["note"].lower()


def test_local_csv_export_round_trips_via_manual_parse(tmp_path: Path) -> None:
    import csv

    predictions = validate_kis_predictions(
        [_FakeCandidate("L21_V001", 261), _FakeCandidate("L21_V002", 45)]
    )
    output_path = tmp_path / "export.csv"

    export_local_verification_csv(predictions, output_path, query_id="q1")

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["format"] == "local_verification_export"
    assert rows[0]["video_id"] == "L21_V001"
    assert int(rows[0]["frame_id"]) == 261
    assert rows[1]["video_id"] == "L21_V002"
    assert int(rows[1]["frame_id"]) == 45


def test_load_local_verification_json_rejects_wrong_format_label(tmp_path: Path) -> None:
    import json

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"format": "something_else", "predictions": []}), encoding="utf-8")

    with pytest.raises(InvalidKisPredictionError):
        load_local_verification_json(bad_path)
