from pathlib import Path

import pytest

from mvr.evaluation.trake import TrakeChainPrediction
from mvr.retrieval.trake_events import TrakeGapConstraint
from mvr.submission.trake_export import (
    InvalidTrakePredictionError,
    TrakePredictionRow,
    apply_never_blank_fallback,
    export_local_trake_verification_csv,
    export_local_trake_verification_json,
    load_local_trake_verification_json,
    validate_trake_prediction,
)

FPS = 25.0

# -- apply_never_blank_fallback -----------------------------------------------


def test_fully_resolved_chain_passes_through_unchanged() -> None:
    prediction = TrakeChainPrediction("v", (100, 200, 300), score=1.0)
    gaps = (TrakeGapConstraint(0, 20), TrakeGapConstraint(0, 20))

    row = apply_never_blank_fallback(prediction, gaps, video_fps=FPS)

    assert row.video_id == "v"
    assert row.frames == (100, 200, 300)
    assert row.is_interpolated == (False, False, False)


def test_single_interior_none_interpolates_at_the_real_time_midpoint() -> None:
    # prev=100 (t=4.0s), next=400 (t=16.0s), symmetric hop widths -> midpoint.
    prediction = TrakeChainPrediction("v", (100, None, 400), score=1.0)
    gaps = (TrakeGapConstraint(0, 20), TrakeGapConstraint(0, 20))

    row = apply_never_blank_fallback(prediction, gaps, video_fps=FPS)

    assert row.frames == (100, 250, 400)  # t=10.0s = frame 250
    assert row.is_interpolated == (False, True, False)


def test_two_consecutive_interior_nones_split_proportionally_by_gap_width() -> None:
    # fps=1 for simple arithmetic: prev=0 (t=0s), next=1000 (t=1000s).
    # Hop widths: event0->1 = 10, event1->2 = 10, event2->3 = 80 (total 100).
    # The wide hop (event2->3) must absorb ~80% of the real 1000s span.
    prediction = TrakeChainPrediction("v", (0, None, None, 1000), score=1.0)
    gaps = (
        TrakeGapConstraint(0, 10),
        TrakeGapConstraint(0, 10),
        TrakeGapConstraint(0, 80),
    )

    row = apply_never_blank_fallback(prediction, gaps, video_fps=1.0)

    marker1, marker2 = row.frames[1], row.frames[2]
    assert row.frames[0] == 0 and row.frames[3] == 1000
    assert row.is_interpolated == (False, True, True, False)
    # marker1 and marker2 must be DISTINCT, not both at the naive midpoint (500).
    assert marker1 != marker2
    # The two narrow hops (widths 10, 10) span marker1->marker2; the one wide
    # hop (width 80) spans marker2->next -- so that gap must be much larger.
    assert (marker2 - marker1) < (1000 - marker2)
    assert marker1 == pytest.approx(100, abs=1)  # 10/100 of 1000s
    assert marker2 == pytest.approx(200, abs=1)  # 20/100 of 1000s


def test_leading_none_extrapolates_backward_using_gap_midpoint() -> None:
    # next=1000 (t=40.0s), gap midpoint = (0+20)/2 = 10s -> t=30.0s = frame 750.
    prediction = TrakeChainPrediction("v", (None, 1000), score=1.0)
    gaps = (TrakeGapConstraint(0, 20),)

    row = apply_never_blank_fallback(prediction, gaps, video_fps=FPS)

    assert row.frames == (750, 1000)
    assert row.is_interpolated == (True, False)


def test_trailing_none_extrapolates_forward_using_gap_midpoint() -> None:
    # prev=0 (t=0s), gap midpoint = (10+30)/2 = 20s -> frame 500.
    prediction = TrakeChainPrediction("v", (0, None), score=1.0)
    gaps = (TrakeGapConstraint(10, 30),)

    row = apply_never_blank_fallback(prediction, gaps, video_fps=FPS)

    assert row.frames == (0, 500)
    assert row.is_interpolated == (False, True)


def test_all_markers_none_is_rejected_not_fabricated() -> None:
    prediction = TrakeChainPrediction("v", (None, None, None), score=0.0)
    gaps = (TrakeGapConstraint(0, 20), TrakeGapConstraint(0, 20))

    with pytest.raises(InvalidTrakePredictionError):
        apply_never_blank_fallback(prediction, gaps, video_fps=FPS)


# -- validate_trake_prediction -------------------------------------------------


def test_validate_accepts_a_well_formed_row() -> None:
    row = TrakePredictionRow("v", (1, 5, 10), (False, False, False))

    validate_trake_prediction(row, k=3)  # must not raise


def test_validate_rejects_wrong_marker_count() -> None:
    row = TrakePredictionRow("v", (1, 5), (False, False))

    with pytest.raises(InvalidTrakePredictionError):
        validate_trake_prediction(row, k=3)


def test_validate_rejects_non_increasing_frames() -> None:
    row = TrakePredictionRow("v", (5, 3, 10), (False, False, False))

    with pytest.raises(InvalidTrakePredictionError):
        validate_trake_prediction(row, k=3)


def test_validate_rejects_negative_frame() -> None:
    row = TrakePredictionRow("v", (-1, 5, 10), (False, False, False))

    with pytest.raises(InvalidTrakePredictionError):
        validate_trake_prediction(row, k=3)


# -- export round-trip ---------------------------------------------------------


def test_json_export_round_trip_preserves_interpolation_flags(tmp_path: Path) -> None:
    rows = [
        TrakePredictionRow("L21_V018", (11068, 11340, 11600), (False, False, False)),
        TrakePredictionRow("L21_V024", (13890, 14260, 14635), (False, True, False)),
    ]
    path = tmp_path / "export.json"

    export_local_trake_verification_json(rows, path, query_id="q1")
    loaded = load_local_trake_verification_json(path)

    assert loaded == rows


def test_csv_export_writes_expected_header_and_rows(tmp_path: Path) -> None:
    rows = [TrakePredictionRow("v", (100, 200), (False, True))]
    path = tmp_path / "export.csv"

    export_local_trake_verification_csv(rows, path, query_id="q1")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "format,query_id,rank,video_id,frames,is_interpolated"
    assert lines[1] == "local_verification_export,q1,1,v,100;200,False;True"
