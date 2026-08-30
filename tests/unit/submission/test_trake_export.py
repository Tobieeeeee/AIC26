from pathlib import Path

import pytest

from mvr.evaluation.trake import TrakeChainPrediction
from mvr.retrieval.trake_events import TrakeGapConstraint
from mvr.submission.trake_export import (
    OFFICIAL_MAX_ROWS_PER_QUERY,
    InvalidTrakePredictionError,
    TrakePredictionRow,
    apply_never_blank_fallback,
    export_local_trake_verification_csv,
    export_local_trake_verification_json,
    export_official_trake_submission_csv,
    load_local_trake_verification_json,
    official_submission_filename,
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


def test_validate_rejects_blank_video_id() -> None:
    row = TrakePredictionRow("", (1, 5, 10), (False, False, False))

    with pytest.raises(InvalidTrakePredictionError):
        validate_trake_prediction(row, k=3)


def test_validate_rejects_whitespace_only_video_id() -> None:
    row = TrakePredictionRow("   ", (1, 5, 10), (False, False, False))

    with pytest.raises(InvalidTrakePredictionError):
        validate_trake_prediction(row, k=3)


def test_validate_rejects_video_id_with_mp4_suffix() -> None:
    row = TrakePredictionRow("L21_V001.mp4", (1, 5, 10), (False, False, False))

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


# -- official AIC26 submission CSV ----------------------------------------------
#
# Fixtures below are the organizer's OWN published examples, fetched from
# https://sotuyenaic.oj.io.vn/ on 2026-08-21 (the "TRAKE (query-3-trake.csv -
# 4 events)" example), not invented -- byte-for-byte match is the strongest
# verification available for a format this codebase does not control.


def test_official_csv_matches_the_organizer_published_example_exactly(tmp_path: Path) -> None:
    rows = [
        TrakePredictionRow("L10_V001", (1200, 1850, 2100, 2450), (False, False, False, False)),
        TrakePredictionRow("L10_V001", (1180, 1820, 2080, 2420), (False, False, False, False)),
        TrakePredictionRow("L11_V003", (5100, 5700, 6200, 6800), (False, False, False, False)),
    ]
    path = tmp_path / "query-3-trake.csv"

    export_official_trake_submission_csv(rows, path)

    # Line ending is CRLF on disk (csv module's default "excel" dialect);
    # read_text() universal-newline-translates it back to "\n" for
    # comparison, which is fine -- the organizer explicitly accepts either.
    assert path.read_text(encoding="utf-8").strip("\n").splitlines() == [
        "L10_V001,1200,1850,2100,2450",
        "L10_V001,1180,1820,2080,2420",
        "L11_V003,5100,5700,6200,6800",
    ]


def test_official_csv_has_no_header_row(tmp_path: Path) -> None:
    rows = [TrakePredictionRow("L01_V028", (25300,), (False,))]
    path = tmp_path / "out.csv"

    export_official_trake_submission_csv(rows, path)

    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "L01_V028,25300"  # data, not "video_id,frame_1..."


def test_official_csv_video_name_carries_no_mp4_suffix(tmp_path: Path) -> None:
    rows = [TrakePredictionRow("L01_V028", (25300,), (False,))]
    path = tmp_path / "out.csv"

    export_official_trake_submission_csv(rows, path)

    assert ".mp4" not in path.read_text(encoding="utf-8")


def test_official_csv_rejects_an_empty_result_set() -> None:
    with pytest.raises(InvalidTrakePredictionError):
        export_official_trake_submission_csv([], "unused.csv")


def test_official_csv_rejects_more_than_100_rows(tmp_path: Path) -> None:
    rows = [TrakePredictionRow("v", (1, 2), (False, False))] * (OFFICIAL_MAX_ROWS_PER_QUERY + 1)

    with pytest.raises(InvalidTrakePredictionError):
        export_official_trake_submission_csv(rows, tmp_path / "out.csv")


def test_official_csv_accepts_exactly_the_100_row_cap(tmp_path: Path) -> None:
    rows = [TrakePredictionRow("v", (1, 2), (False, False))] * OFFICIAL_MAX_ROWS_PER_QUERY
    path = tmp_path / "out.csv"

    export_official_trake_submission_csv(rows, path)

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == OFFICIAL_MAX_ROWS_PER_QUERY


def test_official_submission_filename_matches_the_organizer_examples() -> None:
    """Byte-for-byte match to the organizer's own published examples:
    query-1-kis.csv, query-2-qa.csv, query-3-trake.csv."""
    assert official_submission_filename(1, "kis") == "query-1-kis.csv"
    assert official_submission_filename(2, "qa") == "query-2-qa.csv"
    assert official_submission_filename(3, "trake") == "query-3-trake.csv"


def test_official_submission_filename_rejects_unknown_query_type() -> None:
    with pytest.raises(ValueError):
        official_submission_filename(1, "unknown-type")


def test_official_submission_filename_rejects_non_positive_number() -> None:
    with pytest.raises(ValueError):
        official_submission_filename(0, "trake")
    with pytest.raises(ValueError):
        official_submission_filename(-1, "trake")


def test_official_submission_filename_rejects_non_int_number() -> None:
    with pytest.raises(ValueError):
        official_submission_filename(1.5, "trake")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        official_submission_filename(True, "trake")  # bool is technically an int


def test_official_submission_filename_keeps_a_pack_name() -> None:
    """The submission carries the query file's own name. The organizer says
    so by example: their pack holds `query-1-trake.txt` and their required
    listing holds `query-1-trake.csv` -- same name, new extension. A real
    pack is `query-p1-16-trake`, and building `query-16-trake.csv` from it
    invents a name their listing does not contain."""
    assert (
        official_submission_filename("query-p1-16-trake.txt")
        == "query-p1-16-trake.csv"
    )
    assert official_submission_filename("query-p1-16-trake") == "query-p1-16-trake.csv"


def test_official_submission_filename_keeps_a_batch_prefixed_id() -> None:
    assert official_submission_filename("p1-16", "trake") == "query-p1-16-trake.csv"


def test_official_csv_rejects_mixed_marker_counts_in_one_file(tmp_path: Path) -> None:
    """The organizer compares Frame ID count against the query's own
    declared event count -- a file mixing k=3 and k=4 rows is already
    broken before it reaches the grader, so this must fail loudly here
    rather than silently upload something invalid."""
    rows = [
        TrakePredictionRow("v1", (1, 2, 3), (False, False, False)),
        TrakePredictionRow("v2", (1, 2, 3, 4), (False, False, False, False)),
    ]

    with pytest.raises(InvalidTrakePredictionError):
        export_official_trake_submission_csv(rows, tmp_path / "out.csv")


# -- defense-in-depth: the official exporter must not trust its caller ---------
#
# Found in PR #11 review: export_official_trake_submission_csv() only checked
# row count and cross-row marker-count consistency, never calling
# validate_trake_prediction() on each row -- so a row with a blank video_id
# and out-of-order frames (9, 3) was written through unchanged as ",9,3\n",
# an invalid organizer submission with no error raised anywhere. Reproduced
# first, then fixed by validating every row inside the exporter itself.


def test_official_csv_rejects_a_blank_video_id(tmp_path: Path) -> None:
    rows = [TrakePredictionRow("", (3, 9), (False, False))]

    with pytest.raises(InvalidTrakePredictionError):
        export_official_trake_submission_csv(rows, tmp_path / "out.csv")


def test_official_csv_rejects_out_of_order_frames(tmp_path: Path) -> None:
    """The exact case found during PR #11 review: video_id='' with
    frames (9, 3) was silently written as ',9,3'."""
    rows = [TrakePredictionRow("L21_V001", (9, 3), (False, False))]

    with pytest.raises(InvalidTrakePredictionError):
        export_official_trake_submission_csv(rows, tmp_path / "out.csv")


def test_official_csv_rejects_a_negative_frame(tmp_path: Path) -> None:
    rows = [TrakePredictionRow("L21_V001", (-1, 9), (False, False))]

    with pytest.raises(InvalidTrakePredictionError):
        export_official_trake_submission_csv(rows, tmp_path / "out.csv")


def test_official_csv_rejects_video_id_with_mp4_suffix(tmp_path: Path) -> None:
    rows = [TrakePredictionRow("L21_V001.mp4", (3, 9), (False, False))]

    with pytest.raises(InvalidTrakePredictionError):
        export_official_trake_submission_csv(rows, tmp_path / "out.csv")


def test_official_csv_never_writes_a_partial_file_when_a_later_row_is_invalid(
    tmp_path: Path,
) -> None:
    """Validation must happen before any bytes are written -- a bad row
    anywhere in the batch must not leave a truncated, half-written CSV on
    disk that could be mistaken for a valid (short) submission."""
    rows = [
        TrakePredictionRow("L21_V001", (100, 200), (False, False)),  # valid
        TrakePredictionRow("", (300, 400), (False, False)),  # invalid: blank video_id
    ]
    path = tmp_path / "out.csv"

    with pytest.raises(InvalidTrakePredictionError):
        export_official_trake_submission_csv(rows, path)

    assert not path.exists()
