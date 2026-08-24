"""The round-1 verified set is the only yardstick we have.

The organizer published no scores, so if this scorer is wrong there is
nothing to catch it -- a round-2 change would be judged by a broken ruler.
Every test below pins a decision that would silently corrupt the reading.
"""
from __future__ import annotations

import json

import pytest

from mvr.evaluation.verified_set import (
    WHOLE_VIDEO,
    VerifiedQuery,
    format_report,
    load_verified_set,
    score_query,
    score_set,
)


def _query(**overrides) -> VerifiedQuery:
    payload = {
        "name": "query-p1-1-kis",
        "kind": "kis",
        "video_id": "L30_V046",
        "frame_idx": 6613,
        "query_vi": "một nhóm người tập thể dục",
        "query_en": "a group of people exercising",
    }
    payload.update(overrides)
    return VerifiedQuery(**payload)


def _write(tmp_path, queries: dict):
    path = tmp_path / "gt.json"
    path.write_text(
        json.dumps({"pack": "test", "queries": queries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# -- video level: what the ground truth actually knows -------------------


def test_the_right_video_at_any_frame_scores_a_full_point() -> None:
    """The verified frame is ONE frame a person saw, not the organizer's
    interval, which was never published. So a hit anywhere in the right
    video counts -- scoring the frame against an interval invented here
    would measure the invention."""
    query = _query()

    result = score_query(query, [("L30_V046", 999999)])

    assert result.final_score == 1.0
    assert result.video_rank == 1


def test_the_wrong_video_scores_nothing_however_close_the_frame() -> None:
    query = _query()

    result = score_query(query, [("L30_V047", 6613)])

    assert result.final_score == 0.0
    assert result.video_rank is None


def test_the_video_rank_follows_the_official_cutoffs() -> None:
    """Rank 6 misses R@1 and R@5, takes R@20/50/100: 3/5 = 0.6."""
    query = _query()
    rows = [("L00_V000", i) for i in range(5)] + [("L30_V046", 6613)]

    result = score_query(query, rows)

    assert result.video_rank == 6
    assert result.recall_at_k == {1: 0.0, 5: 0.0, 20: 1.0, 50: 1.0, 100: 1.0}
    assert result.final_score == pytest.approx(0.6)


def test_a_video_past_rank_100_scores_zero_but_still_reports_its_rank() -> None:
    """'rank 137' and 'never found at all' are different failures with
    different fixes, and a bare 0.0000 cannot tell them apart."""
    query = _query()
    rows = [("L00_V000", i) for i in range(136)] + [("L30_V046", 6613)]

    result = score_query(query, rows)

    assert result.final_score == 0.0
    assert result.video_rank == 137


def test_an_empty_submission_scores_zero_rather_than_raising() -> None:
    assert score_query(_query(), []).final_score == 0.0


# -- the frame-level proxy -----------------------------------------------


def test_the_frame_window_narrows_the_interval_round_the_verified_frame() -> None:
    query = _query()

    inside = score_query(query, [("L30_V046", 6650)], frame_window=50)
    outside = score_query(query, [("L30_V046", 6650)], frame_window=10)

    assert inside.final_score == 1.0
    assert outside.final_score == 0.0
    # Absent from the top 100 by the scoring rule, but the video IS there,
    # which is the number a person needs to read.
    assert outside.video_rank == 1


def test_the_frame_window_never_reaches_below_zero() -> None:
    """A verified frame near the start of a video would otherwise build an
    interval with a negative lower bound."""
    query = _query(frame_idx=10)

    assert score_query(query, [("L30_V046", 0)], frame_window=100).final_score == 1.0


def test_a_negative_frame_window_is_refused() -> None:
    with pytest.raises(ValueError, match="frame_window"):
        _query().ground_truth(frame_window=-1)


def test_no_window_means_the_whole_video() -> None:
    ground_truth = _query().ground_truth()

    assert (ground_truth.start_frame, ground_truth.end_frame) == (0, WHOLE_VIDEO)


# -- query text ----------------------------------------------------------


def test_english_is_used_when_a_companion_exists() -> None:
    assert _query().text(language="en") == "a group of people exercising"


def test_a_missing_english_companion_falls_back_to_vietnamese() -> None:
    """Query 16 has no English text. Refusing to score it would hide the
    query; falling back scores it honestly at what it actually retrieves."""
    query = _query(query_en=None)

    assert query.text(language="en") == query.query_vi


def test_an_unknown_language_is_refused() -> None:
    with pytest.raises(ValueError, match="'vi' or 'en'"):
        _query().text(language="fr")


# -- loading -------------------------------------------------------------


def test_the_kinds_filter_keeps_trake_out_by_default_only_when_asked(tmp_path) -> None:
    """TRAKE retrieval belongs to another owner and a chain of sub-event
    frames is not a KIS ranked list, so measuring this system excludes it
    -- but the entry is still in the file, because its video is verified."""
    path = _write(
        tmp_path,
        {
            "query-p1-1-kis": {"kind": "kis", "video_id": "L1", "frame_idx": 1, "query_vi": "a"},
            "query-p1-16-trake": {
                "kind": "trake", "video_id": "L2", "frame_idx": 2,
                "frames": [2, 3, 4], "query_vi": "b",
            },
        },
    )

    assert len(load_verified_set(path)) == 2
    assert [q.name for q in load_verified_set(path, kinds=("kis", "qa"))] == [
        "query-p1-1-kis"
    ]


def test_a_trake_chain_is_loaded_whole(tmp_path) -> None:
    path = _write(
        tmp_path,
        {
            "query-p1-16-trake": {
                "kind": "trake", "video_id": "L24_V035", "frame_idx": 3800,
                "frames": [3800, 4104, 5049], "query_vi": "b",
            }
        },
    )

    assert load_verified_set(path)[0].frames == (3800, 4104, 5049)


def test_the_real_ground_truth_file_loads() -> None:
    """The committed file is the yardstick; a typo in it breaks every
    measurement downstream and nothing else would notice."""
    queries = load_verified_set()

    assert len(queries) == 17
    assert {q.kind for q in queries} == {"kis", "qa", "trake"}
    assert all(q.video_id and q.query_vi for q in queries)


# -- aggregation ---------------------------------------------------------


def test_the_mean_is_taken_over_every_query_including_the_zeroes() -> None:
    """Averaging only the queries that scored would report a system that
    finds one video out of sixteen as perfect."""
    hit = score_query(_query(), [("L30_V046", 1)])
    miss = score_query(_query(name="b"), [("L00_V000", 1)])

    result = score_set([hit, miss])

    assert result.mean_final_score == pytest.approx(0.5)
    assert result.mean_recall_at(1) == pytest.approx(0.5)
    assert result.found == 1


def test_found_counts_a_video_ranked_past_the_scoring_cutoff() -> None:
    """`found` answers 'is it in the index and reachable at all', which a
    score of 0.0000 cannot."""
    rows = [("L00_V000", i) for i in range(200)] + [("L30_V046", 1)]

    assert score_set([score_query(_query(), rows)]).found == 1


def test_an_empty_set_reports_zero_rather_than_dividing_by_zero() -> None:
    result = score_set([])

    assert result.mean_final_score == 0.0
    assert result.mean_recall_at(1) == 0.0
    assert "không có câu nào" in format_report(result)


def test_the_report_names_every_query_and_the_totals() -> None:
    result = score_set([score_query(_query(), [("L30_V046", 1)])])

    report = format_report(result, label="[en]")

    assert "[en]" in report
    assert "query-p1-1-kis" in report
    assert "Mean Final Score  1.0000" in report
    assert "1/1" in report
