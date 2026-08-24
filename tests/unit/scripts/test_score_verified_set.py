"""The before/after comparison that decides whether a round-2 change ships.

The retrieval half needs a 177,321-vector index and a CLIP encoder. These
are the pure parts: reading a submission back, and saying what moved
between two runs.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _module():
    path = REPOSITORY_ROOT / "scripts" / "score_verified_set.py"
    spec = importlib.util.spec_from_file_location("_score_verified_set", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_submission_is_read_back_in_rank_order(tmp_path) -> None:
    path = tmp_path / "query-p1-1-kis.csv"
    path.write_text("L30_V046,6613\nL30_V048,8165\n", encoding="utf-8")

    assert _module().rows_from_submission(path) == [
        ("L30_V046", 6613),
        ("L30_V048", 8165),
    ]


def test_a_bom_and_stray_spaces_do_not_break_the_reading(tmp_path) -> None:
    """A third of the round-1 files had one or the other, and an unstripped
    `" 6613"` would raise on int() or compare unequal to the ground truth."""
    path = tmp_path / "query-p1-22-kis.csv"
    path.write_bytes("﻿L25_V041, 17069 \n".encode("utf-8"))

    assert _module().rows_from_submission(path) == [("L25_V041", 17069)]


def test_a_quoted_qa_answer_holding_a_comma_is_not_split_into_columns(tmp_path) -> None:
    path = tmp_path / "query-p1-3-qa.csv"
    path.write_text('L26_V270,3072,"Có 3 người, gồm nam"\n', encoding="utf-8")

    assert _module().rows_from_submission(path) == [("L26_V270", 3072)]


def test_the_delta_reports_only_the_queries_that_moved(capsys) -> None:
    """A table of seventeen unchanged rows buries the two that changed,
    which are the entire reason for running it."""
    module = _module()
    from mvr.evaluation.verified_set import VerifiedQuery, score_query

    def query(name, video):
        return VerifiedQuery(
            name=name, kind="kis", video_id=video, frame_idx=1, query_vi="x"
        )

    current = [
        score_query(query("a", "L1"), [("L1", 1)]),          # 1.0
        score_query(query("b", "L2"), [("L9", 1), ("L2", 1)]),  # 0.8
    ]
    previous = {
        "label": "trước",
        "queries": {
            "a": {"video_rank": 1, "final_score": 1.0},
            "b": {"video_rank": 20, "final_score": 0.4},
        },
    }

    module.print_delta(current, previous)

    printed = capsys.readouterr().out
    assert "b" in printed and "+0.4000" in printed
    assert "1.0000 1.0000" not in printed  # `a` did not move


def test_a_query_absent_from_the_earlier_run_is_marked_new(capsys) -> None:
    module = _module()
    from mvr.evaluation.verified_set import VerifiedQuery, score_query

    current = [
        score_query(
            VerifiedQuery(name="c", kind="kis", video_id="L1", frame_idx=1, query_vi="x"),
            [("L1", 1)],
        )
    ]

    module.print_delta(current, {"label": "trước", "queries": {}})

    assert "(mới)" in capsys.readouterr().out


def test_a_run_with_nothing_changed_says_so(capsys) -> None:
    module = _module()
    from mvr.evaluation.verified_set import VerifiedQuery, score_query

    current = [
        score_query(
            VerifiedQuery(name="a", kind="kis", video_id="L1", frame_idx=1, query_vi="x"),
            [("L1", 1)],
        )
    ]
    previous = {"label": "trước", "queries": {"a": {"video_rank": 1, "final_score": 1.0}}}

    module.print_delta(current, previous)

    assert "không câu nào đổi điểm" in capsys.readouterr().out


def test_a_saved_run_round_trips_through_the_delta(tmp_path) -> None:
    """`--save` then `--against` is the whole workflow; a payload the
    reader cannot parse would make every comparison report '(mới)'."""
    module = _module()
    from mvr.evaluation.verified_set import VerifiedQuery, score_query

    results = [
        score_query(
            VerifiedQuery(name="a", kind="kis", video_id="L1", frame_idx=1, query_vi="x"),
            [("L1", 1)],
        )
    ]
    path = tmp_path / "before.json"
    path.write_text(
        json.dumps(module.to_payload(results, label="en"), ensure_ascii=False),
        encoding="utf-8",
    )

    reloaded = json.loads(path.read_text(encoding="utf-8"))

    assert reloaded["label"] == "en"
    assert reloaded["queries"]["a"] == {"video_rank": 1, "final_score": 1.0}


def test_trake_is_excluded_by_default_and_the_default_language_is_english() -> None:
    """Vietnamese measured 0.0125 against 0.8375 on this very set, and
    TRAKE retrieval belongs to another owner."""
    args = _module().build_parser().parse_args([])

    assert args.kinds == ["kis", "qa"]
    assert args.language == "en"
