"""Freezing the operator's round-1 verification into a file.

This runs once and its output becomes the yardstick every later
measurement is read against, so a mistake here is invisible and permanent:
a ground truth holding `" 6613"` instead of `6613` would report every
future run as a miss and look like a retrieval regression.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _module():
    path = REPOSITORY_ROOT / "scripts" / "build_round1_ground_truth.py"
    spec = importlib.util.spec_from_file_location("_build_round1_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pack(tmp_path):
    """A miniature version of the real folders."""
    submissions = tmp_path / "submissions"
    queries = tmp_path / "queries"
    submissions.mkdir()
    queries.mkdir()
    return submissions, queries


def _query_files(queries: Path, name: str, *, english: str | None = "en text") -> None:
    (queries / f"{name}.txt").write_text("câu tiếng Việt", encoding="utf-8")
    if english is not None:
        (queries / f"{name}.en.txt").write_text(english, encoding="utf-8")


def test_the_verified_answer_is_row_one(pack) -> None:
    """`:top` in the interactive runner lifts a frame the person found by
    eye to rank 1, and three round-1 answers name frames that are not in
    the keyframe mapping at all -- retrieval could not have produced them."""
    submissions, queries = pack
    (submissions / "query-p1-1-kis.csv").write_text(
        "L30_V046,6613\nL30_V048,999\n", encoding="utf-8"
    )
    _query_files(queries, "query-p1-1-kis")

    payload = _module().build(submissions, queries, (1,))

    entry = payload["queries"]["query-p1-1-kis"]
    assert (entry["video_id"], entry["frame_idx"]) == ("L30_V046", 6613)


def test_an_excel_byte_order_mark_never_reaches_the_ground_truth(pack) -> None:
    """Five round-1 files carried one, and it would make the first
    video_id `\\ufeffL26_V270`, which matches nothing."""
    submissions, queries = pack
    (submissions / "query-p1-3-qa.csv").write_bytes(
        "﻿L26_V270,3072,1.2\n".encode("utf-8")
    )
    _query_files(queries, "query-p1-3-qa")

    entry = _module().build(submissions, queries, (3,))["queries"]["query-p1-3-qa"]

    assert entry["video_id"] == "L26_V270"


def test_spaces_round_the_commas_are_stripped(pack) -> None:
    """Four round-1 files have them, and the organizer does not trim, so
    `" 6613"` would be carried into the yardstick verbatim."""
    submissions, queries = pack
    (submissions / "query-p1-22-kis.csv").write_text(
        "L25_V041, 17069 \n", encoding="utf-8"
    )
    _query_files(queries, "query-p1-22-kis")

    entry = _module().build(submissions, queries, (22,))["queries"]["query-p1-22-kis"]

    assert (entry["video_id"], entry["frame_idx"]) == ("L25_V041", 17069)


def test_a_qa_answer_is_kept(pack) -> None:
    submissions, queries = pack
    (submissions / "query-p1-17-qa.csv").write_text(
        "L21_V006,3924,Cổng Trời Mường Lát\n", encoding="utf-8"
    )
    _query_files(queries, "query-p1-17-qa")

    entry = _module().build(submissions, queries, (17,))["queries"]["query-p1-17-qa"]

    assert entry["answer"] == "Cổng Trời Mường Lát"
    assert entry["kind"] == "qa"


def test_a_trake_chain_is_kept_whole(pack) -> None:
    """A TRAKE answer is one frame per sub-event. Keeping only the first
    would throw away two thirds of what the person verified."""
    submissions, queries = pack
    (submissions / "query-p1-16-trake.csv").write_text(
        "L24_V035,3800,4104,5049\n", encoding="utf-8"
    )
    _query_files(queries, "query-p1-16-trake", english=None)

    entry = _module().build(submissions, queries, (16,))["queries"]["query-p1-16-trake"]

    assert entry["frames"] == [3800, 4104, 5049]
    assert entry["frame_idx"] == 3800


def test_the_query_text_is_copied_in_not_referenced(pack) -> None:
    """`artifacts/` is gitignored, so a ground truth pointing at
    `artifacts/queries/` would be unusable to a teammate who checks the
    repo out."""
    submissions, queries = pack
    (submissions / "query-p1-1-kis.csv").write_text("L1,1\n", encoding="utf-8")
    _query_files(queries, "query-p1-1-kis", english="a group exercising")

    entry = _module().build(submissions, queries, (1,))["queries"]["query-p1-1-kis"]

    assert entry["query_vi"] == "câu tiếng Việt"
    assert entry["query_en"] == "a group exercising"


def test_a_query_with_no_english_companion_omits_the_field(pack) -> None:
    submissions, queries = pack
    (submissions / "query-p1-16-trake.csv").write_text("L1,1,2,3\n", encoding="utf-8")
    _query_files(queries, "query-p1-16-trake", english=None)

    entry = _module().build(submissions, queries, (16,))["queries"]["query-p1-16-trake"]

    assert "query_en" not in entry


def test_a_review_file_is_not_mistaken_for_the_submission(pack) -> None:
    """`*.review.csv` sits beside the submission, has a header and six
    columns, and would put the string `rank` into the ground truth."""
    submissions, queries = pack
    (submissions / "query-p1-3-qa.csv").write_text("L26_V270,3072,1.2\n", encoding="utf-8")
    (submissions / "query-p1-3-qa.review.csv").write_text(
        "rank,video_id,frame_idx,answer,score,image_path\n", encoding="utf-8"
    )
    _query_files(queries, "query-p1-3-qa")

    entry = _module().build(submissions, queries, (3,))["queries"]["query-p1-3-qa"]

    assert entry["video_id"] == "L26_V270"


def test_a_missing_submission_stops_rather_than_skipping(pack) -> None:
    """Silently dropping a query would shrink the yardstick without saying
    so, and every later score would be over a different set."""
    submissions, queries = pack

    with pytest.raises(SystemExit, match="câu 7"):
        _module().build(submissions, queries, (7,))


def test_an_empty_submission_stops(pack) -> None:
    """An empty file is a legal submission -- it keeps the upload slot --
    but it holds no verified answer to freeze."""
    submissions, queries = pack
    (submissions / "query-p1-8-kis.csv").write_text("", encoding="utf-8")
    _query_files(queries, "query-p1-8-kis")

    with pytest.raises(SystemExit, match="rỗng"):
        _module().build(submissions, queries, (8,))


def test_the_default_list_is_the_seventeen_the_operator_confirmed() -> None:
    module = _module()

    assert len(module.VERIFIED_ROUND_1) == 17
    assert 8 not in module.VERIFIED_ROUND_1  # a query that was NOT verified


# -- merge_queries: don't revert a correction this script did not make ---


def test_a_query_verified_by_this_scripts_own_route_is_overwritten() -> None:
    """Re-running the ordinary way -- e.g. the operator fixed a typo in a
    round-1 submission and wants the ground truth to pick it up -- must
    still work without needing --force."""
    module = _module()
    existing = {"query-p1-1-kis": {"video_id": "OLD", "verified_by": "human"}}
    new = {"query-p1-1-kis": {"video_id": "NEW", "verified_by": "human"}}

    merged, skipped = module.merge_queries(existing, new, force=False)

    assert merged["query-p1-1-kis"]["video_id"] == "NEW"
    assert skipped == []


def test_a_query_verified_through_sublan1_is_not_silently_reverted() -> None:
    """This is the failure this function exists to prevent: rerunning the
    original 17-query build must not throw away a correction born from a
    real cross-check disagreement with another team's submission."""
    module = _module()
    existing = {
        "query-p1-2-kis": {
            "video_id": "L28_V018", "verified_by": "human (đối chiếu sublan1)"
        }
    }
    new = {"query-p1-2-kis": {"video_id": "L30_V048", "verified_by": "human"}}

    merged, skipped = module.merge_queries(existing, new, force=False)

    assert merged["query-p1-2-kis"]["video_id"] == "L28_V018"
    assert skipped == [("query-p1-2-kis", "human (đối chiếu sublan1)")]


def test_force_overrides_the_protection() -> None:
    module = _module()
    existing = {
        "query-p1-2-kis": {
            "video_id": "L28_V018", "verified_by": "human (đối chiếu sublan1)"
        }
    }
    new = {"query-p1-2-kis": {"video_id": "L30_V048", "verified_by": "human"}}

    merged, skipped = module.merge_queries(existing, new, force=True)

    assert merged["query-p1-2-kis"]["video_id"] == "L30_V048"
    assert skipped == []


def test_an_entry_from_another_route_absent_from_the_rebuild_survives() -> None:
    """The 8 queries added purely from sublan1 (never in VERIFIED_ROUND_1,
    so never in a fresh build's output) must not disappear just because
    they are not part of what this script knows how to rebuild."""
    module = _module()
    existing = {"query-p1-8-kis": {"video_id": "L26_V171", "verified_by": "human (đối chiếu sublan1)"}}

    merged, skipped = module.merge_queries(existing, new_queries={}, force=False)

    assert merged == existing
    assert skipped == []
