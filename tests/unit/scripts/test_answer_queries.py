"""Query-file dispatch and the review round trip.

The retrieval half needs a 166,550-vector index and a CLIP encoder, so it
is exercised by running the script. These are the pure parts, which are
where a silent mistake would be most expensive: sending a Q&A query down
the KIS path writes a two-column file for a three-column track.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _module():
    path = REPOSITORY_ROOT / "scripts" / "answer_queries.py"
    spec = importlib.util.spec_from_file_location("_answer_queries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name,expected",
    [
        ("query-1-kis.txt", "kis"),
        ("query-3-qa.txt", "qa"),
        ("query-4-trake.txt", "trake"),
        ("QUERY-2-KIS.txt", "kis"),
        ("query_5_qa.txt", "qa"),
        ("readme.txt", None),
        ("kis-query-1.txt", None),
    ],
)
def test_query_type_comes_from_the_filename_suffix(name, expected) -> None:
    """The organizer's convention is the suffix, so `kis-query-1.txt` is
    deliberately NOT kis -- guessing from anywhere in the name would send a
    Q&A query down the KIS path and write the wrong column count."""
    assert _module().classify(Path(name)) == expected


def test_a_filled_in_review_file_becomes_a_submission(tmp_path) -> None:
    review = tmp_path / "query-3-qa.review.csv"
    review.write_text(
        "rank,video_id,frame_idx,answer,score,image_path\n"
        "1,L01_V028,3450,Năm người,0.31,/data/001.jpg\n"
        "2,L02_V011,1200,,0.29,/data/002.jpg\n",
        encoding="utf-8",
    )
    output = tmp_path / "query-3-qa.csv"

    _module().submission_from_review(review, output)

    assert output.read_bytes().decode("utf-8").splitlines() == [
        "L01_V028,3450,Năm người",
        "L02_V011,1200,",
    ]


def test_an_answer_with_a_comma_survives_the_review_round_trip(tmp_path) -> None:
    """A reviewer typing a natural sentence produces exactly the case that
    breaks a naive writer: the comma must end up quoted, not splitting the
    row into four columns."""
    review = tmp_path / "r.csv"
    review.write_text(
        "rank,video_id,frame_idx,answer,score,image_path\n"
        '1,L01_V028,3450,"Có 3 người, gồm nam và nữ",0.3,x\n',
        encoding="utf-8",
    )

    _module().submission_from_review(review, tmp_path / "out.csv")

    assert (tmp_path / "out.csv").read_text(encoding="utf-8").strip() == (
        'L01_V028,3450,"Có 3 người, gồm nam và nữ"'
    )


def test_a_review_file_saved_by_excel_is_still_readable(tmp_path) -> None:
    """Excel writes a byte-order mark. Reading it as plain utf-8 turns the
    first column name into an unrecognisable one and loses every row."""
    review = tmp_path / "r.csv"
    review.write_bytes(
        (
            "﻿rank,video_id,frame_idx,answer,score,image_path\n"
            "1,L01_V028,3450,5,0.3,x\n"
        ).encode("utf-8")
    )

    _module().submission_from_review(review, tmp_path / "out.csv")

    assert (tmp_path / "out.csv").read_text(encoding="utf-8").strip() == "L01_V028,3450,5"


# -- Submission naming in interactive mode ----------------------------
#
# The submission carries the query file's own name, prefix included.


def test_w_accepts_a_bare_number() -> None:
    assert _module().resolve_output_name("23", "kis") == "query-23-kis.csv"


def test_w_keeps_the_packs_own_name() -> None:
    assert (
        _module().resolve_output_name("query-p1-23-kis", "kis")
        == "query-p1-23-kis.csv"
    )


def test_w_accepts_a_bare_id_carrying_the_batch_prefix() -> None:
    assert _module().resolve_output_name("p1-23", "kis") == "query-p1-23-kis.csv"


def test_w_rejects_something_that_is_not_a_name() -> None:
    from mvr.submission.official import SubmissionFormatError

    with pytest.raises(SubmissionFormatError):
        _module().resolve_output_name("a/b", "kis")


# -- Manual top-1 insertion -------------------------------------------


def _row(video_id, frame_idx, rank=1):
    return {
        "rank": rank,
        "video_id": video_id,
        "frame_idx": frame_idx,
        "score": 0.5,
        "image_path": "x.jpg",
    }


def test_manual_row_goes_to_rank_one() -> None:
    rows = [_row("L21_V002", 90, 1), _row("L21_V003", 120, 2)]

    out = _module().insert_top_row(rows, "L21_V001", 5760)

    assert (out[0]["video_id"], out[0]["frame_idx"]) == ("L21_V001", 5760)
    assert [r["rank"] for r in out] == [1, 2, 3]


def test_manual_row_already_present_is_moved_not_duplicated() -> None:
    # A duplicate line wastes one of the 100 slots and scores nothing.
    rows = [_row("L21_V002", 90, 1), _row("L21_V001", 5760, 2)]

    out = _module().insert_top_row(rows, "L21_V001", 5760)

    assert len(out) == 2
    assert (out[0]["video_id"], out[0]["frame_idx"]) == ("L21_V001", 5760)


def test_manual_row_works_from_an_empty_list() -> None:
    # The operator already knows the answer and does not need to search.
    out = _module().insert_top_row([], "L21_V001", 0)

    assert len(out) == 1 and out[0]["frame_idx"] == 0


def test_manual_row_keeps_its_own_image_path_when_already_present() -> None:
    # A row already in the list carries a real image_path from search;
    # promoting it to rank 1 must not lose that path.
    rows = [
        _row("L21_V002", 90, 1),
        {**_row("L21_V001", 5760, 2), "image_path": "L21_V001.jpg"},
    ]

    out = _module().insert_top_row(rows, "L21_V001", 5760)

    assert out[0]["image_path"] == "L21_V001.jpg"


# -- Shared write path for :w and :top ---------------------------------


def test_write_submission_kis_writes_two_columns(tmp_path) -> None:
    m = _module()
    rows = [_row("L21_V001", 5760, 1)]

    m._write_submission(rows, "kis", tmp_path / "query-23-kis.csv")

    assert (tmp_path / "query-23-kis.csv").read_text(encoding="utf-8").strip() == (
        "L21_V001,5760"
    )


def test_write_submission_qa_also_writes_a_review_file(tmp_path) -> None:
    m = _module()
    rows = [_row("L30_V072", 1776, 1)]

    m._write_submission(rows, "qa", tmp_path / "query-15-qa.csv")

    assert (tmp_path / "query-15-qa.csv").exists()
    assert (tmp_path / "query-15-qa.review.csv").exists()


# -- Batch-mode filename (main(), not interactive) ----------------------


def test_batch_output_filename_keeps_the_query_files_name() -> None:
    m = _module()
    assert (
        m.batch_output_filename(Path("query-p1-23-kis.txt"), "kis")
        == "query-p1-23-kis.csv"
    )


def test_batch_output_filename_falls_back_for_an_unparseable_name() -> None:
    # An oddly-named query file must cost that one query, not crash
    # everything still queued behind it in the batch.
    m = _module()
    assert m.batch_output_filename(Path("notes-kis.txt"), "kis") == "notes-kis.csv"
