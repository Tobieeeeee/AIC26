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
