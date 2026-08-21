"""The organizer's submission format, exactly as published.

The repo's only exporter before this was `export_local_verification_csv`,
whose own docstring said it "is explicitly not the official organizer
submission format -- adapt it only once that format is documented". It
wrote a header row and five columns; the organizer wants no header and two.
Submitting it would have scored zero on a working retrieval system.

Rules taken verbatim from the organizer's "Hướng dẫn nộp bài sơ tuyển":

    KIS    <video_id>,<frame_idx>
    Q&A    <video_id>,<frame_idx>,<answer>

    UTF-8, comma delimiter, no header row, at most 100 lines.
    Q&A answers are at most 100 characters.
"""
from __future__ import annotations

import pytest

from mvr.submission.official import (
    MAX_ANSWER_CHARS,
    MAX_ROWS,
    SubmissionFormatError,
    query_number,
    submission_filename,
    write_kis_submission,
    write_qna_submission,
)


def _read(path) -> str:
    return path.read_bytes().decode("utf-8")


# -- Textual KIS -------------------------------------------------------


def test_kis_writes_two_columns_and_no_header(tmp_path) -> None:
    path = tmp_path / "query-1-kis.csv"

    write_kis_submission([("L00_V000", 1234), ("L00_V055", 5555)], path)

    assert _read(path).splitlines() == ["L00_V000,1234", "L00_V055,5555"]


def test_kis_refuses_a_header_by_construction(tmp_path) -> None:
    """The old exporter's first line was `format,query_id,rank,video_id,
    frame_id`. Nothing may reintroduce it."""
    path = tmp_path / "q.csv"

    write_kis_submission([("L00_V000", 1)], path)

    assert not _read(path).startswith("video")
    assert "rank" not in _read(path)


# -- Q&A ---------------------------------------------------------------


def test_qna_leaves_a_simple_answer_unquoted(tmp_path) -> None:
    """The organizer's own example is `L01_V028,3450,5` -- quoting is
    required only for special characters, and their sample file shows the
    plain form."""
    path = tmp_path / "query-3-qa.csv"

    write_qna_submission([("L01_V028", 3450, "5"), ("L02_V011", 1200, "Năm người")], path)

    assert _read(path).splitlines() == ["L01_V028,3450,5", "L02_V011,1200,Năm người"]


def test_qna_quotes_an_answer_containing_a_comma(tmp_path) -> None:
    path = tmp_path / "q.csv"

    write_qna_submission([("L01_V028", 3450, "Có 3 người, bao gồm nam và nữ")], path)

    assert _read(path).splitlines() == [
        'L01_V028,3450,"Có 3 người, bao gồm nam và nữ"'
    ]


def test_qna_escapes_a_quote_by_doubling_it(tmp_path) -> None:
    path = tmp_path / "q.csv"

    write_qna_submission([("L01_V028", 3450, 'Anh ấy nói "Xin chào"')], path)

    assert _read(path).splitlines() == ['L01_V028,3450,"Anh ấy nói ""Xin chào"""']


def test_qna_accepts_an_empty_answer(tmp_path) -> None:
    """Option (c): a frame with no answer yet still scores nothing for the
    answer but keeps the row, and a human can fill the column in later.
    Refusing it would mean submitting nothing at all for that query."""
    path = tmp_path / "q.csv"

    write_qna_submission([("L01_V028", 3450, "")], path)

    assert _read(path).splitlines() == ["L01_V028,3450,"]


def test_qna_refuses_an_answer_over_the_character_limit(tmp_path) -> None:
    with pytest.raises(SubmissionFormatError, match="100 ký tự"):
        write_qna_submission(
            [("L01_V028", 3450, "x" * (MAX_ANSWER_CHARS + 1))], tmp_path / "q.csv"
        )


def test_qna_counts_characters_not_bytes(tmp_path) -> None:
    """Vietnamese is multi-byte in UTF-8. Counting bytes would reject a
    perfectly legal 60-character answer."""
    answer = "à" * 100

    write_qna_submission([("L01_V028", 3450, answer)], tmp_path / "q.csv")


# -- rules shared by every track ---------------------------------------


@pytest.mark.parametrize(
    "writer,rows",
    [
        (write_kis_submission, [("L00_V000", i) for i in range(MAX_ROWS + 20)]),
        (write_qna_submission, [("L00_V000", i, "a") for i in range(MAX_ROWS + 20)]),
    ],
)
def test_at_most_one_hundred_rows_are_written(writer, rows, tmp_path) -> None:
    """Truncate rather than raise. Over the limit the organizer rejects the
    file, and a rejected file scores zero -- while the first 100 ranked
    answers are exactly what the scoring rule reads anyway."""
    path = tmp_path / "q.csv"

    writer(rows, path)

    assert len(_read(path).splitlines()) == MAX_ROWS


def test_a_negative_frame_index_is_refused(tmp_path) -> None:
    with pytest.raises(SubmissionFormatError, match="frame_idx"):
        write_kis_submission([("L00_V000", -1)], tmp_path / "q.csv")


def test_a_blank_video_id_is_refused(tmp_path) -> None:
    with pytest.raises(SubmissionFormatError, match="video_id"):
        write_kis_submission([("  ", 10)], tmp_path / "q.csv")


def test_the_file_is_utf8_without_a_byte_order_mark(tmp_path) -> None:
    """Excel writes a BOM and the organizer asks for plain UTF-8. A leading
    \\ufeff would corrupt the first video_id of every submission."""
    path = tmp_path / "q.csv"

    write_qna_submission([("L01_V028", 3450, "Màu đỏ")], path)

    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8").startswith("L01_V028")


def test_an_empty_prediction_list_still_writes_a_file(tmp_path) -> None:
    """A retrieval failure must not also lose the submission slot: an empty
    file is valid and scores zero, while a missing file cannot be uploaded
    at all."""
    path = tmp_path / "q.csv"

    write_kis_submission([], path)

    assert path.is_file()
    assert path.read_bytes() == b""


# -- File naming -------------------------------------------------------
#
# The organizer's "Cấu trúc thư mục yêu cầu" shows a plain sequential
# number, not the package prefix their query files carry:
#
#     submission/
#     ├── query-1-kis.csv
#     ├── query-3-qa.csv
#     └── query-4-trake.csv
#
# The query pack is named `query-p1-23-kis.txt`, so the `p1-` has to be
# dropped on the way out or the submission is named something the
# organizer never asked for.


def test_submission_filename_uses_a_plain_number() -> None:
    assert submission_filename(23, "kis") == "query-23-kis.csv"
    assert submission_filename(3, "qa") == "query-3-qa.csv"
    assert submission_filename(4, "trake") == "query-4-trake.csv"


def test_submission_filename_accepts_a_numeric_string() -> None:
    assert submission_filename("23", "kis") == "query-23-kis.csv"


def test_submission_filename_rejects_an_unknown_kind() -> None:
    with pytest.raises(SubmissionFormatError, match="dạng truy vấn"):
        submission_filename(1, "kís")


def test_submission_filename_rejects_a_non_numeric_stt() -> None:
    # `p1-23` is the pack's name for the query, not its number. Letting it
    # through would write `query-p1-23-kis.csv`, which is the exact name
    # the organizer's directory listing does not contain.
    with pytest.raises(SubmissionFormatError, match="stt"):
        submission_filename("p1-23", "kis")


def test_query_number_reads_the_number_out_of_a_pack_filename() -> None:
    assert query_number("query-p1-23-kis.txt") == 23
    assert query_number("query-4-trake.txt") == 4


def test_query_number_rejects_a_name_it_cannot_parse() -> None:
    with pytest.raises(SubmissionFormatError):
        query_number("notes.txt")
