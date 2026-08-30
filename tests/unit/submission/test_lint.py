"""Catch a broken submission before it is uploaded, not after.

Every check here is a defect measured in the round-1 files, which were
hand-edited after the writer produced them:

    BOM                query-3-qa, query-9-qa, query-15-qa,
                       query-16-trake, query-17-qa   (Excel adds it)
    spaces round the   query-1-kis  22/22 rows
    comma              query-6-kis  100/100 rows
                       query-22-kis  8/8, query-25-kis  2/2

The organizer asks for `Encoding: UTF-8` and states plainly that
"Khoảng trắng đầu/cuối: Được giữ nguyên, không tự động trim", so
`L30_V046, 6613` risks being read as the string `" 6613"` rather than the
integer 6613. Neither defect makes the file look wrong when opened.
"""
from __future__ import annotations

from pathlib import Path

from mvr.submission.lint import (
    lint_file,
    lint_directory,
    repair_file,
)


def _write(path: Path, text: str, *, bom: bool = False) -> Path:
    raw = text.encode("utf-8")
    path.write_bytes(b"\xef\xbb\xbf" + raw if bom else raw)
    return path


# -- what the round-1 files actually got wrong -------------------------


def test_a_byte_order_mark_is_reported(tmp_path) -> None:
    path = _write(tmp_path / "query-3-qa.csv", "L26_V270,3072,1.2\n", bom=True)

    problems = lint_file(path)

    assert any("BOM" in p for p in problems)


def test_a_space_after_the_comma_is_reported(tmp_path) -> None:
    """`L30_V046, 6613` -- the organizer does not trim, so the second field
    may be compared as a string with a leading space."""
    path = _write(tmp_path / "query-1-kis.csv", "L30_V046, 6613\n")

    problems = lint_file(path)

    assert any("khoảng trắng" in p for p in problems)


def test_a_trailing_space_is_reported(tmp_path) -> None:
    path = _write(tmp_path / "query-22-kis.csv", "L25_V041,17069 \n")

    problems = lint_file(path)

    assert any("khoảng trắng" in p for p in problems)


# -- the rest of the organizer's contract ------------------------------


def test_a_header_row_is_reported(tmp_path) -> None:
    """The repo's old exporter wrote one, and a submission with a header
    loses its first answer to a line the scorer cannot parse."""
    path = _write(tmp_path / "query-1-kis.csv", "video_id,frame_idx\nL30_V046,6613\n")

    problems = lint_file(path)

    assert any("header" in p for p in problems)


def test_the_wrong_column_count_for_the_track_is_reported(tmp_path) -> None:
    path = _write(tmp_path / "query-1-kis.csv", "L30_V046,6613,extra\n")

    problems = lint_file(path)

    assert any("2 cột" in p for p in problems)


def test_a_qa_file_with_two_columns_is_reported(tmp_path) -> None:
    path = _write(tmp_path / "query-3-qa.csv", "L26_V270,3072\n")

    problems = lint_file(path)

    assert any("3 cột" in p for p in problems)


def test_more_than_one_hundred_rows_is_reported(tmp_path) -> None:
    path = _write(tmp_path / "q.csv", "".join(f"L00_V000,{i}\n" for i in range(101)))

    problems = lint_file(path, kind="kis")

    assert any("100 dòng" in p for p in problems)


def test_a_non_integer_frame_index_is_reported(tmp_path) -> None:
    path = _write(tmp_path / "query-1-kis.csv", "L30_V046,6613.5\n")

    problems = lint_file(path)

    assert any("frame_idx" in p for p in problems)


def test_a_duplicate_prediction_is_reported(tmp_path) -> None:
    """A repeated line wastes one of the 100 slots and can never add score."""
    path = _write(
        tmp_path / "query-1-kis.csv", "L30_V046,6613\nL30_V046,6613\n"
    )

    problems = lint_file(path)

    assert any("trùng" in p for p in problems)


def test_every_duplicate_is_reported_at_once(tmp_path) -> None:
    """Reporting only the first turned fixing them into whack-a-mole: three
    round-1 files were cleaned, relinted, and each surfaced a different pair
    that had been there all along."""
    path = _write(
        tmp_path / "query-1-kis.csv",
        "L1,1\nL2,2\nL1,1\nL3,3\nL2,2\nL4,4\nL1,1\n",
    )

    problems = lint_file(path)

    report = next(p for p in problems if "trùng" in p)
    assert report.startswith("3 dòng trùng")
    assert "dòng 3 (giống dòng 1)" in report
    assert "dòng 5 (giống dòng 2)" in report
    assert "dòng 7 (giống dòng 1)" in report


def test_a_long_run_of_duplicates_is_summarised(tmp_path) -> None:
    """A file that is mostly duplicates should say how many, not print a
    dozen lines the reader has to count."""
    path = _write(tmp_path / "query-1-kis.csv", "L1,1\n" * 12)

    report = next(p for p in lint_file(path) if "trùng" in p)

    assert report.startswith("11 dòng trùng")
    assert "và 6 dòng nữa" in report


def test_an_answer_over_the_character_limit_is_reported(tmp_path) -> None:
    path = _write(
        tmp_path / "query-3-qa.csv", f"L26_V270,3072,{'x' * 101}\n"
    )

    problems = lint_file(path)

    assert any("100 ký tự" in p for p in problems)


def test_the_packs_own_name_is_accepted(tmp_path) -> None:
    """`query-p1-23-kis.csv` IS the required name: the submission carries
    the query file's name, and the pack prefixes its queries with the
    batch. An earlier reading renumbered these and was wrong."""
    path = _write(tmp_path / "query-p1-23-kis.csv", "L30_V046,6613\n")

    assert lint_file(path) == []


def test_a_name_with_no_track_suffix_is_reported(tmp_path) -> None:
    """Without `-kis`/`-qa`/`-trake` the scorer cannot tell which format to
    parse the file as."""
    path = _write(tmp_path / "ket-qua.csv", "L30_V046,6613\n")

    problems = lint_file(path)

    assert any("tên file" in p for p in problems)


# -- a clean file --------------------------------------------------------


def test_a_correct_file_reports_nothing(tmp_path) -> None:
    path = _write(
        tmp_path / "query-1-kis.csv", "L30_V046,6613\nL30_V048,8165\n"
    )

    assert lint_file(path) == []


def test_a_correct_qa_file_with_a_quoted_answer_reports_nothing(tmp_path) -> None:
    """Quoting is required around an answer holding a comma, so a legal
    quoted field must not be mistaken for a stray space or extra column."""
    path = _write(
        tmp_path / "query-3-qa.csv", 'L26_V270,3072,"Có 3 người, gồm nam"\n'
    )

    assert lint_file(path) == []


def test_an_empty_file_reports_nothing(tmp_path) -> None:
    """An empty submission is valid and keeps the upload slot; flagging it
    would push someone into uploading nothing at all."""
    path = _write(tmp_path / "query-1-kis.csv", "")

    assert lint_file(path) == []


# -- the directory sweep -------------------------------------------------


def test_the_sweep_reports_each_broken_file_separately(tmp_path) -> None:
    _write(tmp_path / "query-1-kis.csv", "L30_V046, 6613\n")
    _write(tmp_path / "query-3-qa.csv", "L26_V270,3072,1.2\n", bom=True)
    _write(tmp_path / "query-2-kis.csv", "L30_V048,8165\n")

    report = lint_directory(tmp_path)

    assert set(report) == {"query-1-kis.csv", "query-3-qa.csv"}


def test_the_sweep_ignores_review_files(tmp_path) -> None:
    """`*.review.csv` is a working file with a header and six columns on
    purpose. Linting it as a submission would report nothing but noise."""
    _write(
        tmp_path / "query-3-qa.review.csv",
        "rank,video_id,frame_idx,answer,score,image_path\n1,L1,2,,0.3,x\n",
    )

    assert lint_directory(tmp_path) == {}


# -- repair --------------------------------------------------------------


def test_repair_strips_the_bom_and_the_spaces(tmp_path) -> None:
    path = _write(tmp_path / "query-1-kis.csv", "L30_V046, 6613 \n", bom=True)

    repair_file(path)

    assert path.read_bytes() == b"L30_V046,6613\r\n"
    assert lint_file(path) == []


def test_repair_keeps_a_space_that_is_inside_an_answer(tmp_path) -> None:
    """Only the whitespace around a field is stray. `Năm người` has a real
    space in it, and trimming that would change the answer."""
    path = _write(tmp_path / "query-3-qa.csv", "L26_V270,3072, Năm người \n")

    repair_file(path)

    assert path.read_text(encoding="utf-8").strip() == "L26_V270,3072,Năm người"


def test_repair_drops_a_duplicate_and_keeps_the_first(tmp_path) -> None:
    """Safe to automate, unlike the rest. R@k reads the best answer among
    the first k, so a copy of a line already above it cannot add score in
    any ordering -- there is no case where keeping it is right."""
    path = _write(
        tmp_path / "query-1-kis.csv", "L1,1\nL2,2\nL1,1\nL3,3\nL2,2\n"
    )

    repair_file(path)

    assert path.read_text(encoding="utf-8").split() == ["L1,1", "L2,2", "L3,3"]
    assert lint_file(path) == []


def test_repair_keeps_the_ranking_order(tmp_path) -> None:
    """Rank is the whole point of the file: R@1 is one fifth of the score.
    Deduplicating must not reorder what survives."""
    path = _write(tmp_path / "query-1-kis.csv", "L9,9\nL1,1\nL9,9\nL5,5\n")

    repair_file(path)

    assert path.read_text(encoding="utf-8").split() == ["L9,9", "L1,1", "L5,5"]


def test_repair_keys_a_duplicate_on_video_and_frame_only(tmp_path) -> None:
    """Two Q&A rows naming the same frame are the same prediction even when
    the answers differ -- the second can never score where the first did
    not, because the frame is what the interval test reads."""
    path = _write(
        tmp_path / "query-3-qa.csv", "L1,1,năm\nL1,1,sáu\nL2,2,bảy\n"
    )

    repair_file(path)

    assert path.read_text(encoding="utf-8").split() == ["L1,1,năm", "L2,2,bảy"]


def test_repair_leaves_a_correct_file_byte_identical(tmp_path) -> None:
    path = _write(tmp_path / "query-1-kis.csv", "L30_V046,6613\r\n")
    before = path.read_bytes()

    repair_file(path)

    assert path.read_bytes() == before


# -- TRAKE rows are chains, not frames ----------------------------------------
#
# Found in a dress rehearsal: linting a real TRAKE submission reported 60 of
# its 100 rows as duplicates, and `--fix` deleted them. Every one was a
# distinct four-frame chain that merely shared a video and a first frame with
# a row above it. The report tells the operator to run `--fix`, so the path
# from "looks broken" to "60% of the answers are gone" was one command.


TRAKE_CHAINS = (
    "L26_V272,5325,5760,5888,5933\n"
    "L26_V272,5325,5760,5866,5888\n"
    "L26_V272,5325,5562,5760,5866\n"
)


def test_trake_chains_sharing_a_first_frame_are_not_duplicates(tmp_path) -> None:
    path = tmp_path / "query-4-trake.csv"
    path.write_text(TRAKE_CHAINS, encoding="utf-8")

    assert not [problem for problem in lint_file(path) if "trùng" in problem]


def test_repairing_a_trake_file_keeps_every_distinct_chain(tmp_path) -> None:
    path = tmp_path / "query-4-trake.csv"
    path.write_text(TRAKE_CHAINS, encoding="utf-8")

    repair_file(path)

    assert path.read_text(encoding="utf-8").strip().splitlines() == TRAKE_CHAINS.strip().splitlines()


def test_a_genuinely_repeated_trake_chain_is_still_a_duplicate(tmp_path) -> None:
    """The whole chain repeated is still one wasted slot out of 100."""
    path = tmp_path / "query-4-trake.csv"
    path.write_text(
        "L26_V272,5325,5760,5888,5933\nL26_V272,5325,5760,5888,5933\n", encoding="utf-8"
    )

    assert [problem for problem in lint_file(path) if "trùng" in problem]


def test_a_kis_frame_repeated_is_still_a_duplicate(tmp_path) -> None:
    """The rule that made TRAKE wrong is the right rule for KIS: the
    prediction is one frame, so a second row naming it adds nothing."""
    path = tmp_path / "query-4-kis.csv"
    path.write_text("L26_V272,5325\nL26_V272,5325\n", encoding="utf-8")

    assert [problem for problem in lint_file(path) if "trùng" in problem]
