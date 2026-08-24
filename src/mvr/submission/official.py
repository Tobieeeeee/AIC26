"""The organizer's submission file format for the preliminary round.

One `.csv` per query, at most 100 lines, no header row, UTF-8, comma
delimited -- taken from the organizer's "Hướng dẫn nộp bài sơ tuyển":

    KIS    <video_id>,<frame_idx>                     L00_V000,1234
    Q&A    <video_id>,<frame_idx>,<answer>            L01_V028,3450,5

This module exists because the repo had no official exporter. What it had
was `export_local_verification_csv`, whose own docstring said it "is
explicitly not the official organizer submission format -- adapt it only
once that format is documented", and which wrote

    format,query_id,rank,video_id,frame_id

as its first line. Uploading that would have scored zero on a retrieval
system that works, which is the most expensive kind of mistake available
here.

Quoting follows RFC 4180, which is exactly what the organizer describes:
quotes are required only around an answer containing a comma, a quote, or a
newline, and a literal quote is doubled. Their own example file shows the
plain form for simple answers (`L01_V028,3450,5`), so this writes that.
"""
from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path

#: "Đội thi có thể nộp file tối đa 100 dòng."
MAX_ROWS = 100

#: "Độ dài tối đa: 100 ký tự" -- characters, not bytes. Vietnamese is
#: multi-byte in UTF-8, so counting bytes would reject legal answers.
MAX_ANSWER_CHARS = 100


class SubmissionFormatError(ValueError):
    """Raised when a row cannot be written as a valid submission line."""


def _video_id(value: object, *, row: int) -> str:
    text = str(value).strip()
    if not text:
        raise SubmissionFormatError(f"dòng {row}: video_id không được rỗng")
    return text


def _frame_idx(value: object, *, row: int, label: str = "frame_idx") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            value = int(str(value).strip())
        except (TypeError, ValueError):
            raise SubmissionFormatError(
                f"dòng {row}: {label} phải là số nguyên, nhận {value!r}"
            ) from None
    if value < 0:
        raise SubmissionFormatError(f"dòng {row}: {label} không được âm ({value})")
    return int(value)


def _open_writer(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # `newline=""` lets csv control line endings; `encoding="utf-8"` rather
    # than "utf-8-sig" because a BOM would corrupt the first video_id.
    handle = path.open("w", newline="", encoding="utf-8")
    return handle, csv.writer(handle, quoting=csv.QUOTE_MINIMAL)


def write_kis_submission(
    predictions: Iterable[tuple[object, object]],
    path: Path | str,
) -> Path:
    """``<video_id>,<frame_idx>``, one ranked prediction per line."""
    path = Path(path)
    handle, writer = _open_writer(path)
    with handle:
        for row, (video_id, frame_idx) in enumerate(
            list(predictions)[:MAX_ROWS], start=1
        ):
            writer.writerow([_video_id(video_id, row=row), _frame_idx(frame_idx, row=row)])
    return path


def write_qna_submission(
    predictions: Iterable[tuple[object, object, object]],
    path: Path | str,
) -> Path:
    """``<video_id>,<frame_idx>,<answer>``.

    An empty answer is allowed on purpose. A frame with no answer still
    identifies the moment, and a human can fill the column in afterwards --
    refusing the row would mean submitting nothing at all for that query.
    """
    path = Path(path)
    handle, writer = _open_writer(path)
    with handle:
        for row, (video_id, frame_idx, answer) in enumerate(
            list(predictions)[:MAX_ROWS], start=1
        ):
            text = "" if answer is None else str(answer)
            if len(text) > MAX_ANSWER_CHARS:
                raise SubmissionFormatError(
                    f"dòng {row}: answer dài {len(text)} ký tự, tối đa "
                    f"{MAX_ANSWER_CHARS} ký tự"
                )
            writer.writerow(
                [_video_id(video_id, row=row), _frame_idx(frame_idx, row=row), text]
            )
    return path


#: The three tracks the organizer names. `trake` belongs to another
#: teammate, but the filename rule is the same for all three.
QUERY_KINDS = ("kis", "qa", "trake")

#: A query file, whatever the pack calls it: `query-1-kis`, `query-p1-23-kis`.
#: The trailing kind is the only part the format depends on.
_QUERY_NAME = re.compile("query-(.+)-(kis|qa|trake)", re.IGNORECASE)


def submission_filename(query_name: object, kind: object = None) -> str:
    """``"query-p1-23-kis.txt"`` -> ``"query-p1-23-kis.csv"``.

    **The submission carries the query file's own name.** The organizer
    states it by example: their pack holds `query-1-kis.txt` and their
    required listing holds `query-1-kis.csv` -- same name, new extension.
    Their example pack simply has no batch prefix; a real one does.

    This used to renumber `query-p1-23-kis` to `query-23-kis`, read out of
    that same directory listing. That was an over-reading: the listing was
    showing their example pack's names, not a renumbering rule.

    A bare number or a bare id still works for the interactive writer,
    where the operator types a fragment rather than a filename: ``23`` and
    ``p1-23`` become ``query-23-kis.csv`` and ``query-p1-23-kis.csv``.
    Nothing is invented -- the id is used exactly as typed.
    """
    text = str(query_name).strip()
    if text.lower().endswith((".csv", ".txt")):
        text = text[: -len(".csv")]

    match = _QUERY_NAME.fullmatch(text)
    if match is not None:
        return f"query-{match.group(1)}-{match.group(2).lower()}.csv"

    # Not a full query name: treat it as the id, and take the kind from the
    # caller. This is the interactive `:w kis 23` path.
    kind_text = str(kind).strip().lower()
    if kind_text not in QUERY_KINDS:
        raise SubmissionFormatError(
            f"dạng truy vấn không hợp lệ: {kind!r}; phải là một trong {QUERY_KINDS}"
        )
    # An id is a number, optionally behind a batch prefix: `23`, `p1-23`.
    # Anything else is not an id, and building `query-notes-kis-kis.csv`
    # out of it would be worse than refusing.
    if not re.fullmatch(r"(?:[A-Za-z]+[0-9]*-)?[0-9]+", text):
        raise SubmissionFormatError(
            f"không dựng được tên file nộp từ {query_name!r}; gõ tên câu truy "
            f"vấn (ví dụ query-p1-23-kis) hoặc số thứ tự của nó (23, p1-23)"
        )
    return f"query-{text}-{kind_text}.csv"


__all__ = [
    "MAX_ANSWER_CHARS",
    "QUERY_KINDS",
    "MAX_ROWS",
    "SubmissionFormatError",
    "submission_filename",
    "write_kis_submission",
    "write_qna_submission",
]
