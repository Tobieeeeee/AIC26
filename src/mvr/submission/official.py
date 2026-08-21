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


__all__ = [
    "MAX_ANSWER_CHARS",
    "MAX_ROWS",
    "SubmissionFormatError",
    "write_kis_submission",
    "write_qna_submission",
]
