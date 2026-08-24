"""Check a submission file against the organizer's contract before upload.

Written after the round-1 files were audited. Every check below is a defect
that was actually present in them, and none of them makes a file look wrong
when you open it:

    BOM                  5 files, all the ones edited in Excel
    spaces round commas  4 files, one of them on all 100 rows

The organizer asks for `Encoding: UTF-8` -- a BOM is not that -- and states
plainly that "Khoảng trắng đầu/cuối: Được giữ nguyên, không tự động trim",
so `L30_V046, 6613` risks being compared as the string `" 6613"` instead of
the integer 6613.

`lint_file` reports; `repair_file` fixes the two whitespace-class defects
and leaves everything else alone, because the rest need a decision a person
has to make.
"""
from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping
from pathlib import Path

from .official import MAX_ANSWER_CHARS, MAX_ROWS

#: Column count per track, from the organizer's format section. TRAKE is
#: `<video_id>` plus one frame per event, so its width depends on the query
#: and only a minimum can be checked here.
COLUMNS = {"kis": 2, "qa": 3}

#: The submission carries the query file's own name, so the id between
#: `query-` and the kind is whatever the pack used -- `1`, `p1-23`.
SUBMISSION_NAME = re.compile(r"^query-(.+)-(kis|qa|trake)\.csv$")

#: Words that mean the first line is a header rather than an answer.
_HEADER_TOKENS = ("video_id", "frame_id", "frame_idx", "rank", "answer", "format")


def kind_from_name(name: str) -> str | None:
    """`query-3-qa.csv` -> `qa`. None when the name says nothing."""
    match = SUBMISSION_NAME.match(Path(name).name)
    if match:
        return match.group(2)
    stem = Path(name).stem.lower()
    for kind in ("kis", "qa", "trake"):
        if stem.endswith(f"-{kind}"):
            return kind
    return None


def _rows(text: str) -> list[list[str]]:
    return [row for row in csv.reader(io.StringIO(text)) if row and any(row)]


def lint_file(path: Path | str, *, kind: str | None = None) -> list[str]:
    """Every problem found in one submission file, in reading order."""
    path = Path(path)
    kind = kind or kind_from_name(path.name)
    problems: list[str] = []

    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append(
            "có BOM ở đầu file — Excel thêm vào khi lưu. BTC yêu cầu UTF-8, "
            "và BOM làm video_id dòng đầu không còn khớp."
        )
    text = raw.decode("utf-8-sig")

    if not text.strip():
        # An empty submission is valid and keeps the upload slot. Flagging
        # it would push someone into uploading nothing at all.
        return problems

    if not SUBMISSION_NAME.match(path.name):
        problems.append(
            f"tên file {path.name!r} không đúng chuẩn BTC — phải là "
            f"query-<tên câu truy vấn>-<kis|qa|trake>.csv, giữ nguyên tên "
            f"file đề, chỉ đổi .txt thành .csv"
        )

    rows = _rows(text)

    first = ",".join(rows[0]).lower() if rows else ""
    if any(token in first for token in _HEADER_TOKENS):
        problems.append(
            "dòng đầu trông như header — file nộp bắt đầu thẳng bằng dữ liệu"
        )

    if len(rows) > MAX_ROWS:
        problems.append(f"có {len(rows)} dòng, tối đa {MAX_ROWS} dòng")

    # Judged on the PARSED field, not the raw line. A raw-line regex flags
    # `"Có 3 người, gồm nam"` -- where the comma and space sit inside the
    # quotes and belong to the answer -- while the csv reader has already
    # resolved quoting, so a field differing from its own strip() is stray
    # whitespace and nothing else.
    for number, row in enumerate(rows, start=1):
        if any(value != value.strip() for value in row):
            problems.append(
                f"dòng {number}: có khoảng trắng thừa quanh dấu phẩy hoặc cuối "
                f"dòng — BTC không tự trim, nên ' 6613' khác '6613'"
            )
            break  # one report per file is enough to act on

    expected = COLUMNS.get(kind or "")
    seen: set[tuple[str, ...]] = set()
    for number, row in enumerate(rows, start=1):
        if expected is not None and len(row) != expected:
            problems.append(
                f"dòng {number}: có {len(row)} cột, {kind} cần {expected} cột"
            )
            break
        if kind == "trake" and len(row) < 2:
            problems.append(f"dòng {number}: trake cần video_id và ít nhất 1 frame")
            break

        for index, value in enumerate(row[1 : (3 if kind == "qa" else len(row))], 1):
            if kind == "qa" and index > 1:
                continue
            try:
                int(value.strip())
            except ValueError:
                problems.append(
                    f"dòng {number}: frame_idx {value!r} không phải số nguyên"
                )
                break
        else:
            key = tuple(value.strip() for value in row[:2])
            if key in seen:
                problems.append(
                    f"dòng {number}: trùng với một dòng trước ({key[0]},{key[1]}) "
                    f"— tốn một slot mà không thêm điểm"
                )
                break
            seen.add(key)
            if kind == "qa" and len(row) >= 3 and len(row[2]) > MAX_ANSWER_CHARS:
                problems.append(
                    f"dòng {number}: answer dài {len(row[2])} ký tự, "
                    f"tối đa {MAX_ANSWER_CHARS} ký tự"
                )
                break
            continue
        break

    return problems


def lint_directory(directory: Path | str) -> dict[str, list[str]]:
    """Every broken submission in a folder, keyed by filename.

    `*.review.csv` is skipped: it is a working file with a header and six
    columns by design, and linting it as a submission produces only noise.
    """
    directory = Path(directory)
    report: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.csv")):
        if path.name.endswith(".review.csv"):
            continue
        problems = lint_file(path)
        if problems:
            report[path.name] = problems
    return report


def repair_file(path: Path | str) -> list[str]:
    """Strip the BOM and the stray whitespace. Nothing else.

    Only whitespace *around* a field goes: a space inside an answer, as in
    ``Năm người``, is part of the answer and trimming it would change what
    was submitted.

    Everything the linter reports beyond these two needs a person to decide
    -- a wrong column count could be a missing field or an extra one, and
    guessing would be worse than reporting.
    """
    path = Path(path)
    text = path.read_bytes().decode("utf-8-sig")
    if not text.strip():
        return []

    rows = _rows(text)
    cleaned = [[value.strip() for value in row] for row in rows]
    if cleaned == rows and not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        return []

    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, quoting=csv.QUOTE_MINIMAL).writerows(cleaned)
    return ["đã bỏ BOM và khoảng trắng thừa"]


def format_report(report: Mapping[str, list[str]], total: int) -> str:
    """The operator-facing summary."""
    if not report:
        return f"{total} file, không file nào có lỗi định dạng."
    lines = [f"{len(report)}/{total} file có lỗi:", ""]
    for name, problems in report.items():
        lines.append(f"  {name}")
        for problem in problems:
            lines.append(f"      - {problem}")
    lines += [
        "",
        "Sửa được hai lỗi BOM và khoảng trắng bằng: --lint <thư mục> --fix",
        "Các lỗi còn lại phải sửa tay — chúng cần người quyết định.",
    ]
    return "\n".join(lines)


__all__ = [
    "COLUMNS",
    "SUBMISSION_NAME",
    "format_report",
    "kind_from_name",
    "lint_directory",
    "lint_file",
    "repair_file",
]
