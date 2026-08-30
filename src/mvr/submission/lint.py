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


def _duplicate_key(row: list[str], kind: str | None) -> tuple[str, ...]:
    """What makes two submission rows the same prediction.

    KIS and Q&A predict one frame, so ``(video_id, frame_idx)`` decides:
    a second row naming that frame can never score where the first did not.

    A TRAKE row is a whole chain -- ``video_id`` plus one frame per event --
    and two chains sharing a video and a first frame are different answers.
    Keying them on the first two columns reported most of a real submission
    as duplicates and let ``--fix`` delete 60 of its 100 chains.
    """
    stripped = tuple(value.strip() for value in row)
    return stripped if kind == "trake" else stripped[:2]


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
    seen: dict[tuple[str, ...], int] = {}
    duplicates: list[str] = []

    for number, row in enumerate(rows, start=1):
        # Shape problems are systemic -- if one row has the wrong column
        # count the whole file was built wrong -- so one report is enough
        # and stops the scan.
        if expected is not None and len(row) != expected:
            problems.append(
                f"dòng {number}: có {len(row)} cột, {kind} cần {expected} cột"
            )
            break
        if kind == "trake" and len(row) < 2:
            problems.append(f"dòng {number}: trake cần video_id và ít nhất 1 frame")
            break

        frames = row[1:2] if kind == "qa" else row[1:]
        bad = next(
            (value for value in frames if not _is_integer(value)),
            None,
        )
        if bad is not None:
            problems.append(f"dòng {number}: frame_idx {bad!r} không phải số nguyên")
            break

        # Duplicates are reported ALL AT ONCE. Reporting only the first
        # turned fixing them into whack-a-mole: three files were cleaned,
        # relinted, and each surfaced a different duplicate pair that had
        # been there the whole time.
        key = _duplicate_key(row, kind)
        if key in seen:
            duplicates.append(f"dòng {number} (giống dòng {seen[key]}): {','.join(key)}")
        else:
            seen[key] = number

        if kind == "qa" and len(row) >= 3 and len(row[2]) > MAX_ANSWER_CHARS:
            problems.append(
                f"dòng {number}: answer dài {len(row[2])} ký tự, "
                f"tối đa {MAX_ANSWER_CHARS} ký tự"
            )
            break

    if duplicates:
        shown = duplicates[:5]
        more = f" (và {len(duplicates) - 5} dòng nữa)" if len(duplicates) > 5 else ""
        problems.append(
            f"{len(duplicates)} dòng trùng — mỗi dòng tốn một slot mà không "
            f"thêm điểm: " + "; ".join(shown) + more
        )

    return problems


def _is_integer(value: str) -> bool:
    try:
        int(value.strip())
    except ValueError:
        return False
    return True


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
    """Strip the BOM, the stray whitespace, and the duplicate rows.

    Only whitespace *around* a field goes: a space inside an answer, as in
    ``Năm người``, is part of the answer and trimming it would change what
    was submitted.

    **Deduplication is safe to automate**, unlike everything else the
    linter reports. R@k reads the best answer among the first k, so a copy
    of a line already above it cannot add score under any ordering -- there
    is no case where keeping it is right, and each one costs a slot out of
    100. Measured in round 1: one file spent 72 of its 100 slots on
    duplicates and offered 28 distinct answers.

    The first occurrence stays and the order is preserved, because rank is
    what the file is for: R@1 is one fifth of the score.

    KIS and Q&A rows are keyed on ``(video_id, frame_idx)`` only. Two Q&A
    rows naming one frame are the same prediction even when the answers
    differ -- the frame is what the interval test reads, so the second can
    never score where the first did not. A TRAKE row is keyed on the whole
    chain; see ``_duplicate_key``.

    Everything else the linter reports needs a person: a wrong column count
    could be a missing field or an extra one, and guessing is worse than
    reporting.
    """
    path = Path(path)
    kind = kind_from_name(path.name)
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if not text.strip():
        return []

    rows = _rows(text)
    cleaned: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        stripped = [value.strip() for value in row]
        key = _duplicate_key(row, kind)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(stripped)

    if cleaned == rows and not raw.startswith(b"\xef\xbb\xbf"):
        return []

    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, quoting=csv.QUOTE_MINIMAL).writerows(cleaned)

    notes = ["đã bỏ BOM và khoảng trắng thừa"]
    dropped = len(rows) - len(cleaned)
    if dropped:
        notes.append(f"bỏ {dropped} dòng trùng")
    return notes


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
