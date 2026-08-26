"""Turn the round-1 submissions a person verified into a reusable ground truth.

The organizer published no scores for round 1, so the only record of what
was right is the operator's own inspection: they opened the frames, and
told us which files held a correct or near-correct answer. That knowledge
lives nowhere but in a chat message, and it is the only thing that can say
whether a round-2 change made retrieval better or worse.

This freezes it into a file.

**The verified answer is row 1** of the submission. Two things back that
reading rather than assuming it: the operator drove `answer_queries.py
--interactive`, whose `:top` command exists precisely to lift a frame the
person found by eye to rank 1; and three of the seventeen answers name a
frame that is **not in the keyframe mapping at all** (query 5, 6, 11), so
retrieval could not have produced them -- they were read off the video.

The query text is copied into the output on purpose. `artifacts/` is
gitignored, so a ground truth that pointed at `artifacts/queries/` would be
unusable to the teammate who checks the repo out. Round 1 is over; there
is nothing to keep back.

Name collision, deliberately avoided: `artifacts/qna/ground_truth.json`
already holds three entries keyed `query-p1-15-qa`, `query-p1-19-qa`,
`query-p1-22-qa` -- from an EARLIER practice pack that numbered its
queries the same way. Its `query-p1-15-qa` asks the name of a commune;
round 1's asks about an earthquake map. They are different queries with
the same name, so this writes a separate file and never merges the two.

    python scripts/build_round1_ground_truth.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: The queries the operator inspected and reported as correct or nearly so
#: after round 1. Everything outside this list is unknown, not wrong.
VERIFIED_ROUND_1 = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 22, 25)


def _first_row(path: Path) -> list[str]:
    """Row 1 of a submission, BOM and stray whitespace removed.

    Five round-1 files carry an Excel BOM and four have spaces around the
    commas; both were still present when this ran, and reading them raw
    would put `" 6613"` into the ground truth.
    """
    text = path.read_bytes().decode("utf-8-sig")
    for row in csv.reader(io.StringIO(text)):
        if row and any(row):
            return [value.strip() for value in row]
    raise SystemExit(f"{path}: file rỗng, không có đáp án nào để lấy")


def _query_text(queries_dir: Path, name: str) -> tuple[str, str | None]:
    vietnamese = (queries_dir / f"{name}.txt").read_text(encoding="utf-8").strip()
    english_path = queries_dir / f"{name}.en.txt"
    english = (
        english_path.read_text(encoding="utf-8").strip()
        if english_path.exists()
        else None
    )
    return vietnamese, english


def build(
    submissions: Path, queries: Path, verified: tuple[int, ...]
) -> dict[str, object]:
    entries: dict[str, object] = {}
    for number in verified:
        matches = sorted(submissions.glob(f"query-p1-{number}-*.csv"))
        matches = [p for p in matches if not p.name.endswith(".review.csv")]
        if len(matches) != 1:
            raise SystemExit(
                f"cần đúng 1 file cho câu {number}, tìm thấy {len(matches)}: "
                f"{[p.name for p in matches]}"
            )
        path = matches[0]
        name = path.stem
        kind = name.rsplit("-", 1)[1]
        row = _first_row(path)
        vietnamese, english = _query_text(queries, name)

        entry: dict[str, object] = {
            "kind": kind,
            "video_id": row[0],
            "query_vi": vietnamese,
            "verified_by": "human",
        }
        if english is not None:
            entry["query_en"] = english
        if kind == "trake":
            # A TRAKE answer is a chain of one frame per sub-event, not a
            # single frame. Kept whole; the frame-level proxy uses the
            # first, which is the only one a KIS-shaped scorer can read.
            entry["frames"] = [int(value) for value in row[1:]]
            entry["frame_idx"] = int(row[1])
        else:
            entry["frame_idx"] = int(row[1])
        if kind == "qa" and len(row) >= 3 and row[2]:
            entry["answer"] = row[2]
        entries[name] = entry

    return {
        "pack": "round-1",
        "source": "artifacts/submissions row 1, verified by the operator",
        "note": (
            "Đáp án được người xác minh bằng mắt sau vòng 1 — BTC không công "
            "bố kết quả. Người xác minh mô tả là 'đúng hoặc tương đối đúng', "
            "nên chỉ có VIDEO là chắc chắn; frame_idx là một frame đúng đã "
            "thấy, KHÔNG phải khoảng [start,end] thật của BTC. Vì vậy hãy đo "
            "ở mức video (xem mvr.evaluation.verified_set)."
        ),
        "queries": entries,
    }


def merge_queries(
    existing: dict, new_queries: dict, *, force: bool
) -> tuple[dict, list[tuple[str, object]]]:
    """Fold freshly-built entries onto whatever ground truth already
    exists, without reverting a correction this script did not make.

    This script's own route is "human": the operator's eyeball pass over
    artifacts/submissions. apply_sublan1_review.py can add or CORRECT
    entries afterwards and tags them with a different verified_by.
    Re-running this script with its defaults must not silently revert
    those -- a correction born from a real cross-check disagreement would
    vanish and nothing would say so.
    """
    skipped: list[tuple[str, object]] = []
    merged = dict(existing)
    for name, entry in new_queries.items():
        prior = existing.get(name)
        if prior is not None and prior.get("verified_by") != "human" and not force:
            skipped.append((name, prior.get("verified_by")))
            continue
        merged[name] = entry
    return merged, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--submissions", type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "submissions",
    )
    parser.add_argument(
        "--queries", type=Path, default=REPOSITORY_ROOT / "artifacts" / "queries"
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPOSITORY_ROOT / "data" / "ground_truth" / "round1_verified.json",
    )
    parser.add_argument(
        "--verified", type=int, nargs="+", default=list(VERIFIED_ROUND_1),
        help="Query numbers the operator confirmed. Defaults to round 1's 17.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Overwrite an entry even if it was verified through a "
            "different route (e.g. scripts/apply_sublan1_review.py). "
            "Without this, such an entry is left untouched and reported, "
            "never silently reverted."
        ),
    )
    args = parser.parse_args()

    payload = build(args.submissions, args.queries, tuple(args.verified))
    new_queries = payload["queries"]
    assert isinstance(new_queries, dict)

    existing: dict = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8")).get("queries", {})

    merged, skipped = merge_queries(existing, new_queries, force=args.force)
    payload["queries"] = merged

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    kinds: dict[str, int] = {}
    for entry in merged.values():
        assert isinstance(entry, dict)
        kind = str(entry["kind"])
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"wrote {len(merged)} câu vào {args.output}", file=sys.stderr)
    print(f"  {kinds}", file=sys.stderr)
    if skipped:
        print(
            f"  giữ nguyên {len(skipped)} câu đã xác minh bằng nguồn khác "
            f"(dùng --force nếu thật sự muốn ghi đè):",
            file=sys.stderr,
        )
        for name, source in skipped:
            print(f"    {name}  (verified_by={source!r})", file=sys.stderr)


if __name__ == "__main__":
    main()
