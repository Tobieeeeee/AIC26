"""Fold a completed sublan1 review sheet back into the ground truth.

Reads the two sheets `build_sublan1_review.py` wrote, once a person has
filled the `chon_dong_nay` column with an `x` on the row they confirmed by
looking at the image. Writes nothing on its own until that column has
content -- an unfilled sheet changes nothing, rather than guessing.

    video_mismatch.csv   the row marked `x` REPLACES that query's existing
                          entry in round1_verified.json. If the row marked
                          is the "ta (đã xác minh)" one, the existing entry
                          is left untouched -- the person looked and
                          confirmed the operator was right after all.
    new_candidates.csv    the row marked `x` becomes a NEW entry, for a
                          query round 1 never verified at all. Skipped
                          entirely if nothing was marked -- a query nobody
                          could confirm from either source stays unknown.

Every changed or added entry is tagged
``"verified_by": "human (đối chiếu sublan1)"`` rather than the plain
``"human"`` the original 17 carry, so the provenance is visible in the
file itself, not just in a commit message.

    python scripts/apply_sublan1_review.py            # xem trước, không ghi
    python scripts/apply_sublan1_review.py --write     # ghi vào ground truth
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = REPOSITORY_ROOT / "artifacts" / "sublan1_review"
GROUND_TRUTH_PATH = REPOSITORY_ROOT / "data" / "ground_truth" / "round1_verified.json"
QUERIES_DIR = REPOSITORY_ROOT / "artifacts" / "queries"

VERIFIED_BY_SUBLAN1 = "human (đối chiếu sublan1)"


def _rows(path: Path) -> list[dict[str, str]]:
    text = path.read_bytes().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _marked(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """One marked row per query. Raises if a query has more than one `x`:
    that means the sheet was filled in a hurry, and applying either row
    silently would pick the wrong one without saying so."""
    chosen: dict[str, dict[str, str]] = {}
    for row in rows:
        if not row.get("chon_dong_nay", "").strip():
            continue
        query = row["query"]
        if query in chosen:
            raise SystemExit(
                f"{query}: có hơn một dòng đánh dấu 'x' — xoá bớt, chỉ giữ 1 dòng"
            )
        chosen[query] = row
    return chosen


def _english_text(name: str) -> str | None:
    path = QUERIES_DIR / f"{name}.en.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def apply_mismatch(ground_truth: dict, rows: list[dict[str, str]]) -> list[str]:
    changes = []
    for query, row in _marked(rows).items():
        if row["source"].startswith("ta"):
            continue  # người xem lại xác nhận đáp án cũ đúng -- không đổi gì
        entry = ground_truth[query]
        old = (entry["video_id"], entry["frame_idx"])
        entry["video_id"] = row["video_id"]
        entry["frame_idx"] = int(row["frame_idx"])
        entry["verified_by"] = VERIFIED_BY_SUBLAN1
        changes.append(f"{query}: {old} -> ({row['video_id']}, {row['frame_idx']})")
    return changes


def apply_new(ground_truth: dict, rows: list[dict[str, str]]) -> list[str]:
    changes = []
    for query, row in _marked(rows).items():
        if query in ground_truth:
            continue  # đã có sẵn -- không ghi đè một mục đã xác minh trước đó
        kind = query.rsplit("-", 1)[1]
        entry = {
            "kind": kind,
            "video_id": row["video_id"],
            "frame_idx": int(row["frame_idx"]),
            "query_vi": row["query_vi"],
            "verified_by": VERIFIED_BY_SUBLAN1,
        }
        english = _english_text(query)
        if english is not None:
            entry["query_en"] = english
        ground_truth[query] = entry
        changes.append(f"{query}: thêm mới ({row['video_id']}, {row['frame_idx']})")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="Ghi vào ground truth. Mặc định chỉ xem trước.")
    args = parser.parse_args()

    payload = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    ground_truth = payload["queries"]

    mismatch_path = REVIEW_DIR / "video_mismatch.csv"
    new_path = REVIEW_DIR / "new_candidates.csv"
    changes = []
    if mismatch_path.exists():
        changes += apply_mismatch(ground_truth, _rows(mismatch_path))
    if new_path.exists():
        changes += apply_new(ground_truth, _rows(new_path))

    if not changes:
        print("Chưa có dòng nào được đánh dấu 'x' -- không có gì để áp dụng.", file=sys.stderr)
        return

    print(f"{len(changes)} thay đổi:", file=sys.stderr)
    for line in changes:
        print(f"  {line}", file=sys.stderr)

    if not args.write:
        print("\n(chỉ xem trước -- chạy lại với --write để ghi thật)", file=sys.stderr)
        return

    GROUND_TRUTH_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nđã ghi {GROUND_TRUTH_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
