"""Turn a shared team's round-1 submission into two sheets a person opens
and eyeballs -- never into ground truth on its own.

Comparing `artifacts/sublan1` (shared voluntarily by another team) against
`data/ground_truth/round1_verified.json` found 8 of 17 already-"verified"
queries naming a DIFFERENT video entirely. Not a frame off by a few
hundred -- a different video. That is direct evidence neither source is
trustworthy alone: our operator's eyeball pass could be wrong, or the
other team's could, and "they scored highest overall" says nothing about
any one specific query.

So this never writes into round1_verified.json. It writes two CSVs to
`artifacts/sublan1_review/` for a person to open, look at the frames, and
fill in the last column. `scripts/build_round1_ground_truth.py` reads the
result once the person is done, exactly like it reads any other verified
submission.

    video_mismatch.csv    the 8 queries where the two sources disagree on
                           the VIDEO -- fix candidates for existing entries
    new_candidates.csv     the 8 queries round 1 never verified at all
                           (8, 14, 18, 19, 20, 21, 23, 24) -- sublan1 is
                           the only lead either side has

Every row resolves to the NEAREST keyframe image actually on disk, because
sublan1's frame_idx is the competition frame index, not a keyframe id, and
there is no guarantee a keyframe exists at that exact frame.

    python scripts/build_sublan1_review.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
from bisect import bisect_left
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: Queries round 1 never verified. Anything sublan1 offers here is the
#: only lead either team has -- pure candidate generation.
UNVERIFIED_ROUND_1 = (8, 14, 18, 19, 20, 21, 23, 24)

REVIEW_COLUMNS = (
    "query", "source", "video_id", "frame_idx",
    "nearest_keyframe_frame", "distance_frames", "image_path",
    "query_vi", "chon_dong_nay",
)


def _rows(path: Path) -> list[list[str]]:
    text = path.read_bytes().decode("utf-8-sig")
    return [
        [v.strip() for v in row]
        for row in csv.reader(io.StringIO(text))
        if row and any(row)
    ]


def load_frame_index(mapping_csv: Path) -> dict[str, list[tuple[int, str]]]:
    """`video_id` -> sorted `[(frame_idx, image_path), ...]`, for nearest-
    keyframe lookup. Loaded once; the mapping has 177k rows."""
    by_video: dict[str, list[tuple[int, str]]] = {}
    with mapping_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_video.setdefault(row["video_id"], []).append(
                (int(row["frame_idx"]), row["image_path"])
            )
    for frames in by_video.values():
        frames.sort()
    return by_video


def nearest_keyframe(
    frame_index: dict[str, list[tuple[int, str]]], video_id: str, frame_idx: int
) -> tuple[int, str, int] | None:
    """The keyframe closest to `frame_idx` in `video_id`, and the distance.

    None when the video itself is not in the mapping -- happens for a
    handful of round-1 answers a person read straight off the video
    rather than off a search result (see build_round1_ground_truth.py).
    """
    frames = frame_index.get(video_id)
    if not frames:
        return None
    values = [f for f, _ in frames]
    i = bisect_left(values, frame_idx)
    candidates = [c for c in (i - 1, i) if 0 <= c < len(frames)]
    best = min(candidates, key=lambda c: abs(frames[c][0] - frame_idx))
    best_frame, best_path = frames[best]
    return best_frame, best_path, abs(best_frame - frame_idx)


def query_text(queries_dir: Path, name: str) -> str:
    path = queries_dir / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def sublan1_candidates(
    submission_dir: Path, name: str
) -> list[tuple[str, int]]:
    path = submission_dir / f"{name}.csv"
    if not path.exists():
        return []
    rows = _rows(path)
    kind = name.rsplit("-", 1)[1]
    if kind == "trake":
        # Each row is one candidate CHAIN across possibly-different videos,
        # not ranked frames in one video. Keep the first frame per row as
        # the video-identification signal; the chain itself is not a KIS
        # answer and is out of scope for this review.
        return [(row[0], int(row[1])) for row in rows]
    return [(row[0], int(row[1])) for row in rows]


def build_row(
    *, query: str, source: str, video_id: str, frame_idx: int,
    frame_index: dict, query_vi: str,
) -> dict:
    nearest = nearest_keyframe(frame_index, video_id, frame_idx)
    if nearest is None:
        nearest_frame, image_path, distance = "", "(video không có trong keyframe_mapping.csv)", ""
    else:
        nearest_frame, image_path, distance = nearest
    return {
        "query": query,
        "source": source,
        "video_id": video_id,
        "frame_idx": frame_idx,
        "nearest_keyframe_frame": nearest_frame,
        "distance_frames": distance,
        "image_path": image_path,
        "query_vi": query_vi,
        "chon_dong_nay": "",
    }


def write_sheet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ground_truth = json.loads(
        (REPOSITORY_ROOT / "data" / "ground_truth" / "round1_verified.json").read_text(
            encoding="utf-8"
        )
    )["queries"]
    sublan1_dir = REPOSITORY_ROOT / "artifacts" / "sublan1" / "submission"
    queries_dir = REPOSITORY_ROOT / "artifacts" / "queries"
    frame_index = load_frame_index(REPOSITORY_ROOT / "artifacts" / "keyframe_mapping.csv")

    mismatch_rows: list[dict] = []
    for name, entry in sorted(ground_truth.items()):
        if entry["kind"] == "trake":
            continue  # chain answers, not a single video/frame comparison
        candidates = sublan1_candidates(sublan1_dir, name)
        sub_videos = {v for v, _ in candidates}
        if not sub_videos or entry["video_id"] in sub_videos:
            continue  # agree, or sublan1 has nothing to say -- not a conflict
        vi = query_text(queries_dir, name)
        mismatch_rows.append(
            build_row(
                query=name, source="ta (đã xác minh)", video_id=entry["video_id"],
                frame_idx=entry["frame_idx"], frame_index=frame_index, query_vi=vi,
            )
        )
        for video_id, frame_idx in candidates:
            mismatch_rows.append(
                build_row(
                    query=name, source="sublan1", video_id=video_id,
                    frame_idx=frame_idx, frame_index=frame_index, query_vi=vi,
                )
            )

    new_rows: list[dict] = []
    for number in UNVERIFIED_ROUND_1:
        matches = sorted(sublan1_dir.glob(f"query-p1-{number}-*.csv"))
        if not matches:
            continue
        name = matches[0].stem
        vi = query_text(queries_dir, name)
        for video_id, frame_idx in sublan1_candidates(sublan1_dir, name):
            new_rows.append(
                build_row(
                    query=name, source="sublan1", video_id=video_id,
                    frame_idx=frame_idx, frame_index=frame_index, query_vi=vi,
                )
            )

    out_dir = REPOSITORY_ROOT / "artifacts" / "sublan1_review"
    write_sheet(out_dir / "video_mismatch.csv", mismatch_rows)
    write_sheet(out_dir / "new_candidates.csv", new_rows)

    mismatch_queries = len({r["query"] for r in mismatch_rows})
    new_queries = len({r["query"] for r in new_rows})
    print(f"video_mismatch.csv : {mismatch_queries} câu, {len(mismatch_rows)} dòng", file=sys.stderr)
    print(f"new_candidates.csv : {new_queries} câu, {len(new_rows)} dòng", file=sys.stderr)
    print(f"ghi vào {out_dir}", file=sys.stderr)
    print(
        "\nMở từng file, xem ảnh ở image_path, điền 'x' vào cột "
        "chon_dong_nay ở dòng đúng. Không tự merge gì cả cho tới khi "
        "làm xong -- xem docs/DUNG_SUBLAN1.md.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
