"""Uniqueness check for hand-authored keyframe-selector benchmark GT.

For each candidate GT row, embeds every organizer keyframe of that row's
video (BTC's own dense keyframe set, already on disk -- no ingest needed)
with the project's existing OpenCLIP encoder, then checks whether the
query text's best match actually falls inside [start_frame, end_frame].
Reused across videos: each video's keyframes are only embedded once.

This does not verify uniqueness against the *whole* corpus (all L2x
videos) -- that would mean encoding tens of thousands of images on CPU.
It checks uniqueness within the row's own video, which is the dominant
real risk for a single-video KIS-style query (near-duplicate moments in
the same video), and is what the manual-GT plan asked for.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_map_keyframes(path: Path) -> dict[int, int]:
    """Return {keyframe_number (n) -> frame_idx} from an organizer CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {int(row["n"]): int(row["frame_idx"]) for row in csv.DictReader(stream)}


def _video_keyframes(dataset_root: Path, video_id: str) -> list[tuple[Path, int]]:
    """Return [(image_path, frame_idx), ...] sorted by keyframe number.

    Resolves the collection-specific keyframe folder from the video_id
    prefix (e.g. "L23_V001" -> "Keyframes_L23/keyframes/L23_V001"), since
    organizer keyframes are split per-collection but map_keyframes is one
    shared folder across all collections.
    """
    collection = video_id.split("_")[0]
    video_dir = None
    for candidate_root in sorted(dataset_root.glob(f"Keyframes_{collection}*")):
        candidate_dir = candidate_root / "keyframes" / video_id
        if candidate_dir.is_dir():
            video_dir = candidate_dir
            break
    if video_dir is None:
        raise FileNotFoundError(
            f"no organizer keyframes for {video_id} under any "
            f"{dataset_root}/Keyframes_{collection}*/keyframes/{video_id}"
        )
    mapping = _load_map_keyframes(dataset_root / "map_keyframes" / f"{video_id}.csv")
    items = []
    for image_path in sorted(video_dir.glob("*.jpg")):
        n = int(image_path.stem)
        items.append((image_path, mapping[n]))
    if not items:
        raise FileNotFoundError(f"no .jpg keyframes found under {video_dir}")
    return items


def check_uniqueness(
    gt_rows: list[dict],
    *,
    dataset_root: Path,
    encoder,
    ambiguity_margin: float = 0.03,
) -> list[dict]:
    """Return gt_rows with uniqueness_checked/uniqueness_notes filled in."""
    cache: dict[str, tuple[list[int], np.ndarray]] = {}
    results = []
    for row in gt_rows:
        video_id = row["video_id"]
        if video_id not in cache:
            items = _video_keyframes(dataset_root, video_id)
            vectors = encoder.encode_images([path for path, _ in items])
            cache[video_id] = ([frame_idx for _, frame_idx in items], vectors)
        frame_indices, vectors = cache[video_id]

        query_vector = encoder.encode_text([row["query_en"]])[0]
        scores = vectors @ query_vector
        order = np.argsort(-scores, kind="stable")
        best_position = int(order[0])
        best_frame = frame_indices[best_position]
        best_score = float(scores[best_position])
        in_range = row["start_frame"] <= best_frame <= row["end_frame"]

        # Best-scoring frame outside the intended range, close in score to
        # the best in-range frame -> flag as ambiguous instead of guessing.
        in_range_scores = [
            float(scores[position])
            for position, frame_idx in enumerate(frame_indices)
            if row["start_frame"] <= frame_idx <= row["end_frame"]
        ]
        best_in_range_score = max(in_range_scores) if in_range_scores else None
        ambiguous = (
            not in_range
            and best_in_range_score is not None
            and (best_score - best_in_range_score) < ambiguity_margin
        )

        notes = []
        if not in_range:
            notes.append(
                f"best match is frame {best_frame} (score {best_score:.4f}), "
                f"outside the claimed [{row['start_frame']}, {row['end_frame']}]"
            )
        if ambiguous:
            notes.append("close-scoring in-range competitor exists -- ambiguous query")

        result = dict(row)
        result["uniqueness_checked"] = True
        result["uniqueness_passed"] = bool(in_range and not ambiguous)
        result["uniqueness_best_frame"] = best_frame
        result["uniqueness_best_score"] = round(best_score, 6)
        result["uniqueness_notes"] = "; ".join(notes)
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="candidate ground_truth.json")
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="organizer dataset root containing Keyframes_Lxx/ and map_keyframes/, "
             "e.g. .../AIC_DATASET_ROOT",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from mvr.enrichment.embeddings.openclip import OpenClipEncoder

    payload = json.loads(args.gt.read_text(encoding="utf-8"))
    encoder = OpenClipEncoder(device=args.device)
    checked = check_uniqueness(
        payload["test_cases"], dataset_root=args.dataset_root, encoder=encoder
    )
    payload["test_cases"] = checked
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    failed = [row for row in checked if not row["uniqueness_passed"]]
    print(f"{len(checked)} rows checked, {len(failed)} failed uniqueness")
    for row in failed:
        print(f"  FAIL {row['query_id']}: {row['uniqueness_notes']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
