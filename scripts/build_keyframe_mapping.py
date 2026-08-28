"""Turn the organizer's keyframe data into the team's canonical mapping CSV.

One command, one file. Everything downstream reads keyframe identity from
`keyframe_mapping.csv` and nothing else:

    indexing.task05a.load_canonical_mapping
    enrichment.ocr.docker_ocr.process_dataset_docker_ocr
    enrichment.feature_extraction.sort_keyframes_deterministically
    enrichment.asr.mapping
    indexing.semantic_builder.build_semantic_index

so adopting the organizer's 166,550 keyframes as the team's identity means
generating this CSV -- not changing any of those pipelines. If your code
already reads a mapping CSV, it needs no edit at all.

Why adopt them: the organizer decoded 873 videos (83 GB, ~130 hours) and
shipped CLIP features for every keyframe. Re-deriving that costs ~13 hours
of video decoding even at 10x realtime, and a GPU does not remove it --
the cost is I/O and decoding, not matrix multiplication.

Point `--dataset-root` at either the raw download or a normalized tree;
both layouts are searched:

    raw          <root>/Keyframes_L30/keyframes/L30_V072/034.jpg
                 <root>/map-keyframes-aic25-b1/map-keyframes/L30_V072.csv
    normalized   <root>/keyframes/L30_V072/034.jpg
                 <root>/map_keyframes/L30_V072.csv

Example:

    python scripts/build_keyframe_mapping.py \\
        --dataset-root <your-dataset-root> \\
        --output artifacts/keyframe_mapping.csv

Takes about 10 seconds for the full corpus. The result is verified by
handing it to the real consumers before the script exits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def discover_video_ids(dataset_root: Path) -> list[str]:
    """Every video the organizer shipped a keyframe mapping for.

    Discovered from the mapping CSVs rather than the keyframe directories,
    deliberately: 23 of 873 videos have a mapping but no downloaded
    keyframes, and those are still videos we know about. They contribute no
    rows, which is correct and not an error -- but discovering from
    directories would make them silently cease to exist, and the difference
    matters when someone asks why the count moved.
    """
    stems: set[str] = set()
    for directory in _mapping_directories(dataset_root):
        stems.update(path.stem for path in directory.glob("*.csv"))
    return sorted(stems)


def _mapping_directories(dataset_root: Path) -> list[Path]:
    candidates = [dataset_root / "map_keyframes"]
    candidates.extend(
        path / "map-keyframes" for path in sorted(dataset_root.glob("map-keyframes*"))
    )
    return [path for path in candidates if path.is_dir()]


def load_shot_boundaries(shots_json: Path) -> dict[str, list[tuple[int, int]]]:
    """Read `shots.json` from the team's own ingestion into frame ranges.

    The detectors number shots from 1 in start_frame order
    (`ingestion.shot_detection.pyscenedetect`, `.transnetv2`), and the
    adapter numbers them positionally, so the two agree only while that
    holds. It is checked rather than assumed: renumbering silently would
    break every join back to this file, and the symptom would appear far
    away as OCR attached to the wrong shot.
    """
    payload = json.loads(shots_json.read_text(encoding="utf-8"))
    by_video: dict[str, list[dict]] = {}
    for shot in payload["shots"]:
        by_video.setdefault(str(shot["video_id"]), []).append(shot)

    boundaries: dict[str, list[tuple[int, int]]] = {}
    for video_id, shots in by_video.items():
        shots.sort(key=lambda s: int(s["start_frame"]))
        expected = list(range(1, len(shots) + 1))
        found = [int(s["shot_id"]) for s in shots]
        if found != expected:
            raise SystemExit(
                f"{shots_json}: {video_id} numbers its shots {found[:5]}... but "
                f"this adapter numbers positionally from 1. Renumbering would "
                f"break every join back to shots.json, so it refuses instead."
            )
        boundaries[video_id] = [
            (int(s["start_frame"]), int(s["end_frame"])) for s in shots
        ]
    return boundaries


def _verify(output: Path) -> int:
    """Hand the file to the consumers that will actually read it.

    Checking the column header proves nothing. An earlier version of the
    adapter emitted every required column and still could not be loaded by
    a single downstream module, because `shot_id` was blank and all of them
    do `int(row["shot_id"])`. That shipped, because nothing ever opened the
    file it produced.
    """
    from mvr.enrichment.feature_extraction import sort_keyframes_deterministically
    from mvr.indexing.task05a import load_canonical_mapping

    keyframes = load_canonical_mapping(output)
    sort_keyframes_deterministically(output)
    return len(keyframes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build keyframe_mapping.csv from AIC organizer data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Raw organizer download, or a normalized tree; both are searched",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--videos", nargs="+", default=None,
        help="Restrict to these video ids (default: every video with a mapping CSV)",
    )
    parser.add_argument(
        "--shots-json", type=Path, default=None,
        help=(
            "shots.json from the team's ingestion. Fills shot_id from real "
            "shot detection; without it each organizer keyframe is its own "
            "shot, which disables OCR dedupe within a shot but never groups "
            "frames that do not belong together."
        ),
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip loading the result back through Task05A and feature extraction",
    )
    args = parser.parse_args()

    from mvr.evaluation.benchmark import build_aic_keyframe_mapping_csv

    if not args.dataset_root.is_dir():
        raise SystemExit(f"khong tim thay dataset root: {args.dataset_root}")

    video_ids = args.videos or discover_video_ids(args.dataset_root)
    if not video_ids:
        raise SystemExit(
            f"khong tim thay map_keyframes CSV nao duoi {args.dataset_root}\n"
            f"  raw       : <root>/map-keyframes-aic25-b1/map-keyframes/*.csv\n"
            f"  normalized: <root>/map_keyframes/*.csv"
        )
    print(f"video co mapping CSV : {len(video_ids)}")

    boundaries = None
    if args.shots_json is not None:
        boundaries = load_shot_boundaries(args.shots_json)
        print(f"shot detection       : {len(boundaries)} video tu {args.shots_json}")
    else:
        print("shot detection       : khong co -> moi keyframe la mot shot rieng")

    started = time.perf_counter()
    output = build_aic_keyframe_mapping_csv(
        args.dataset_root, args.output, video_ids, shot_boundaries=boundaries
    )
    elapsed = time.perf_counter() - started

    import csv

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    videos_with_rows = {row["video_id"] for row in rows}

    print()
    print(f"dong ghi ra          : {len(rows):,} trong {elapsed:.1f}s")
    print(f"video co keyframe    : {len(videos_with_rows)}")
    # Not a failure. The organizer ships a mapping for all 873 videos but
    # keyframes for 850, and a video whose JPEGs were never downloaded
    # legitimately contributes nothing. Reported because a number that moves
    # unexpectedly is the first sign the download is incomplete.
    missing = len(video_ids) - len(videos_with_rows)
    if missing:
        print(f"video khong co anh   : {missing} (co mapping, chua tai keyframe)")

    if not args.no_verify:
        loaded = _verify(output)
        print(f"kiem tra lai         : Task05A + feature extraction doc duoc {loaded:,} keyframe")

    print(f"-> {output}")


if __name__ == "__main__":
    main()
