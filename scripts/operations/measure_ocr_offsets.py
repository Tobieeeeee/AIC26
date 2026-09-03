"""Measure what cheaper OCR sampling costs in recognised text.

OCR reads every keyframe at each temporal offset, so its cost is a direct
multiple of how many offsets it uses. The default three (-1, 0, +1) make it the
single most expensive stage in an enrichment run -- on batch1 the difference
between roughly a day and a third of one.

The neighbours are not decoration: they catch text that motion blur smears in
the anchor frame, and that text feeds Vietnamese lexical search. Whether losing
it matters is a question about this dataset, so this runs both samplings over
one video and reports the difference instead of guessing.

    python scripts/operations/measure_ocr_offsets.py \\
      --mapping-csv ~/enrich_m1/ocr_jobs/<hash>/keyframe_mapping.csv \\
      --video-root ~/enrich_m1/data/raw/batch1 \\
      --video-id L26_V001 \\
      --work-root ~/offset_measurement

Both passes run on the GPU, so expect them to compete with any enrichment run
already using it.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.enrichment.ocr.docker_ocr import process_dataset_docker_ocr
from mvr.enrichment.ocr.offset_comparison import compare_offset_runs
from mvr.enrichment.ocr.scene_text import DEFAULT_TEMPORAL_OFFSETS

LOGGER = logging.getLogger("measure-ocr-offsets")


def parse_offsets(raw: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in raw.replace(" ", "").split(",") if part)
    if not values:
        raise ValueError("offsets must name at least one frame")
    if 0 not in values:
        raise ValueError("offsets must include 0, the anchor keyframe itself")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--video-id",
        required=True,
        help="One video to measure on; a whole collection would cost the hours this exists to save",
    )
    parser.add_argument(
        "--candidate-offsets",
        default="0",
        help="The cheaper sampling to test, comma separated (default: 0, anchor only)",
    )
    args = parser.parse_args(argv)
    try:
        args.candidate_offsets = parse_offsets(args.candidate_offsets)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def write_single_video_mapping(mapping_csv: Path, video_id: str, destination: Path) -> int:
    """Copy one video's rows, and the shots.json the sampler reads beside them."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with mapping_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{mapping_csv} has no header row")
        rows = [row for row in reader if row.get("video_id") == video_id]
        fieldnames = reader.fieldnames

    if not rows:
        raise ValueError(f"{mapping_csv} has no rows for video {video_id!r}")

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    shots = mapping_csv.with_name("shots.json")
    if not shots.is_file():
        raise FileNotFoundError(f"the sampler needs shot boundaries at {shots}")
    shutil.copy2(shots, destination.with_name("shots.json"))
    return len(rows)


def _run(mapping: Path, video_root: Path, work_root: Path, label: str, offsets) -> float:
    output_dir = work_root / label / "output"
    temp_dir = work_root / label / "runtime"
    LOGGER.info("running OCR with offsets %s", list(offsets))
    started = time.perf_counter()
    process_dataset_docker_ocr(
        mapping,
        video_root,
        output_dir,
        temp_dir,
        temporal_offsets=offsets,
    )
    return time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    try:
        from dotenv import load_dotenv

        load_dotenv(REPOSITORY_ROOT / ".env", override=False)
    except ImportError:
        pass

    work_root = args.work_root
    mapping = work_root / "keyframe_mapping.csv"
    try:
        keyframes = write_single_video_mapping(args.mapping_csv, args.video_id, mapping)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    LOGGER.info("measuring %s over %d keyframes", args.video_id, keyframes)
    baseline_seconds = _run(
        mapping, args.video_root, work_root, "baseline", DEFAULT_TEMPORAL_OFFSETS
    )
    candidate_seconds = _run(
        mapping, args.video_root, work_root, "candidate", args.candidate_offsets
    )

    report = compare_offset_runs(
        work_root / "baseline" / "output", work_root / "candidate" / "output"
    )
    report.update(
        {
            "video_id": args.video_id,
            "keyframes": keyframes,
            "baseline_offsets": list(DEFAULT_TEMPORAL_OFFSETS),
            "candidate_offsets": list(args.candidate_offsets),
            "baseline_seconds": round(baseline_seconds, 1),
            "candidate_seconds": round(candidate_seconds, 1),
            "speedup": round(baseline_seconds / candidate_seconds, 2)
            if candidate_seconds
            else None,
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
