"""Run OCR candidate -> canonical consolidation from the command line.

All paths are supplied as arguments so the script is portable across machines.
Example:
    python run_canonical.py \
        --keyframe-mapping artifacts/<run>/ingest/keyframe_mapping.csv \
        --candidates artifacts/<run>/candidates \
        --output artifacts/<run>/ocr_output
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mvr.enrichment.ocr.docker_ocr import (
    load_keyframes,
    process_candidates_to_canonical,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keyframe-mapping",
        type=Path,
        required=True,
        help="Path to keyframe_mapping.csv produced by the ingest stage.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Directory of OCR candidate JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write canonical OCR output.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.80,
        help="Phrase similarity threshold for temporal consolidation.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.30,
        help="IoU threshold for spatial merge within a frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.keyframe_mapping.is_file():
        raise SystemExit(f"keyframe mapping not found: {args.keyframe_mapping}")
    if not args.candidates.is_dir():
        raise SystemExit(f"candidates directory not found: {args.candidates}")

    keyframes = load_keyframes(args.keyframe_mapping)
    process_candidates_to_canonical(
        args.candidates,
        args.output,
        keyframes,
        similarity_threshold=args.similarity_threshold,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()
