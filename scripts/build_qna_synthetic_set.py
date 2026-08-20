"""Build a synthetic Q&A benchmark by reading frames we already have.

Real ground truth is capped at three queries, permanently -- p2/p3 answers
lie outside the data we hold -- and only one of the three is currently
scorable. Every decision about answering is therefore being made on n=1.

This samples keyframes from the search index, runs OCR on them, and derives
questions whose answers are spans of that text. Because the frame is the
*input*, `video_id` and `frame_id` are correct by construction: no search,
no human confirmation, nothing to get wrong.

Read `mvr.evaluation.qna_synthetic` before using the output. In short: this
measures **answer extraction**, not retrieval and not human-judged
correctness. Scores from it are never comparable with scores on the three
real queries.

Example:
    python scripts/build_qna_synthetic_set.py \\
        --index artifacts/indexes/qna_organizer_all \\
        --output artifacts/qna/synthetic_set.json \\
        --frames 400
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a synthetic Q&A benchmark")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frames", type=int, default=400,
        help=(
            "How many keyframes to OCR. Most yield nothing -- only text "
            "matching a rule produces a case -- so sample generously. "
            "~6.7s per frame on CPU."
        ),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Falls back to this registry when the index has no image_path",
    )
    parser.add_argument(
        "--max-occurrences", type=int, default=3,
        help=(
            "Drop an answer seen on more than this many frames. A banner like "
            "'THOI SU' appears everywhere, and the scoring rule would mark a "
            "right answer wrong for pointing at the wrong frame."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    from PIL import Image

    from mvr.enrichment.ocr.easyocr_engine import EasyOcrEngine
    from mvr.evaluation.qna_synthetic import drop_ambiguous_cases, generate_cases
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.ingestion.aic_dataset import AicDatasetRegistry, MissingFrameMappingError

    index = FaissIndex.load(args.index)
    registry = (
        AicDatasetRegistry(args.dataset_root) if args.dataset_root is not None else None
    )

    entries = [e for e in index.metadata if e.get("image_path")]
    random.Random(args.seed).shuffle(entries)
    entries = entries[: args.frames]
    print(f"lay mau {len(entries)} keyframe tu {len(index.metadata)} (seed {args.seed})")

    ocr = EasyOcrEngine()
    cases = []
    unresolved = 0
    for position, entry in enumerate(entries, start=1):
        video_id, keyframe_id = str(entry["video_id"]), str(entry["frame_id"])

        # The competition identity has to come from a verified mapping. A
        # keyframe we cannot resolve is skipped, never guessed -- a fabricated
        # frame_id would poison the very thing this set exists to be.
        frame_id = None
        if registry is not None:
            try:
                frame_id = registry.resolve_frame(video_id, keyframe_id).frame_idx
            except MissingFrameMappingError:
                frame_id = None
        if frame_id is None:
            unresolved += 1
            continue

        try:
            image = Image.open(entry["image_path"]).convert("RGB")
        except OSError:
            continue
        text = ocr.read_text(image)
        found = generate_cases(
            text, video_id=video_id, keyframe_id=keyframe_id, frame_id=frame_id
        )
        cases.extend(found)
        if position % 25 == 0 or found:
            print(f"  [{position}/{len(entries)}] {video_id}/{keyframe_id}: "
                  f"{len(found)} case  (tong {len(cases)})")

    kept = drop_ambiguous_cases(cases, max_occurrences=args.max_occurrences)
    dropped = len(cases) - len(kept)

    payload = {
        "format": "mvr-qna-synthetic-set-v1",
        "note": (
            "Ground truth built backwards from known frames. Measures ANSWER "
            "EXTRACTION only -- not retrieval, not human-judged correctness. "
            "Scores are not comparable with the real ground-truth queries. "
            "See mvr.evaluation.qna_synthetic."
        ),
        "index": str(args.index),
        "frames_sampled": len(entries),
        "seed": args.seed,
        "max_occurrences": args.max_occurrences,
        "cases": [
            {
                "video_id": c.video_id,
                "keyframe_id": c.keyframe_id,
                "frame_id": c.frame_id,
                "question": c.question,
                "answer": c.answer,
                "ocr_text": c.ocr_text,
                "template": c.template,
            }
            for c in kept
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print(f"sinh duoc : {len(cases)} case")
    print(f"bi loai   : {dropped} (dap an lap tren >{args.max_occurrences} frame)")
    print(f"khong resolve duoc frame_id: {unresolved} keyframe (bo qua, khong doan)")
    print(f"GIU LAI   : {len(kept)} case -> {args.output}")
    by_template: dict[str, int] = {}
    for case in kept:
        by_template[case.template] = by_template.get(case.template, 0) + 1
    for template, count in sorted(by_template.items()):
        print(f"   {template}: {count}")


if __name__ == "__main__":
    main()
