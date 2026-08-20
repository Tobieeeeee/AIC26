"""Recover already-computed OCR from prompt batches on disk.

EasyOCR costs ~6.7s per frame on CPU, and several hundred frames have
already been through it across earlier runs. That text is sitting in
`artifacts/qna*/ *.prompts.json`, embedded in the prompts that were sent to
the GPU host.

Harvesting it turns a 1.8-hour feedback loop into a sub-second one. That is
what makes the synthetic-set rules tunable at all: the first version scored
36% precision, and each diagnosis cost a full OCR run.

Example:
    python scripts/build_qna_ocr_cache.py \\
        --output artifacts/qna/ocr_cache.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest OCR text already computed")
    parser.add_argument(
        "--artifacts", type=Path, default=Path("artifacts"),
        help="Root to scan for */*.prompts.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mvr.evaluation.qna_synthetic import ocr_text_from_prompt

    # (video_id, keyframe_id) -> entry. Later batches win: a frame re-OCR'd
    # in a newer run reflects the current engine and settings.
    entries: dict[tuple[str, str], dict] = {}
    sources = 0
    for path in sorted(args.artifacts.glob("qna*/*.prompts.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "mvr-qna-prompt-batch-v1":
            continue
        sources += 1
        recovered = 0
        for item in payload["items"]:
            text = ocr_text_from_prompt(item["prompt"])
            if text is None:
                continue
            key = (str(item["video_id"]), str(item["keyframe_id"]))
            entries[key] = {
                "video_id": key[0],
                "keyframe_id": key[1],
                "frame_id": int(item["frame_id"]),
                "ocr_text": text,
            }
            recovered += 1
        print(f"  {path.parent.name}/{path.name}: {recovered} frame")

    with_text = sum(1 for e in entries.values() if e["ocr_text"])
    payload = {
        "format": "mvr-qna-ocr-cache-v1",
        "note": (
            "OCR text recovered from prompt batches, not recomputed. An empty "
            "ocr_text means the frame was read and held no text; a frame "
            "absent from this cache was never read."
        ),
        "frames": sorted(entries.values(), key=lambda e: (e["video_id"], e["keyframe_id"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print(f"nguon      : {sources} prompt batch")
    print(f"frame duy nhat: {len(entries)}")
    print(f"co text OCR   : {with_text}")
    print(f"-> {args.output}")
    print()
    print(f"tiet kiem  : ~{len(entries) * 6.7 / 60:.0f} phut OCR moi lan chay lai luat")


if __name__ == "__main__":
    main()
