"""Measure Q&A retrieval quality without running any answer model.

Answering a Q&A query correctly requires the right video to be retrievable
in the first place. That half can be measured on its own, locally, in
seconds -- no GPU, no LLM, no Kaggle round trip -- which makes it the right
feedback loop for retrieval changes (translation, encoder, index coverage).

For each query it reports the rank at which the *expected* video's best
keyframe appears. Expected videos come from ``--expected`` (a JSON map of
``query_id -> video_id``); build one from
``scripts/find_qna_source_videos.py`` output or from human verification.

Ranks map onto the organizer's own cutoffs (1/5/20/50/100), so the report
states directly whether a query is reachable within the 100-answer budget.

Example:
    python scripts/measure_qna_retrieval.py \\
        --query-dir "D:\\AIC_query" \\
        --index "artifacts\\indexes\\qna_organizer_all" \\
        --expected artifacts/qna/expected_videos.json \\
        --translate
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CUTOFFS = (1, 5, 20, 50, 100)


def rank_of_video(index, encoder, query: str, video_rows: set[int]) -> int | None:
    """1-based rank of the first keyframe belonging to the expected video."""
    vector = encoder.encode_text([query])[0].astype(np.float32).reshape(1, -1)
    _, indices = index.index.search(vector, index.index.ntotal)
    for rank, row in enumerate(indices[0], start=1):
        if int(row) in video_rows:
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Q&A retrieval reachability")
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--expected", type=Path, required=True,
        help='JSON: {"query-p1-19-qa": "L24_V011", ...}',
    )
    parser.add_argument("--pattern", default="*-qa.txt")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--allow-translation-download", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    from mvr.enrichment.embeddings.openclip import OpenClipEncoder
    from mvr.enrichment.qna_translation import (
        HelsinkiViEnTranslator,
        translate_for_retrieval,
    )
    from mvr.indexing.faiss_index import FaissIndex

    sys.path.insert(0, str(Path(__file__).parent))
    from run_qna_pipeline import split_event_and_question

    expected = json.loads(args.expected.read_text(encoding="utf-8"))

    index = FaissIndex.load(args.index)
    encoder = OpenClipEncoder(
        model_name=index.manifest["embedding_model"],
        pretrained=index.manifest["pretrained"],
        device=args.device,
    )
    print(f"index: {index.index.ntotal} vectors, {index.manifest['embedding_model']}"
          f"/{index.manifest['pretrained']}")

    translator = None
    if args.translate:
        translator = HelsinkiViEnTranslator(
            allow_model_download=args.allow_translation_download
        )
        print("translation: ON")
    print()

    rows_by_video: dict[str, set[int]] = {}
    for row, meta in enumerate(index.metadata):
        rows_by_video.setdefault(meta["video_id"], set()).add(row)

    ranks: list[int | None] = []
    for query_file in sorted(args.query_dir.glob(f"**/{args.pattern}")):
        query_id = query_file.stem
        target = expected.get(query_id)
        if not target:
            print(f"{query_id:20} (khong co expected video, bo qua)")
            continue
        if target not in rows_by_video:
            print(f"{query_id:20} {target:12} KHONG CO trong index")
            ranks.append(None)
            continue

        event, _ = split_event_and_question(query_file.read_text(encoding="utf-8"))
        query = translate_for_retrieval(event, translator)
        rank = rank_of_video(index, encoder, query, rows_by_video[target])
        ranks.append(rank)

        if rank is None:
            verdict = "khong tim thay"
        else:
            reached = [k for k in CUTOFFS if rank <= k]
            verdict = f"trong top-{reached[0]}" if reached else "ngoai top-100"
        print(f"{query_id:20} {target:12} hang #{str(rank):>7}  {verdict}")

    measured = [r for r in ranks if r is not None]
    if measured:
        print()
        for cutoff in CUTOFFS:
            hit = sum(1 for r in measured if r <= cutoff)
            print(f"  top-{cutoff:<4} {hit}/{len(ranks)} query")
        print(f"  hang trung vi: {int(np.median(measured))}")


if __name__ == "__main__":
    main()
