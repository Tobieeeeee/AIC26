"""Merge GPU-generated Q&A answers back into local retrieval candidates.

Standalone and dependency-light on purpose: it needs neither the FAISS
index, nor OpenCLIP, nor OCR, nor a model -- the prompt batch already
carries every retrieval field, so this step is instant and runs anywhere.

Example:
    python scripts/import_qna_answers.py \\
        --prompts artifacts/qna/query-p1-19-qa.prompts.json \\
        --answers artifacts/qna/query-p1-19-qa.answers.json \\
        --output  artifacts/qna/query-p1-19-qa.predictions.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge batch Q&A answers into predictions")
    parser.add_argument("--prompts", type=Path, required=True, help="prompts.json from --export-prompts")
    parser.add_argument("--answers", type=Path, required=True, help="answers.json produced on the GPU host")
    parser.add_argument("--output", type=Path, default=None, help="Write validated predictions JSON here")
    args = parser.parse_args()

    from mvr.enrichment.qna_batch import merge_answer_batch, read_prompt_batch
    from mvr.retrieval.qna import QnaCandidate

    batch = read_prompt_batch(args.prompts)
    print(f"[query_id] {batch.query_id}")
    print(f"[question] {batch.question}")

    # Reconstruct the exact retrieval candidates from the prompt batch --
    # no need to re-run retrieval or OCR just to attach answers.
    candidates = [
        QnaCandidate(
            rank=item.index + 1,
            video_id=item.video_id,
            frame_id=item.frame_id,
            score=item.score,
            keyframe_id=item.keyframe_id,
            timestamp=item.timestamp,
            answer="PENDING_BATCH_ANSWER",
            answer_source="mixed",
        )
        for item in batch.items
    ]

    merged = merge_answer_batch(candidates, args.answers, query_id=batch.query_id)
    dropped = len(candidates) - len(merged)
    if dropped:
        print(f"(dropped {dropped} candidate(s) whose model output did not parse)")

    for candidate in merged:
        print(
            f"#{candidate.rank} {candidate.video_id} frame={candidate.frame_id} "
            f"score={candidate.score:.4f} answer={candidate.answer!r} "
            f"source={candidate.answer_source}"
        )

    if args.output is not None:
        import json

        from mvr.submission.kis_export import MAX_PREDICTIONS_PER_QUERY

        if len(merged) > MAX_PREDICTIONS_PER_QUERY:
            raise SystemExit(
                f"{len(merged)} predictions exceeds the {MAX_PREDICTIONS_PER_QUERY}-per-query limit"
            )
        payload = {
            "format": "local_verification_export",
            "note": (
                "Local debug export for internal verification only. "
                "NOT the official organizer submission format."
            ),
            "query_id": batch.query_id,
            "question": batch.question,
            "predictions": [
                {
                    "rank": c.rank,
                    "video_id": c.video_id,
                    "frame_id": c.frame_id,
                    "answer": c.answer,
                    "answer_source": c.answer_source,
                }
                for c in merged
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {len(merged)} predictions to {args.output}")


if __name__ == "__main__":
    main()
