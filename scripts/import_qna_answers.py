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
    from mvr.submission.qna_export import MAX_PREDICTIONS_PER_QUERY

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
            temporal_offset=item.temporal_offset,
            expanded_from=item.expanded_from,
        )
        for item in batch.items
    ]

    merged = merge_answer_batch(candidates, args.answers, query_id=batch.query_id)
    dropped = len(candidates) - len(merged)
    if dropped:
        print(f"(dropped {dropped} candidate(s) whose model output did not parse)")

    for candidate in merged:
        origin = (
            "" if candidate.expanded_from is None
            else f" [{candidate.temporal_offset:+d} from {candidate.expanded_from}]"
        )
        print(
            f"#{candidate.rank} {candidate.video_id} frame={candidate.frame_id} "
            f"score={candidate.score:.4f} answer={candidate.answer!r} "
            f"source={candidate.answer_source}{origin}"
        )

    # The 100-answer budget is the scored resource, so say plainly how much
    # of it survived: R@100 is one fifth of the final score and every slot
    # lost to unparseable output is a fifth of a fifth thrown away.
    hedges = sum(1 for c in merged if c.expanded_from is not None)
    print()
    print(
        f"[budget] {len(merged)}/{MAX_PREDICTIONS_PER_QUERY} slots used "
        f"({len(merged) - hedges} retrieval hits, {hedges} temporal hedges"
        + (f", {dropped} lost to unparsed output" if dropped else "")
        + ")"
    )

    if args.output is not None:
        from mvr.submission.qna_export import (
            InvalidQnaPredictionError,
            export_local_verification_csv,
            export_local_verification_json,
            validate_qna_predictions,
        )

        try:
            predictions = validate_qna_predictions(merged)
        except InvalidQnaPredictionError as exc:
            raise SystemExit(f"predictions failed contract validation: {exc}") from exc

        exporter = (
            export_local_verification_csv
            if args.output.suffix.lower() == ".csv"
            else export_local_verification_json
        )
        exporter(predictions, args.output, query_id=batch.query_id)
        print(f"wrote {len(predictions)} validated predictions to {args.output}")


if __name__ == "__main__":
    main()
