"""Score Q&A predictions against human-verified ground truth.

Applies the organizer's documented rule through
``mvr.evaluation.aic_qna``: per-answer R-Score requires the video, the
frame interval **and** the answer to match; ``R@k`` for k in 1/5/20/50/100
takes the best R-Score among the first k answers; Final Score is their
mean.

Two things this report is careful about, because both are easy to
overstate:

- The answer comparison is ``qna_answer_matches``, a documented
  *approximation* of the organizer's semantic judging (normalization plus
  VI/EN number-word equivalence). Scores here are an internal estimate,
  never the organizer's judged result.
- Retrieval and answering are reported separately. A query can fail
  because the right frame was never retrieved or because the answer was
  wrong, and those call for completely different fixes -- a single Final
  Score hides which.

Example:
    python scripts/benchmark_qna.py \\
        --ground-truth artifacts/qna/ground_truth.json \\
        --predictions-dir artifacts/qna_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_predictions(path: Path) -> list[tuple[str, int, str]]:
    """Read (video_id, frame_id, answer) triples from a batch artifact.

    Accepts either a merged predictions JSON or a prompt batch whose
    answers have not been filled in yet; the latter yields the retrieval
    identity with a placeholder answer, which is exactly what the
    retrieval-only half of the report needs.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "predictions" in payload:
        return [
            (row["video_id"], int(row["frame_id"]), row.get("answer", ""))
            for row in payload["predictions"]
        ]
    if payload.get("format") == "mvr-qna-prompt-batch-v1":
        return [
            (row["video_id"], int(row["frame_id"]), "")
            for row in payload["items"]
        ]
    raise ValueError(f"{path}: unrecognized prediction artifact")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Q&A against ground truth")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument(
        "--suffix", default=None,
        help="Artifact suffix to read (default: predictions if present, else prompts)",
    )
    args = parser.parse_args()

    from mvr.evaluation.aic_qna import (
        QNA_CUTOFFS,
        QnaAnswer,
        QnaGroundTruth,
        evaluate_qna,
        qna_answer_matches_lenient,
        qna_answer_matches,
    )

    ground_truth = {
        k: v for k, v in
        json.loads(args.ground_truth.read_text(encoding="utf-8")).items()
        if not k.startswith("_")
    }
    print(f"ground truth: {len(ground_truth)} query")
    print(f"predictions : {args.predictions_dir}")
    print()

    finals: list[float] = []
    finals_lenient: list[float] = []
    frame_hits = 0
    answer_hits = 0
    scored = 0

    for query_id, entry in sorted(ground_truth.items()):
        candidates = None
        for suffix in ([args.suffix] if args.suffix else
                       [".predictions.json", ".answers.json", ".prompts.json"]):
            path = args.predictions_dir / f"{query_id}{suffix}"
            if path.is_file():
                try:
                    candidates = load_predictions(path)
                    break
                except ValueError:
                    continue
        if candidates is None:
            print(f"{query_id:20} (khong tim thay file du doan, bo qua)")
            continue

        gt = QnaGroundTruth(
            video_id=entry["video_id"],
            start_frame=entry["start_frame"],
            end_frame=entry["end_frame"],
            answer=entry["answer"],
        )
        answers = [QnaAnswer(v, f, a) for v, f, a in candidates]
        # Two bounds, because one number cannot say "somewhere between".
        # strict  = normalized equality (+ number words, + a leading
        #           Vietnamese administrative-unit word)
        # lenient = the answer appears inside the reply, or vice versa
        # The organizer's human judge sits somewhere in between.
        score = evaluate_qna(answers, gt)
        score_lenient = evaluate_qna(answers, gt, matcher=qna_answer_matches_lenient)
        finals.append(score.final_score)
        finals_lenient.append(score_lenient.final_score)
        scored += 1

        # Split the failure mode: a wrong frame and a wrong answer need
        # entirely different fixes.
        frame_ok = next(
            (i + 1 for i, (v, f, _) in enumerate(candidates)
             if v == gt.video_id and gt.start_frame <= f <= gt.end_frame),
            None,
        )
        answer_ok = any(qna_answer_matches(a, gt.answer) for _, _, a in candidates)
        frame_hits += 1 if frame_ok else 0
        answer_hits += 1 if answer_ok else 0

        recall = " ".join(f"R@{k}={score.recall_at_k[k]:.0f}" for k in QNA_CUTOFFS)
        spread = (
            "" if score_lenient.final_score == score.final_score
            else f"  (lenient={score_lenient.final_score:.3f})"
        )
        print(f"{query_id:20} Final={score.final_score:.3f}{spread}  {recall}")
        print(f"{'':20} frame dung: {'hang #' + str(frame_ok) if frame_ok else 'KHONG CO'}"
              f"   |   dap an dung o dau do: {'co' if answer_ok else 'khong'}")

    if not scored:
        raise SystemExit("khong cham duoc query nao")

    print()
    print("=" * 62)
    strict_mean = sum(finals) / len(finals)
    lenient_mean = sum(finals_lenient) / len(finals_lenient)
    print(f"  Mean Final Score  strict          : {strict_mean:.4f}")
    print(f"  Mean Final Score  lenient (can tren): {lenient_mean:.4f}")
    print(f"  Query co frame dung trong top-100   : {frame_hits}/{scored}")
    print(f"  Query co dap an dung o bat ky dau   : {answer_hits}/{scored}")
    print("=" * 62)
    print("Luu y: BTC cham theo NGU NGHIA (nguoi cham). Hai con so tren la hai")
    print("can cua uoc luong noi bo -- diem that nam giua, khong phai diem chinh thuc.")


if __name__ == "__main__":
    main()
