"""Score answer EXTRACTION against the synthetic set.

Read `mvr.evaluation.qna_synthetic` first. This measures one thing: given a
frame's own context and a question about it, does the reasoner return the
right span? It does **not** measure retrieval -- the frame is handed over,
not searched for -- and it does not measure whether the answer is what a
human would say, since the expectation came from OCR and OCR makes mistakes.

Scores here are not comparable with `benchmark_qna.py`, which scores the
three real ground-truth queries end to end. Two different questions.

Because the frame is given, this isolates exactly the half that real ground
truth cannot measure at n=1: prompt wording, model choice, answer
formatting, and where `qna_answer_matches` is too strict or too loose.

Example:
    python scripts/benchmark_qna_synthetic.py \\
        --cases artifacts/qna/synthetic_set.json \\
        --export-prompts artifacts/qna/synthetic.prompts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROMPT_BATCH_FORMAT = "mvr-qna-prompt-batch-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score answer extraction")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--export-prompts", type=Path, default=None,
        help="Write prompts for a GPU host instead of scoring answers now",
    )
    parser.add_argument(
        "--answers", type=Path, default=None,
        help="answers.json produced from the exported prompts; scores when given",
    )
    args = parser.parse_args()

    from mvr.enrichment.qna_answer import MalformedAnswerError, _build_context_block, _parse_answer
    from mvr.enrichment.qna_context import FrameContext
    from mvr.evaluation.aic_qna import qna_answer_matches, qna_answer_matches_lenient

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    if payload.get("format") != "mvr-qna-synthetic-set-v1":
        raise SystemExit(f"unexpected case-set format: {payload.get('format')!r}")
    cases = payload["cases"]
    print(f"bo tong hop: {len(cases)} case tu {payload['frames_sampled']} frame")

    prompts = [
        _build_context_block(
            case["question"],
            FrameContext(
                video_id=case["video_id"],
                frame_id=case["frame_id"],
                ocr_text=case["ocr_text"],
                objects=None,
            ),
        )
        for case in cases
    ]

    if args.export_prompts is not None:
        args.export_prompts.parent.mkdir(parents=True, exist_ok=True)
        args.export_prompts.write_text(
            json.dumps(
                {
                    "format": PROMPT_BATCH_FORMAT,
                    "query_id": "synthetic-extraction",
                    "event_description": "(synthetic answer-extraction benchmark)",
                    "question": "(per-case, see items)",
                    "items": [
                        {
                            "index": index,
                            "video_id": case["video_id"],
                            "frame_id": case["frame_id"],
                            "keyframe_id": case["keyframe_id"],
                            "score": 1.0,
                            "timestamp": None,
                            "prompt": prompt,
                        }
                        for index, (case, prompt) in enumerate(zip(cases, prompts))
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"wrote {len(prompts)} prompts to {args.export_prompts}")
        print("next: generate on GPU, then re-run with --answers")
        return

    if args.answers is None:
        raise SystemExit("pass --export-prompts or --answers")

    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    by_index = {row["index"]: row["raw_output"] for row in answers["items"]}

    strict = lenient = unparsed = 0
    wrong: list[tuple[str, str, str]] = []
    for index, case in enumerate(cases):
        raw = by_index.get(index)
        if raw is None:
            unparsed += 1
            continue
        try:
            produced = _parse_answer(raw).text
        except MalformedAnswerError:
            unparsed += 1
            continue
        if qna_answer_matches(produced, case["answer"]):
            strict += 1
            lenient += 1
        elif qna_answer_matches_lenient(produced, case["answer"]):
            lenient += 1
            wrong.append((case["question"], case["answer"], produced))
        else:
            wrong.append((case["question"], case["answer"], produced))

    scored = len(cases) - unparsed
    print()
    print("=" * 62)
    print(f"  Trich xuat dung  strict  : {strict}/{scored}"
          f" ({strict / scored:.1%})" if scored else "  khong cham duoc case nao")
    print(f"  Trich xuat dung  lenient : {lenient}/{scored}"
          f" ({lenient / scored:.1%})" if scored else "")
    print(f"  Khong parse duoc         : {unparsed}/{len(cases)}")
    print("=" * 62)
    print("Do TRICH XUAT dap an, KHONG do retrieval. Khong so sanh duoc voi")
    print("benchmark_qna.py -- hai cau hoi khac nhau.")

    if wrong:
        print(f"\n{min(len(wrong), 15)} truong hop lech dau tien:")
        for question, expected, produced in wrong[:15]:
            print(f"  {question}")
            print(f"     mong doi : {expected!r}")
            print(f"     model tra: {produced!r}")


if __name__ == "__main__":
    main()
