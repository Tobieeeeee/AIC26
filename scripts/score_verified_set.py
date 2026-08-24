"""Measure retrieval against the 17 answers verified after round 1.

The organizer published no scores, so without this there is no way to tell
whether a round-2 change helped or hurt. Run it before a change, run it
after, compare the two numbers.

    python scripts/score_verified_set.py                    # tiếng Anh
    python scripts/score_verified_set.py --compare          # Việt vs Anh
    python scripts/score_verified_set.py --save before.json
    python scripts/score_verified_set.py --against before.json

**Scored at the video level**, because that is what the ground truth
actually knows: the operator confirmed a video and one frame inside it,
never the organizer's ``[start_frame, end_frame]`` interval. See
``mvr.evaluation.verified_set`` for why a frame-level number here would be
measuring a guess. ``--frame-window`` gives that stricter reading anyway,
clearly labelled as a proxy.

TRAKE is excluded by default: its retrieval belongs to another owner and a
chain of sub-event frames is not a KIS ranked list. ``--kinds kis qa
trake`` includes it, which measures this system's semantic search on the
TRAKE query text -- a fair question, just not the TRAKE track's score.

One caveat about ``--submissions``: scoring the ROUND-1 submissions with
it is circular. The ground truth was read out of row 1 of those very
files, so every query would report rank 1 and a perfect score. The flag is
for scoring a *new* batch produced later.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mvr.evaluation.verified_set import (  # noqa: E402
    DEFAULT_PATH,
    QueryResult,
    format_report,
    load_verified_set,
    score_query,
    score_set,
)

DEFAULT_INDEX = "artifacts/indexes/qna_organizer_all"
DEFAULT_MAPPING = "artifacts/keyframe_mapping.csv"


def _force_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def rows_from_submission(path: Path) -> list[tuple[str, int]]:
    """``(video_id, frame_idx)`` in rank order from a submission CSV.

    Decoded with ``utf-8-sig`` and stripped, because a third of the
    round-1 files carried an Excel BOM or spaces around the commas.
    """
    text = path.read_bytes().decode("utf-8-sig")
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if not row or not any(row):
            continue
        values = [value.strip() for value in row]
        rows.append((values[0], int(values[1])))
    return rows


def run_retrieval(
    queries,
    searcher,
    *,
    language: str,
    top_k: int,
    overfetch: int,
    per_video: int,
    frame_window: int | None,
) -> list[QueryResult]:
    results = []
    for query in queries:
        text = query.text(language=language)
        if language == "en" and query.query_en is None:
            print(
                f"  ! {query.name}: chưa có bản tiếng Anh, dùng tiếng Việt "
                f"— câu này sẽ kéo điểm xuống",
                file=sys.stderr,
            )
        rows = searcher.search(
            text, top_k=top_k, overfetch=overfetch, per_video=per_video
        )
        results.append(
            score_query(
                query,
                [(r["video_id"], r["frame_idx"]) for r in rows],
                frame_window=frame_window,
            )
        )
    return results


def to_payload(results, *, label: str) -> dict:
    return {
        "label": label,
        "queries": {
            item.query.name: {
                "video_rank": item.video_rank,
                "final_score": item.final_score,
            }
            for item in results
        },
    }


def print_delta(current, previous: dict) -> None:
    """Per-query change against a saved run. This is the whole point."""
    before = previous.get("queries", {})
    lines = [
        "",
        f"so với {previous.get('label', 'lần trước')}:",
        f"{'câu':<20} {'trước':>7} {'sau':>7} {'đổi':>8}",
        "-" * 46,
    ]
    moved = 0
    for item in sorted(current, key=lambda r: r.query.name):
        old = before.get(item.query.name)
        if old is None:
            lines.append(f"{item.query.name:<20} {'—':>7} {item.final_score:>7.4f}  (mới)")
            continue
        delta = item.final_score - old["final_score"]
        if abs(delta) < 1e-9:
            continue
        moved += 1
        arrow = "↑" if delta > 0 else "↓"
        lines.append(
            f"{item.query.name:<20} {old['final_score']:>7.4f} "
            f"{item.final_score:>7.4f} {delta:>+8.4f} {arrow}"
        )
    if not moved:
        lines.append("(không câu nào đổi điểm)")
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--language", choices=("en", "vi"), default="en",
        help="Ngôn ngữ gửi vào encoder. Mặc định tiếng Anh: encoder là "
             "OpenCLIP tiếng Anh thuần, câu tiếng Việt đo được 0.0000.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Chấm cả hai ngôn ngữ trong một lần nạp model.",
    )
    parser.add_argument(
        "--kinds", nargs="+", default=["kis", "qa"],
        help="Track nào được chấm. Mặc định bỏ trake (người khác phụ trách).",
    )
    parser.add_argument(
        "--submissions", type=Path, default=None,
        help="Chấm các file CSV có sẵn thay vì chạy truy xuất. KHÔNG dùng "
             "với thư mục vòng 1: ground truth lấy từ dòng 1 các file đó.",
    )
    parser.add_argument("--index", type=Path, default=REPOSITORY_ROOT / DEFAULT_INDEX)
    parser.add_argument("--mapping", type=Path, default=REPOSITORY_ROOT / DEFAULT_MAPPING)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--overfetch", type=int, default=1500)
    parser.add_argument(
        "--per-video", type=int, default=3,
        help="Số slot tối đa một video được chiếm. 0 = không giới hạn.",
    )
    parser.add_argument(
        "--frame-window", type=int, default=None,
        help="Đọc chặt hơn: chỉ tính đúng khi frame nằm trong "
             "frame_idx ± N. Đây là ƯỚC LƯỢNG, không phải luật BTC.",
    )
    parser.add_argument("--save", type=Path, default=None, help="Ghi kết quả để so lần sau")
    parser.add_argument("--against", type=Path, default=None, help="So với một file --save trước đó")
    return parser


def main() -> None:
    _force_utf8_output()
    args = build_parser().parse_args()

    queries = load_verified_set(args.ground_truth, kinds=args.kinds)
    if not queries:
        raise SystemExit(f"{args.ground_truth}: không có câu nào khớp --kinds {args.kinds}")
    print(f"Ground truth: {len(queries)} câu ({args.ground_truth})", file=sys.stderr)

    runs: list[tuple[str, list[QueryResult]]] = []

    if args.submissions is not None:
        results = []
        for query in queries:
            path = args.submissions / f"{query.name}.csv"
            if not path.exists():
                print(f"  ! thiếu {path.name}, bỏ qua", file=sys.stderr)
                continue
            results.append(
                score_query(
                    query,
                    rows_from_submission(path),
                    frame_window=args.frame_window,
                )
            )
        runs.append((f"submissions {args.submissions.name}", results))
    else:
        from answer_queries import Searcher

        start = time.perf_counter()
        searcher = Searcher(args.index, args.mapping, args.device)
        print(f"Nạp xong sau {time.perf_counter() - start:.1f}s\n", file=sys.stderr)

        languages = ("vi", "en") if args.compare else (args.language,)
        for language in languages:
            start = time.perf_counter()
            results = run_retrieval(
                queries,
                searcher,
                language=language,
                top_k=args.top_k,
                overfetch=args.overfetch,
                per_video=args.per_video,
                frame_window=args.frame_window,
            )
            print(
                f"{language}: {len(results)} câu trong "
                f"{time.perf_counter() - start:.1f}s",
                file=sys.stderr,
            )
            runs.append((language, results))

    for label, results in runs:
        window = "" if args.frame_window is None else f", frame ± {args.frame_window}"
        print()
        print(format_report(score_set(results), label=f"[{label}{window}]"))

    if args.frame_window is not None:
        print(
            "\nLƯU Ý: khoảng frame ± N là ước lượng của chúng ta, không phải "
            "khoảng [start,end] của BTC. Chỉ dùng để so tương đối.",
            file=sys.stderr,
        )

    last_label, last_results = runs[-1]
    if args.against is not None:
        print_delta(last_results, json.loads(args.against.read_text(encoding="utf-8")))
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(
            json.dumps(to_payload(last_results, label=last_label), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nđã ghi {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
