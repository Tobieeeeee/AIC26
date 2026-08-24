"""Answer a package of competition queries and write submission CSVs.

One command, one model load. The organizer hands out a folder of text
files -- `query-1-kis.txt`, `query-3-qa.txt` -- and wants one `.csv` back
per query. This reads the folder, dispatches on the filename suffix, and
writes files that match their published format exactly.

Loading the CLIP text encoder takes ~38 seconds and a query takes ~2.5
seconds on CPU. Running a script per query would spend 45 seconds of every
minute loading a model that is already loaded, so everything here happens
inside one process.

**KIS** is fully automatic: search, resolve each keyframe to its
competition frame index, write `<video_id>,<frame_idx>`.

**Q&A** cannot be: answering "how many people are in this frame" needs a
vision-language model, and there is none available here. So it writes two
files -- a submission with the frames found and the answer column empty,
which is valid and keeps the upload slot, and a review file listing each
frame's image path so a person can look at the frame and type the answer.
`--from-review` turns the filled-in review file back into a submission.

**Queries must be English.** The index encoder is OpenCLIP
`ViT-B-32/openai`, English-only. Measured on the three human-verified p1
ground truths: the original Vietnamese scores **0.0000**, hand-written
English scores **0.3333**. Put the English beside each query as
`query-1-kis.en.txt` and this uses it; without one it warns and searches
the Vietnamese, which will score nothing.

Write the English as a description of what the frame LOOKS LIKE, not as a
translation of the question. CLIP matches pictures to visual descriptions,
and proper nouns mean nothing to it::

    Vietnamese  "2 câu thơ ca ngợi anh hùng Nguyễn Trung Trực..."
    good        "a stone tablet with two lines of Vietnamese poetry
                 inside a traditional temple"

**TRAKE** files are skipped: that track belongs to someone else, and two
people writing the same submission would be worse than one.

The keyframe id in the index (`"001"`) is NOT the competition frame index.
It is the JPEG stem, and the organizer's 166,550 keyframes carry only 632
distinct ones. Every row goes through `KeyframeFrameIndexResolver`, keyed
on `(video_id, keyframe_id)`, or it is dropped -- never guessed.

Two ways to run it::

    python scripts/answer_queries.py --queries artifacts/queries \\
        --output artifacts/submissions            # cả gói, một lượt

    python scripts/answer_queries.py --interactive     # thử và sửa câu mô tả

``--interactive`` exists because the wording of the English description is
the difference between 0.0000 and 0.3333. Re-running the batch to test one
rephrasing costs 40 seconds of model loading; inside the loop it costs 0.2,
so a person can iterate on the variable that actually decides the score.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mvr.submission.lint import (  # noqa: E402
    format_report,
    lint_directory,
    repair_file,
)
from mvr.submission.official import (  # noqa: E402
    MAX_ROWS,
    SubmissionFormatError,
    submission_filename,
    write_kis_submission,
    write_qna_submission,
)
from mvr.retrieval.kis import resolve_keyframe_ref  # noqa: E402

DEFAULT_INDEX = "artifacts/indexes/qna_organizer_all"
DEFAULT_MAPPING = "artifacts/keyframe_mapping.csv"

#: Columns of the human review file. Not a submission -- `image_path` lets
#: the reviewer open the frame, and `answer` is the column they fill.
REVIEW_COLUMNS = ("rank", "video_id", "frame_idx", "answer", "score", "image_path")


def _force_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def resolve_output_name(typed: str, kind: str) -> str:
    """What the operator types after ``:w kis`` -> the submission filename.

    The submission carries the query file's own name, so typing that name
    is the direct path. A bare id (``23``, ``p1-23``) is accepted too,
    because mid-competition an operator types a fragment, not a filename.
    """
    return submission_filename(typed.strip(), kind)


def insert_top_row(rows: list[dict], video_id: str, frame_idx: int) -> list[dict]:
    """Put a manually-identified frame at rank 1.

    Used when the operator has already found the right keyframe by eye and
    typed it in with ``:top`` -- searching again would be slower and could
    push the known-correct frame back down the list. A row for the same
    ``(video_id, frame_idx)`` already present is moved rather than
    duplicated: a repeated line wastes one of the 100 slots for nothing.
    """
    existing = next(
        (
            row
            for row in rows
            if (row["video_id"], row["frame_idx"]) == (video_id, frame_idx)
        ),
        None,
    )
    rest = [row for row in rows if row is not existing]
    top = {
        "rank": 1,
        "video_id": video_id,
        "frame_idx": frame_idx,
        "score": existing["score"] if existing else None,
        "image_path": existing["image_path"] if existing else "",
    }
    out = [top, *rest]
    for i, row in enumerate(out, start=1):
        row["rank"] = i
    return out


def classify(path: Path) -> str | None:
    """`query-3-qa.txt` -> `qa`. Unknown suffixes are reported, not guessed.

    `query-3-qa.en.txt` returns ``"companion"``: it is the English text for
    another query, not a query of its own, and treating it as one would
    write a second submission file under a name the organizer never asked
    for.
    """
    stem = path.stem.lower()
    if stem.endswith(".en"):
        return "companion"
    for kind in ("kis", "qa", "trake"):
        if stem.endswith(f"-{kind}") or stem.endswith(f"_{kind}"):
            return kind
    return None


class Searcher:
    """Index, encoder and resolver, loaded once and reused for every query."""

    def __init__(self, index_dir: Path, mapping_csv: Path, device: str) -> None:
        from mvr.enrichment.embeddings.openclip import OpenClipEncoder
        from mvr.indexing.faiss_index import FaissIndex
        from mvr.retrieval.kis import KeyframeFrameIndexResolver
        from mvr.retrieval.semantic import SemanticRetriever

        start = time.perf_counter()
        index = FaissIndex.load(index_dir)
        print(
            f"Index  : {index.manifest.get('vector_count')} vector, "
            f"{index.manifest.get('embedding_model')}/"
            f"{index.manifest.get('pretrained')}"
            f"  ({time.perf_counter() - start:.1f}s)",
            file=sys.stderr,
        )

        start = time.perf_counter()
        # Model and pretrained tag come from the manifest, never a constant:
        # a ViT-B-32 index queried by an RN50 text encoder returns ranked,
        # plausible, meaningless results instead of an error.
        encoder = OpenClipEncoder(
            model_name=index.manifest["embedding_model"],
            pretrained=index.manifest["pretrained"],
            device=device,
        )
        print(f"Encoder: nạp xong ({time.perf_counter() - start:.1f}s)", file=sys.stderr)

        start = time.perf_counter()
        self.resolver = KeyframeFrameIndexResolver.from_csv(mapping_csv)
        print(
            f"Mapping: {len(self.resolver._mapping)} keyframe "
            f"({time.perf_counter() - start:.1f}s)",
            file=sys.stderr,
        )
        self.retriever = SemanticRetriever(encoder, index)

    def search(
        self, text: str, *, top_k: int, overfetch: int, per_video: int
    ) -> list[dict]:
        """Ranked, de-duplicated rows carrying the competition frame index.

        Overfetches because two keyframes of one video can resolve to the
        same frame, and because a keyframe missing from the mapping is
        dropped rather than guessed -- both shrink the list.

        ``per_video`` caps how many slots one video may take. Uncapped, a
        measured 100-row KIS submission covered just 21 videos with one
        taking 33 slots; a wrong video scores zero for every slot it holds,
        so concentration is a bet rather than a hedge. At 3 the same
        submission covers ~54.
        """
        rows: list[dict] = []
        seen: set[tuple[str, int]] = set()
        used: dict[str, int] = {}
        for hit in self.retriever.search(text.strip(), top_k=overfetch):
            frame_idx = self.resolver.resolve(hit.video_id, hit.frame_id)
            if frame_idx is None:
                continue
            key = (hit.video_id, frame_idx)
            if key in seen:
                continue
            if per_video and used.get(hit.video_id, 0) >= per_video:
                continue
            seen.add(key)
            used[hit.video_id] = used.get(hit.video_id, 0) + 1
            rows.append(
                {
                    "rank": len(rows) + 1,
                    "video_id": hit.video_id,
                    "frame_idx": frame_idx,
                    "score": round(float(hit.score), 4),
                    "image_path": hit.image_path,
                }
            )
            if len(rows) >= top_k:
                break
        return rows


def write_review_file(rows: list[dict], path: Path) -> Path:
    """The working file a person fills in. Not a submission."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "answer": ""})
    return path


def submission_from_review(review_path: Path, output_path: Path) -> Path:
    """Rebuild a Q&A submission from a review file a person filled in.

    Rows keep their order, so the ranking the retrieval produced survives
    the round trip through a spreadsheet.
    """
    with Path(review_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    missing = {"video_id", "frame_idx", "answer"} - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{review_path}: thiếu cột {sorted(missing)}")
    return write_qna_submission(
        [(r["video_id"], r["frame_idx"], r.get("answer") or "") for r in rows],
        output_path,
    )


HELP = """
Lệnh:
  <mô tả tiếng Anh>        tìm và in top-10
  :n <số>                  đổi số kết quả hiển thị (mặc định 10)
  :v <số>                  đổi tối đa dòng mỗi video (0 = bỏ giới hạn)
  :w kis <stt>             ghi kết quả gần nhất -> query-<stt>-kis.csv
  :w qa <stt>              ghi -> query-<stt>-qa.csv + query-<stt>-qa.review.csv
  :top <video_id>_<keyframe_id> <kis|qa> <stt>
                           chèn keyframe đã biết lên hạng 1 rồi ghi luôn
  :o                       mở ảnh của kết quả hạng 1
  :q                       thoát

Mô tả cái NHÌN THẤY trong khung hình, bỏ tên riêng. CLIP chỉ đọc 77 token.
"""


def _write_submission(rows: list[dict], kind: str, out: Path) -> None:
    """Write the full top-k (not just what's on screen) to ``out``.

    ``kind == "qa"`` also writes the sibling ``.review.csv`` a human fills
    the answer column in -- shared by both ``:w`` and ``:top`` so the two
    commands cannot drift into writing different columns.
    """
    if kind == "kis":
        write_kis_submission([(r["video_id"], r["frame_idx"]) for r in rows], out)
        print(f"đã ghi {out}  ({min(len(rows), MAX_ROWS)} dòng)")
    else:
        write_qna_submission(
            [(r["video_id"], r["frame_idx"], "") for r in rows], out
        )
        review = out.with_suffix("").with_suffix(".review.csv")
        write_review_file(rows, review)
        print(f"đã ghi {out} và {review.name}  (điền cột answer)")


def batch_output_filename(path: Path, kind: str) -> str:
    """The query file's own name with a ``.csv`` extension.

    ``query-p1-23-kis.txt`` -> ``query-p1-23-kis.csv``. The organizer shows
    the rule by example: their pack holds `query-1-kis.txt` and their
    required listing holds `query-1-kis.csv`.

    Falls back to the stem for a name that does not follow the convention:
    one oddly-named query should cost that query, not crash everything
    still queued behind it in the batch.
    """
    try:
        return resolve_output_name(path.stem, kind)
    except SubmissionFormatError:
        return f"{path.stem}.csv"


def interactive(searcher: "Searcher", args) -> int:
    """Try a description, see the frames, fix the wording, try again.

    Exists because the measured gap between a Vietnamese query and an
    English one is 0.0000 against 0.3333 -- the wording of the description
    is the difference between scoring and not. Re-running the batch script
    to test one rephrasing costs 40 seconds of model loading; here it costs
    0.2 seconds, so a person can actually iterate on the variable that
    matters.
    """
    top_n = 10
    per_video = args.per_video
    last: list[dict] = []
    last_query = ""
    print(HELP)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line in (":q", ":quit", ":exit"):
            return 0
        if line in (":h", ":help", "?"):
            print(HELP)
            continue

        if line.startswith(":n "):
            top_n = max(1, int(line[3:].strip() or top_n))
            print(f"hiển thị {top_n} kết quả")
            continue

        if line.startswith(":v "):
            per_video = max(0, int(line[3:].strip() or per_video))
            print(f"tối đa {per_video or 'không giới hạn'} dòng mỗi video")
            continue

        if line == ":o":
            if not last:
                print("chưa có kết quả nào")
                continue
            path = last[0]["image_path"]
            print(f"mở {path}")
            try:
                import os

                os.startfile(path)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - platform dependent
                print(f"không mở được: {exc}")
            continue

        if line.startswith(":w "):
            parts = line[3:].split(maxsplit=1)
            if len(parts) != 2 or parts[0] not in ("kis", "qa"):
                print("dùng: :w kis <stt>   hoặc   :w qa <stt>")
                continue
            if not last:
                print("chưa có kết quả nào để ghi")
                continue
            kind, typed = parts[0], parts[1].strip()
            try:
                filename = resolve_output_name(typed, kind)
            except SubmissionFormatError as exc:
                print(f"tên không hợp lệ: {exc}")
                continue
            _write_submission(last, kind, args.output / filename)
            continue

        if line.startswith(":top "):
            parts = line[5:].split()
            if len(parts) != 3 or parts[1] not in ("kis", "qa"):
                print("dùng: :top <video_id>_<keyframe_id> <kis|qa> <stt>")
                continue
            ref, kind, typed = parts
            video_id, keyframe_id, frame_idx = resolve_keyframe_ref(
                searcher.resolver, ref
            )
            if frame_idx is None:
                print(
                    f"không tra được {video_id}/{keyframe_id} trong "
                    "keyframe_mapping.csv -- kiểm lại video_id/keyframe_id, "
                    "không đoán"
                )
                continue
            last = insert_top_row(last, video_id, frame_idx)
            print(f"{video_id}/{keyframe_id} -> frame {frame_idx}, đã đưa lên hạng 1")
            try:
                filename = resolve_output_name(typed, kind)
            except SubmissionFormatError as exc:
                print(f"tên không hợp lệ: {exc}")
                continue
            _write_submission(last, kind, args.output / filename)
            continue

        if line.startswith(":"):
            print(f"lệnh lạ: {line}. Gõ :h để xem danh sách.")
            continue

        start = time.perf_counter()
        last = searcher.search(
            line, top_k=args.top_k, overfetch=args.overfetch, per_video=per_video
        )
        last_query = line
        elapsed = time.perf_counter() - start
        if not last:
            print("không có kết quả nào phân giải được frame")
            continue
        videos = len({r["video_id"] for r in last})
        print(
            f"  {len(last)} dòng / {videos} video  ({elapsed:.2f}s)"
            f"   [{last_query[:50]}]"
        )
        for row in last[:top_n]:
            print(
                f"  {row['rank']:3d}. {row['video_id']:>10}/{row['frame_idx']:<7d}"
                f" {row['score']:.3f}  {row['image_path']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--queries", type=Path, help="thư mục chứa query-*.txt")
    parser.add_argument("--output", type=Path, default=Path("artifacts/submissions"))
    parser.add_argument("--index-dir", type=Path, default=Path(DEFAULT_INDEX))
    parser.add_argument("--mapping-csv", type=Path, default=Path(DEFAULT_MAPPING))
    parser.add_argument("--top-k", type=int, default=MAX_ROWS)
    parser.add_argument(
        "--overfetch",
        type=int,
        default=1500,
        help="số ứng viên lấy trước khi khử trùng lặp và phân giải frame",
    )
    parser.add_argument(
        "--per-video",
        type=int,
        default=3,
        help="tối đa bao nhiêu dòng cho một video (0 = không giới hạn)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="thử và sửa câu mô tả liên tục, model chỉ nạp một lần",
    )
    parser.add_argument(
        "--from-review",
        type=Path,
        help="dựng lại file nộp Q&A từ file review đã điền answer",
    )
    parser.add_argument(
        "--lint",
        type=Path,
        help="kiểm thư mục file nộp trước khi upload",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="với --lint: bỏ BOM và khoảng trắng thừa",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)

    if args.lint:
        files = [
            path
            for path in sorted(Path(args.lint).glob("*.csv"))
            if not path.name.endswith(".review.csv")
        ]
        if args.fix:
            # Repair first, then report what is left: the two whitespace
            # defects are mechanical, and everything still listed afterwards
            # is something a person has to decide about.
            fixed = [path.name for path in files if repair_file(path)]
            if fixed:
                print(f"Đã sửa {len(fixed)} file: {', '.join(fixed)}")
        report = lint_directory(args.lint)
        print(format_report(report, len(files)))
        return 1 if report else 0

    if args.from_review:
        output = args.output
        if output.is_dir() or not output.suffix:
            output = output / (args.from_review.stem.replace(".review", "") + ".csv")
        path = submission_from_review(args.from_review, output)
        print(f"Đã ghi {path}")
        return 0

    if not args.queries and not args.interactive:
        raise SystemExit(
            "cần --queries <thư mục>, --interactive, hoặc --from-review <file>"
        )
    if not args.mapping_csv.is_file():
        raise SystemExit(
            f"không thấy {args.mapping_csv}. Sinh bằng:\n"
            f"  python scripts/build_keyframe_mapping.py "
            f"--dataset-root <data> --output {args.mapping_csv}"
        )

    files = sorted(Path(args.queries).glob("*.txt")) if args.queries else []
    if args.queries and not files:
        raise SystemExit(f"không thấy file .txt nào trong {args.queries}")

    searcher = Searcher(args.index_dir, args.mapping_csv, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    print(file=sys.stderr)

    if args.interactive:
        return interactive(searcher, args)

    written: list[str] = []
    for path in files:
        kind = classify(path)
        if kind is None:
            print(f"BỎ QUA {path.name}: không rõ loại truy vấn", file=sys.stderr)
            continue
        if kind == "companion":
            continue
        if kind == "trake":
            print(f"BỎ QUA {path.name}: TRAKE do người khác phụ trách", file=sys.stderr)
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"BỎ QUA {path.name}: file rỗng", file=sys.stderr)
            continue

        english = path.with_suffix(".en.txt")
        if english.is_file() and english.read_text(encoding="utf-8").strip():
            text = english.read_text(encoding="utf-8").strip()
        else:
            print(
                f"CẢNH BÁO {path.name}: không có {english.name}. Truy vấn tiếng "
                f"Việt vào encoder tiếng Anh cho 0 điểm -- đo trên ground truth "
                f"p1: tiếng Việt 0.0000, tiếng Anh 0.3333.",
                file=sys.stderr,
            )

        start = time.perf_counter()
        rows = searcher.search(
            text,
            top_k=args.top_k,
            overfetch=args.overfetch,
            per_video=args.per_video,
        )
        elapsed = time.perf_counter() - start

        out_name = batch_output_filename(path, kind)
        if kind == "kis":
            out = args.output / out_name
            write_kis_submission([(r["video_id"], r["frame_idx"]) for r in rows], out)
            written.append(out.name)
            print(f"{path.name:24} -> {out.name:24} {len(rows):3d} dòng  {elapsed:.1f}s")
        else:
            out = args.output / out_name
            write_qna_submission(
                [(r["video_id"], r["frame_idx"], "") for r in rows], out
            )
            review = args.output / (out_name[: -len(".csv")] + ".review.csv")
            write_review_file(rows, review)
            written.append(out.name)
            print(
                f"{path.name:24} -> {out.name:24} {len(rows):3d} dòng  {elapsed:.1f}s"
                f"\n{'':24}    {review.name}  (điền cột answer rồi chạy --from-review)"
            )

    print(f"\nĐã ghi {len(written)} file nộp vào {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
