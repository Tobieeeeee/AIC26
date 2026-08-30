"""Turn the organizer's TRAKE `.txt` into the JSON the pipeline reads.

`run_trake_pipeline.py` wants `{"events": [...], "gap_phrases": [...]}`.
The organizer hands out prose with the events already numbered E1..Ek::

    Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ...
    E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.
    E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ.
    E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân.

Nothing connected the two, so TRAKE was the one track this system could
not answer from the files it was handed -- the events had to be retyped
by hand, mid-run, with the clock going.

    python scripts/build_trake_query_json.py \
        --queries artifacts/queries --output artifacts/queries

The JSON keeps the query file's own stem, because the submission is named
from it: `query-p1-16-trake.txt` -> `query-p1-16-trake.json` ->
`query-p1-16-trake.csv`.

The sentence before E1 is carried as `preamble` and never scored. It sets
the scene rather than naming a moment, so there is no frame it could point
at; `load_query_file` ignores the key.

**The events stay in the source language.** CLIP's text tower is English,
so pass `--translate-query` to the pipeline, or hand-write an English
version the way the KIS track uses `.en.txt` -- measured on the p1 ground
truth, Vietnamese scored 0.0000 and English 0.3333.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mvr.retrieval.trake_events import parse_organizer_trake_query  # noqa: E402


def is_trake_query(path: Path) -> bool:
    """Same suffix convention `answer_queries.classify` dispatches on."""
    stem = path.stem.lower()
    return stem.endswith("-trake") or stem.endswith("_trake")


def convert_file(source: Path, output_dir: Path) -> Path:
    """Write `<output_dir>/<source stem>.json`, return where it landed.

    Names the file in the error, because mid-competition the operator sees
    the error text and nothing else -- a bare "no numbered events" does not
    say which query to go and fix.
    """
    try:
        query = parse_organizer_trake_query(source.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{source.name}: {exc}") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{source.stem}.json"
    destination.write_text(
        json.dumps(
            {
                "events": list(query.events),
                "gap_phrases": list(query.gap_phrases),
                "preamble": query.preamble,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def convert_directory(queries: Path, output_dir: Path) -> list[Path]:
    """Convert every TRAKE query in a pack, skipping the other tracks.

    A failure is reported and stepped over rather than raised: the queries
    behind it in the pack are still worth points.
    """
    written: list[Path] = []
    for source in sorted(Path(queries).glob("*.txt")):
        if not is_trake_query(source):
            continue
        try:
            written.append(convert_file(source, output_dir))
        except ValueError as exc:
            print(f"BỎ QUA {exc}", file=sys.stderr)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chuyển file đề TRAKE (.txt) sang JSON cho run_trake_pipeline.py"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query-file", type=Path, help="một file .txt")
    source.add_argument("--queries", type=Path, help="thư mục đề, chỉ lấy file *-trake.txt")
    parser.add_argument(
        "--output", type=Path, required=True, help="thư mục ghi file .json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if args.query_file:
        written = [convert_file(args.query_file, args.output)]
    else:
        written = convert_directory(args.queries, args.output)
    for path in written:
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"{path.name}  {len(payload['events'])} sự kiện")
    if not written:
        print("không chuyển được file TRAKE nào", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
