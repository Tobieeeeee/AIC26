"""Run a Task05B v1 lexical retrieval verification query."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.retrieval.lexical_bm25 import search_task05b_lexical


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify Task05B default-BM25 lexical retrieval."
    )
    parser.add_argument("--elasticsearch-url", required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise RuntimeError(
            "Elasticsearch client is missing; install the project's search extra"
        ) from exc
    client = Elasticsearch(args.elasticsearch_url)
    results = search_task05b_lexical(
        client,
        args.query,
        index_name=args.index_name,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            [result.to_dict() for result in results],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
