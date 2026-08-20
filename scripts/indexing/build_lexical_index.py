"""Build the Task05B v1 Elasticsearch lexical index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.indexing.lexical_index import build_lexical_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Task05B default-BM25 Elasticsearch index from Task05A documents."
    )
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--elasticsearch-url", required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--recreate", action="store_true")
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
    result = build_lexical_index(
        client,
        processed_root=args.processed_root,
        index_name=args.index_name,
        recreate=args.recreate,
        manifest_path=args.manifest_path,
    )
    print(
        json.dumps(
            {
                "source_documents": result.source_document_count,
                "lexical_indexable": result.lexical_indexable_count,
                "indexed": result.indexed_count,
                "skipped": result.skipped_count,
                "failed": result.failed_count,
                "missing_ids": len(result.missing_ids),
                "extra_ids": len(result.extra_ids),
                "duplicate_ids": len(result.duplicate_ids),
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
