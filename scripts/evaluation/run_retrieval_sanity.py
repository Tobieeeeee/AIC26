"""Run the GT-free retrieval sanity and regression harness."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.evaluation.sanity_adapters import BM25Adapter, ProviderUnavailableError
from mvr.evaluation.sanity_regression import (
    build_snapshot,
    compare_snapshots,
    load_snapshot,
    save_snapshot,
)
from mvr.evaluation.sanity_reporting import write_run_reports
from mvr.evaluation.sanity_runner import (
    build_run_manifest,
    load_sanity_queries,
    run_sanity_suite,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run GT-free retrieval contract, latency, determinism, and "
            "regression checks."
        )
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation" / "queries" / "sanity_queries.json",
    )
    parser.add_argument("--provider", choices=["bm25"], default="bm25")
    parser.add_argument(
        "--elasticsearch-url",
        default=os.environ.get("MVR_ES_URL", "http://localhost:9200"),
    )
    parser.add_argument(
        "--index-name",
        default=os.environ.get("MVR_ES_INDEX"),
        help="Existing Task05B lexical index name.",
    )
    parser.add_argument(
        "--provider-manifest",
        type=Path,
        default=None,
        help="Optional Task05B manifest used for provenance.",
    )
    parser.add_argument("--top-k", type=positive_int, default=10)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument(
        "--score-tolerance",
        type=non_negative_float,
        default=1e-8,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "retrieval_sanity",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--save-baseline", type=Path)
    return parser


def _index_name(args: argparse.Namespace) -> str:
    if args.index_name:
        return args.index_name
    if args.provider_manifest and args.provider_manifest.is_file():
        payload = json.loads(args.provider_manifest.read_text(encoding="utf-8"))
        index_name = payload.get("index_name")
        if isinstance(index_name, str) and index_name.strip():
            return index_name.strip()
    raise ValueError(
        "--index-name is required unless --provider-manifest contains index_name"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        queries = load_sanity_queries(args.queries)
        index_name = _index_name(args)
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise ProviderUnavailableError(
                "Elasticsearch client is missing; install the project's search extra"
            ) from exc

        adapter = BM25Adapter(
            Elasticsearch(args.elasticsearch_url),
            index_name=index_name,
            manifest_path=args.provider_manifest,
        )
        adapter.ensure_available()
        manifest = build_run_manifest(
            repository_root=REPOSITORY_ROOT,
            query_suite_path=args.queries,
            adapter=adapter,
            top_k=args.top_k,
            repetitions=args.repetitions,
        )
        run = run_sanity_suite(
            adapter=adapter,
            queries=queries,
            top_k=args.top_k,
            repetitions=args.repetitions,
            score_tolerance=args.score_tolerance,
            manifest=manifest,
        )
        snapshot = build_snapshot(run)
        regression_diff = (
            compare_snapshots(load_snapshot(args.baseline), snapshot)
            if args.baseline
            else None
        )
        run_directory = write_run_reports(
            run,
            output_root=args.output,
            regression_diff=regression_diff,
        )
        if args.save_baseline:
            save_snapshot(args.save_baseline, snapshot)
        print(
            json.dumps(
                {
                    "run_directory": str(run_directory.resolve()),
                    "summary": run.summary,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if run.summary["failed_attempts"] == 0 else 1
    except ProviderUnavailableError as exc:
        print(f"provider unavailable: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
