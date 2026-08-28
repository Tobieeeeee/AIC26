"""Standalone R2 push script.

Usage:
    python scripts/operations/r2_push.py --staging artifacts/ocr --feature ocr --run-id run-001
    python scripts/operations/r2_push.py --parquet out/ocr.parquet --feature ocr --batch 1 --collection L21
    python scripts/operations/r2_push.py --embeddings out/beit3.npy --provider beit3
    python scripts/operations/r2_push.py --index out/beit3.faiss --name indexes,beit3
    python scripts/operations/r2_push.py --report out/summary.json --name reports,summary
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvr.operations.cache import BoundedCache, DEFAULT_CACHE_DIR, DEFAULT_MAX_SIZE_GB
from mvr.operations.r2_client import R2Client
from mvr.operations.r2_sync import R2Sync


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Push local artifacts to Cloudflare R2")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--staging", help="Directory of files to push under data/staging/<feature>/<run_id>")
    group.add_argument("--parquet", help="Canonical metadata Parquet file")
    group.add_argument("--embeddings", help="Provider embedding file (e.g. vectors.npy)")
    group.add_argument("--index", help="Index artifact file")
    group.add_argument("--report", help="Report JSON file")
    p.add_argument("--feature", help="Staging feature name (ocr/asr/objects/embeddings)")
    p.add_argument("--run-id", help="Staging run id")
    p.add_argument("--provider", help="Provider id for --embeddings (beit3/metaclip2/perception_encoder)")
    p.add_argument("--batch", help="Batch id for parquet metadata")
    p.add_argument("--collection", help="Collection id for parquet metadata")
    p.add_argument("--name", help="Comma-separated path parts for --index/--report")
    p.add_argument("--local-root", default=".")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--cache-max-gb", type=float, default=DEFAULT_MAX_SIZE_GB)
    args = p.parse_args(argv)

    client = R2Client()
    cache = BoundedCache(args.cache_dir, max_size_gb=args.cache_max_gb)
    sync = R2Sync(client=client, local_root=args.local_root, cache=cache)

    if args.staging:
        if not args.feature or not args.run_id:
            p.error("--staging requires --feature and --run-id")
        staging = Path(args.staging)
        files = sorted(staging.iterdir()) if staging.is_dir() else [staging]
        stats = sync.push_staging(args.feature, args.run_id, files)
        print(json.dumps({"mode": "staging", **stats.to_dict()}, indent=2))
        return

    if args.parquet:
        if not args.feature or not args.batch or not args.collection:
            p.error("--parquet requires --feature, --batch, --collection")
        stats = sync.push_corpus_metadata(args.feature, args.batch, args.collection, Path(args.parquet))
        print(json.dumps({"mode": "parquet", **stats.to_dict()}, indent=2))
        return

    if args.embeddings:
        if not args.provider:
            p.error("--embeddings requires --provider")
        stats = sync.push_corpus_embeddings(args.provider, Path(args.embeddings))
        print(json.dumps({"mode": "embeddings", **stats.to_dict()}, indent=2))
        return

    if args.index:
        parts = [x.strip() for x in args.name.split(",") if x.strip()] if args.name else []
        stats = sync.push_corpus_index(Path(args.index), *parts) if parts else sync.push_corpus_index(Path(args.index))
        print(json.dumps({"mode": "index", **stats.to_dict()}, indent=2))
        return

    if args.report:
        parts = [x.strip() for x in args.name.split(",") if x.strip()] if args.name else []
        stats = sync.push_corpus_report(Path(args.report), *parts) if parts else sync.push_corpus_report(Path(args.report))
        print(json.dumps({"mode": "report", **stats.to_dict()}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())