"""Standalone R2 pull script.

Usage:
    python scripts/operations/r2_pull.py --corpus --local-root /data
    python scripts/operations/r2_pull.py --prefix data/corpus/keyframes
    python scripts/operations/r2_pull.py --videos L21_V001,L21_V029 --batch 1 --collection L21
"""
import argparse
import json
import logging
import os
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
    p = argparse.ArgumentParser(description="Pull data from Cloudflare R2 to local filesystem")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prefix", help="Pull every object under this R2 prefix")
    group.add_argument("--videos", help="Comma-separated video IDs")
    group.add_argument("--corpus", action="store_true", help="Pull entire canonical corpus tree")
    group.add_argument("--manifest", action="store_true", help="Pull only the dataset manifest JSON")
    p.add_argument("--batch", help="Batch id for --videos (e.g. 1)")
    p.add_argument("--collection", help="Collection id for --videos (e.g. L21)")
    p.add_argument("--local-root", default=".", help="Local root directory (default: .)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--cache-max-gb", type=float, default=DEFAULT_MAX_SIZE_GB)
    p.add_argument("--force", action="store_true", help="Re-download even if local file exists")
    p.add_argument("--dry-run", action="store_true", help="List remote objects without downloading")
    args = p.parse_args(argv)

    local_root = Path(args.local_root).resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    client = R2Client()
    cache = BoundedCache(args.cache_dir, max_size_gb=args.cache_max_gb)
    sync = R2Sync(client=client, local_root=local_root, cache=cache)

    if args.dry_run:
        prefix = args.prefix or sync.corpus_prefix
        objects = sync.list_remote(prefix)
        total = sum(int(o.get("Size", 0)) for o in objects)
        print(json.dumps({
            "dry_run": True,
            "prefix": prefix,
            "count": len(objects),
            "total_bytes": total,
            "total_gb": round(total / (1024**3), 2),
        }, indent=2))
        return

    if args.manifest:
        path = sync.pull_dataset_manifest(force=args.force)
        print(json.dumps({"mode": "manifest", "path": str(path) if path else None}, indent=2))
        return

    if args.corpus:
        stats = sync.pull_corpus(force=args.force)
        print(json.dumps({"mode": "corpus", **stats.to_dict()}, indent=2))
        return

    if args.prefix:
        stats = sync.pull_prefix(args.prefix, force=args.force)
        print(json.dumps({"mode": "prefix", "prefix": args.prefix, **stats.to_dict()}, indent=2))
        return

    if args.videos:
        if not args.batch or not args.collection:
            p.error("--videos requires --batch and --collection")
        ids = [v.strip() for v in args.videos.split(",") if v.strip()]
        stats = sync.pull_videos(args.batch, args.collection, ids, force=args.force)
        print(json.dumps({"mode": "videos", **stats.to_dict()}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())