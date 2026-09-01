"""Activate one validated R2 ingestion candidate as the canonical corpus.

This command is intentionally explicit: a candidate that supersedes existing
downstream artifacts requires one ``--acknowledge-stale`` flag per affected
collection.  The operation snapshots the old canonical state before copying or
deleting any canonical keyframe.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.operations.ingestion_activation import activate_promotion
from mvr.operations.ingestion_promotion import IngestionPromotionError
from mvr.operations.r2_client import R2Client, R2OperationError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backup current R2 corpus and activate a validated ingestion candidate"
    )
    parser.add_argument("--promotion-id", required=True)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
        help="Local candidate bundle directory containing promotion_report.json",
    )
    parser.add_argument(
        "--acknowledge-stale",
        action="append",
        default=[],
        metavar="COLLECTION",
        help="Explicitly accept superseding downstream artifacts; repeat as needed",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("artifacts/operations/r2_ingestion_activation"),
    )
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        from dotenv import load_dotenv

        load_dotenv(REPOSITORY_ROOT / ".env", override=False)
    except ImportError:
        pass

    try:
        result = activate_promotion(
            R2Client(max_pool_connections=args.workers),
            bundle_root=args.bundle_root,
            promotion_id=args.promotion_id,
            acknowledged_stale_collections=args.acknowledge_stale,
            work_root=args.work_root,
            workers=args.workers,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (IngestionPromotionError, R2OperationError) as exc:
        print(
            json.dumps({"status": "FAILED", "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
