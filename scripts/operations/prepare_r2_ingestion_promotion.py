"""Prepare and optionally upload a validated multi-run ingestion bundle.

This command never mutates ``data/corpus``.  It consolidates the selected
run-scoped staging trees and publishes the candidate under
``data/staging/ingestion-consolidated/<promotion-id>/`` when ``--upload`` is
explicitly supplied.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.operations.ingestion_promotion import (
    IngestionPromotionError,
    parse_source_assignment,
    prepare_promotion_bundle,
    upload_prepared_bundle,
)
from mvr.operations.r2_client import R2Client


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and consolidate R2 ingestion staging without activating corpus"
    )
    parser.add_argument("--batch-id", default="1")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="RUN_ID:L21,L22",
        help="Authoritative run and collections; repeat for each worker run",
    )
    parser.add_argument(
        "--promotion-id",
        default=None,
        help="Stable candidate ID; defaults to batchN-<UTC timestamp>",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("artifacts/operations/r2_ingestion_promotion"),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the prepared candidate to R2 staging; corpus is still untouched",
    )
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

    promotion_id = args.promotion_id or (
        f"batch{args.batch_id}-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    try:
        sources = [parse_source_assignment(value) for value in args.source]
        client = R2Client()
        bundle = prepare_promotion_bundle(
            client,
            batch_id=args.batch_id,
            sources=sources,
            promotion_id=promotion_id,
            work_root=args.work_root,
            workers=args.workers,
        )
        result: dict[str, object] = {
            "promotion_id": bundle.promotion_id,
            "bundle_root": str(bundle.root),
            "summary": bundle.report["summary"],
            "canonical_impact": bundle.report["canonical_impact"],
            "safe_to_activate": bundle.report["safe_to_activate"],
            "uploaded": False,
        }
        if args.upload:
            result["upload"] = upload_prepared_bundle(client, bundle)
            result["uploaded"] = True
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except IngestionPromotionError as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
