"""Activate one validated immutable enrichment version as the active pointer.

The enrichment runner publishes into ``data/versions/enrichment/<version>/``
and never mutates ``data/corpus``.  This command validates that version's
markers and then swaps a single pointer in ``dataset_manifest.json``.  No
artifact is copied, so activating a new version and rolling back to an older
one cost one manifest PUT each.

Rollback is the same command with the previous version id.
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

from mvr.operations.enrichment_version import (
    EnrichmentVersionError,
    activate_enrichment_version,
    validate_enrichment_version,
)
from mvr.operations.r2_client import R2Client, R2OperationError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an enrichment version and point dataset_manifest.json at it"
        )
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Enrichment version id, i.e. the runner's --run-id",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("artifacts/operations/enrichment_activation"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the version and report, without touching dataset_manifest.json",
    )
    parser.add_argument(
        "--activated-at",
        default=None,
        help="Override the activation timestamp (defaults to now, UTC)",
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

    try:
        if args.validate_only:
            result = validate_enrichment_version(
                client=R2Client(),
                version=args.version,
                work_root=args.work_root,
            )
            result = {"status": "VALIDATED", **{
                key: value for key, value in result.items() if key != "ready"
            }}
        else:
            result = activate_enrichment_version(
                client=R2Client(),
                version=args.version,
                work_root=args.work_root,
                activated_at=args.activated_at,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (EnrichmentVersionError, R2OperationError, KeyError) as exc:
        print(
            json.dumps({"status": "FAILED", "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
