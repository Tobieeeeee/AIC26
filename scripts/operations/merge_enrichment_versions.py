"""Combine per-machine enrichment versions into one publishable version.

Splitting a batch across machines is the cheap way to shorten a run: the
collections are disjoint, so nothing is downloaded or computed twice. What the
split cannot share is ``_READY.json``. It lives at the version root and carries
totals for the whole run, so machines writing one version id would each leave a
marker describing only their own share, and the validator would reject the
result.

Each machine therefore publishes its own version, and this command republishes
their collections under one target with a marker that covers all of them:

    python scripts/operations/merge_enrichment_versions.py \\
      --source enrich-b1-m1 --source enrich-b1-m2 --source enrich-b1-m3 \\
      --target enrich-batch1-v1

Nothing here counts machines. Three sources and thirty take the same path, and
copies are server-side, so no artifact travels through this process.
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
    merge_enrichment_versions,
)
from mvr.operations.r2_client import R2Client, R2OperationError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-machine enrichment versions into one version"
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="VERSION[,VERSION...]",
        help=(
            "A machine's enrichment version id, i.e. its --run-id. Repeat the "
            "flag or separate ids with commas; both may be mixed."
        ),
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Version id to publish the merged collections under",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("artifacts/operations/enrichment_merge"),
    )
    parser.add_argument(
        "--merged-at",
        default=None,
        help="Override the merge timestamp (defaults to now, UTC)",
    )
    args = parser.parse_args(argv)

    sources = [
        value.strip()
        for group in args.source
        for value in group.split(",")
        if value.strip()
    ]
    if not sources:
        parser.error("--source needs at least one version id")
    # A version listed twice means the operator lost track of who ran what.
    # Collapsing it quietly would hide that while the totals stayed plausible.
    duplicates = sorted({name for name in sources if sources.count(name) > 1})
    if duplicates:
        parser.error(f"--source repeats {', '.join(duplicates)}")
    args.source = sources
    return args


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
        report = merge_enrichment_versions(
            client=R2Client(),
            sources=args.source,
            target=args.target,
            work_root=args.work_root,
            merged_at=args.merged_at,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (EnrichmentVersionError, R2OperationError, KeyError) as exc:
        print(
            json.dumps({"status": "FAILED", "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
