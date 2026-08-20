"""Verify the frozen five-keyframe Task05A/Task05B regression baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.evaluation.fasttest_baseline import (
    FasttestBaselineError,
    load_fasttest_config,
    verify_fasttest_baseline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the deterministic five-keyframe fasttest baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fasttest_baseline.json"),
    )
    parser.add_argument(
        "--require-lexical",
        action="store_true",
        help="Fail when the optional Task05B backend manifest is absent.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_fasttest_config(args.config)
        report = verify_fasttest_baseline(
            REPOSITORY_ROOT,
            config,
            require_lexical=args.require_lexical,
        )
    except FasttestBaselineError as exc:
        report = {
            "schema_version": "1.0",
            "status": "fail",
            "errors": [str(exc)],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
