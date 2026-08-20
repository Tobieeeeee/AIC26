"""CLI entrypoint for the read-only Data-01 corpus readiness audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.audit.data_readiness import (  # noqa: E402
    AuditConfigurationError,
    AuditConfig,
    BLOCKED,
    run_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a read-only corpus and artifact readiness audit."
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("reports/data_readiness"))
    parser.add_argument("--mapping-path", type=Path)
    parser.add_argument("--videos-root", type=Path)
    parser.add_argument("--keyframes-root", type=Path)
    parser.add_argument("--media-info-root", type=Path)
    parser.add_argument("--ocr-root", type=Path)
    parser.add_argument("--asr-root", type=Path)
    parser.add_argument("--objects-root", type=Path)
    parser.add_argument("--embeddings-root", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--indexes-root", type=Path)
    parser.add_argument(
        "--required-provider",
        action="append",
        default=[],
        help="Provider name that must be present; repeat for multiple providers.",
    )
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_audit(
            AuditConfig(
                dataset_root=args.dataset_root,
                artifacts_root=args.artifacts_root,
                output_root=args.output_root,
                mapping_path=args.mapping_path,
                videos_root=args.videos_root,
                keyframes_root=args.keyframes_root,
                media_info_root=args.media_info_root,
                ocr_root=args.ocr_root,
                asr_root=args.asr_root,
                objects_root=args.objects_root,
                embeddings_root=args.embeddings_root,
                processed_root=args.processed_root,
                indexes_root=args.indexes_root,
                required_providers=tuple(args.required_provider),
                run_id=args.run_id,
            )
        )
    except AuditConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"audit failed: {exc}", file=sys.stderr)
        return 4
    print(
        json.dumps(
            {
                "status": result.summary["overall_status"],
                "output_dir": str(result.output_dir),
                "report": str(result.report_path),
                "issues": result.summary["issue_counts"],
                "read_only": True,
            },
            sort_keys=True,
        )
    )
    return 3 if result.summary["overall_status"] == BLOCKED else 0


if __name__ == "__main__":
    raise SystemExit(main())
