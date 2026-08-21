"""CLI for Task05A v1 feature filtering and document preparation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.indexing.filtering import AsrFilterConfig, OcrFilterConfig
from mvr.indexing.task05a import Task05AConfig, build_index_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter immutable Task04 feature artifacts and write one "
            "Task05A index-ready JSON document per canonical keyframe."
        )
    )
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path)
    parser.add_argument("--embeddings-root", type=Path)
    parser.add_argument(
        "--required-embedding-providers",
        default="",
        help="Comma-separated provider IDs that must be present and valid",
    )
    parser.add_argument("--objects-root", type=Path)
    parser.add_argument("--ocr-root", type=Path)
    parser.add_argument("--asr-root", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--ocr-absolute-min-conf", type=float, default=0.50)
    parser.add_argument("--ocr-high-conf", type=float, default=0.85)
    parser.add_argument("--ocr-min-temporal-support", type=int, default=2)
    parser.add_argument("--asr-min-conf", type=float)
    parser.add_argument("--clip-expected-dimension", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_index_documents(
        features_root=args.features_root,
        mapping_csv=args.mapping_csv,
        output_root=args.output_root,
        clip_root=args.clip_root,
        embeddings_root=args.embeddings_root,
        objects_root=args.objects_root,
        ocr_root=args.ocr_root,
        asr_root=args.asr_root,
        dataset=args.dataset,
        config=Task05AConfig(
            ocr=OcrFilterConfig(
                absolute_min_conf=args.ocr_absolute_min_conf,
                high_conf=args.ocr_high_conf,
                min_temporal_support=args.ocr_min_temporal_support,
            ),
            asr=AsrFilterConfig(min_conf=args.asr_min_conf),
            clip_expected_dimension=args.clip_expected_dimension,
            required_embedding_providers=tuple(
                provider.strip()
                for provider in args.required_embedding_providers.split(",")
                if provider.strip()
            ),
        ),
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "document_count": result.document_count,
                "manifest": str(result.manifest_path),
                "filtering_report": str(result.filtering_report_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
