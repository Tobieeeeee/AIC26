"""Build isolated semantic indexes from validated Task05A documents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.indexing.provider_indexes import FaissProviderBackend, build_provider_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one FAISS index per embedding provider")
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provider-specs", type=Path, default=Path("configs/embedding_providers.json"))
    parser.add_argument("--providers", required=True, help="Comma-separated provider IDs")
    parser.add_argument("--environment", default="local")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--policy", choices=("create", "rebuild", "append"), default="create")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    providers = sorted({item.strip() for item in args.providers.split(",") if item.strip()})
    if not providers:
        raise ValueError("at least one provider is required")
    backend = FaissProviderBackend(args.output_root / "collections")
    manifests = []
    for provider_id in providers:
        result = build_provider_index(
            provider_root=args.embeddings_root / provider_id,
            processed_root=args.processed_root,
            output_root=args.output_root,
            provider_specs_path=args.provider_specs,
            backend=backend,
            environment=args.environment,
            dataset=args.dataset,
            policy=args.policy,
            batch_size=args.batch_size,
        )
        manifests.append(str(result.manifest_path))
    print(json.dumps({"provider_manifests": manifests}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
