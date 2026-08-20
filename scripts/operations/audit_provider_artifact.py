"""Audit a provider artifact against the authoritative canonical ID set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.operations.full_corpus import audit_provider_identity


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit one full-corpus embedding provider")
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--canonical-mapping", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_provider_identity(args.provider_root, args.canonical_mapping)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
