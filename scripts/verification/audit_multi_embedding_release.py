"""Run the Task 16 architecture and repository-hygiene gate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.evaluation.multi_embedding_release import (
    audit_multi_embedding_release,
    serialize_release_report,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit multi-embedding release gates")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_multi_embedding_release(REPOSITORY_ROOT)
    text = serialize_release_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["code_structural_gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
