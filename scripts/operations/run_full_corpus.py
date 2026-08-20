"""Dry-run or explicitly execute the audited full-corpus stage plan."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.operations.full_corpus import (
    dry_run_plan,
    load_full_corpus_config,
    run_full_corpus,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the audited multi-embedding full build")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approved-expensive-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_full_corpus_config(args.config, REPOSITORY_ROOT)
    if args.dry_run:
        print(json.dumps(dry_run_plan(config), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if not args.approved_expensive_run:
        parser.error("non-dry-run requires --approved-expensive-run after owner approval")

    def execute(stage, command):
        if any(part.startswith("<") and part.endswith(">") for part in command):
            return {"status": "fail", "reason": "command contains unresolved placeholder"}
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
        return {"status": "pass" if completed.returncode == 0 else "fail", "returncode": completed.returncode}

    state = run_full_corpus(config, approved=True, execute_stage=execute)
    print(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
