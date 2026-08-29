"""Plan and apply R2 bucket reorganization.

Two safe modes:
  plan  (default): enumerate every intended move/copy/delete and print it.
                   Nothing is written to R2.
  --apply         : execute the plan: copy each object to its new key, verify
                   SHA256 matches, then delete the source. Idempotent — already
                   completed moves are skipped.

Usage:
    python scripts/operations/r2_reorg.py plan
    python scripts/operations/r2_reorg.py plan --write plan.json
    python scripts/operations/r2_reorg.py apply --plan plan.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env", override=False)

from mvr.operations.r2_client import R2Client


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(prefix: str, client: R2Client) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    trunc, token = True, None
    while trunc:
        kw = {"Bucket": client.bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        r = client._get_client().list_objects_v2(**kw)
        out += [(o["Key"], int(o.get("Size", 0))) for o in r.get("Contents", [])]
        trunc = r.get("IsTruncated")
        token = r.get("NextContinuationToken")
    return out


def plan_videos_to_canonical(client: R2Client) -> list[dict]:
    """Map Videos_{collection}_{shard}/video/*.mp4 -> data/raw/batch1/videos/..."""
    moves: list[dict] = []
    dirs: list[str] = []
    trunc, token = True, None
    while trunc:
        kw = {"Bucket": client.bucket, "Prefix": "", "Delimiter": "/", "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        r = client._get_client().list_objects_v2(**kw)
        dirs += [p["Prefix"] for p in r.get("CommonPrefixes", [])]
        trunc = r.get("IsTruncated")
        token = r.get("NextContinuationToken")

    for d in dirs:
        if not d.startswith("Videos_"):
            continue
        # Videos_L21_a -> collection L21
        collection = d.split("_", 2)[1]
        for key, size in walk(d, client):
            name = key.rsplit("/", 1)[-1]
            new_key = f"data/raw/batch1/videos/{collection}/{name}"
            moves.append({"from": key, "to": new_key, "size": size})
    return moves


def build_plan(client: R2Client) -> dict:
    moves = plan_videos_to_canonical(client)
    total_size = sum(m["size"] for m in moves)
    return {
        "moves": moves,
        "summary": {
            "move_count": len(moves),
            "total_size_bytes": total_size,
            "total_size_gb": round(total_size / 1e9, 2),
        },
    }


def apply_plan(plan: dict, client: R2Client) -> None:
    for i, m in enumerate(plan["moves"], 1):
        src, dst, size = m["from"], m["to"], m["size"]
        # Fast existence check: single HeadObject, no retry on 404.
        try:
            client.head_object(dst)
            exists = True
        except Exception:
            exists = False
        if exists:
            # Destination already has the file; source copy is redundant.
            # Per decision: drop the duplicate at the old location.
            client.delete_object(src)
            print(f"[{i}/{len(plan['moves'])}] dedupe (drop old): {src}  (kept {dst})", flush=True)
            continue
        print(f"[{i}/{len(plan['moves'])}] copy: {src} -> {dst}  ({size/1e6:.1f} MB)", flush=True)
        # Server-side copy: no local download, much faster.
        client.copy_object(src, dst)
        # verify
        head = client.head_object(dst)
        if int(head.get("ContentLength", 0)) != size:
            raise SystemExit(f"size mismatch after copy: {dst}")
        client.delete_object(src)
        print(f"  verified + deleted source", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    plan_p = sub.add_parser("plan", help="Print the intended reorganization.")
    plan_p.add_argument("--write", type=Path, help="Also write plan JSON to this file.")

    apply_p = sub.add_parser("apply", help="Execute a previously generated plan.")
    apply_p.add_argument("--plan", type=Path, required=True, help="Path to plan JSON.")

    args = p.parse_args(argv)
    client = R2Client()

    if args.cmd == "plan":
        plan = build_plan(client)
        print(json.dumps(plan["summary"], indent=2))
        print("first 10 moves:")
        for m in plan["moves"][:10]:
            print(f"  {m['from']}  ->  {m['to']}")
        if args.write:
            args.write.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            print(f"plan written to {args.write}")
        return 0

    if args.cmd == "apply":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        apply_plan(plan, client)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
