"""Cloud GPU runner: provision a rented GPU machine, then run the full pipeline.

Designed for a fresh Ubuntu GPU instance (e.g. RunPod, Vast, Lambda). It clones
the repo, builds a venv, installs deps, writes .env from a local env file, then
delegates to run_pipeline_local.py for the actual pull -> ingest -> embeddings
-> Parquet -> promote -> push autopilot.

Usage (from your local machine, on the rented GPU):
    python scripts/operations/run_pipeline_cloud.py \
        --env-file ~/.aic26.env \
        --repo https://github.com/Tobieeeeee/AIC26.git \
        --work-root /data/corpus_work \
        --device cuda

The --env-file is copied to <repo>/.env on the remote machine (never committed).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str | Path], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(list(map(str, cmd)), check=True, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python scripts/operations/run_pipeline_cloud.py",
        description="Provision a cloud GPU box and run the full offline pipeline.",
    )
    p.add_argument("--repo", default="https://github.com/Tobieeeeee/AIC26.git")
    p.add_argument("--branch", default="main")
    p.add_argument("--dest", type=Path, default=Path("/data/AIC26"))
    p.add_argument("--env-file", type=Path, required=True,
                   help="Local .env file to copy into the repo as .env (not committed).")
    p.add_argument("--work-root", type=Path, default=Path("/data/corpus_work"))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--collections", default=None)
    p.add_argument("--videos", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--providers", default="beit3,metaclip2,perception_encoder")
    p.add_argument(
        "--ingest-only",
        action="store_true",
        help="Run canonical ingestion only and stage each completed tree on R2",
    )
    p.add_argument("--no-push", action="store_true")
    p.add_argument(
        "--evict-artifacts-after-push",
        action="store_true",
        help="Bound disk use by evicting ingestion output after successful upload",
    )
    p.add_argument(
        "--allow-model-downloads",
        action="store_true",
        help="Explicitly allow pinned model downloads on the GPU worker",
    )
    p.add_argument("--skip-provision", action="store_true",
                   help="Skip clone/install (repo already provisioned).")
    return p.parse_args(argv)


def provision(args: argparse.Namespace) -> None:
    dest: Path = args.dest
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--branch", args.branch, args.repo, str(dest)])
    else:
        run(["git", "pull"], cwd=dest)

    # Install system python3.12-venv if missing (best-effort)
    run(["sudo", "apt-get", "update", "-y"])
    run(["sudo", "apt-get", "install", "-y", "python3.12", "python3.12-venv", "python3-pip"])

    venv = dest / ".venv"
    if not venv.exists():
        run(["python3.12", "-m", "venv", str(venv)])
    pip = venv / "bin" / "pip"
    run([pip, "install", "--upgrade", "pip", "setuptools", "wheel"])
    run(
        [
            pip,
            "install",
            "-e",
            ".[dev,api,semantic,search,ocr,objects,milvus,operations,metaclip2]",
        ],
        cwd=dest,
    )

    # Deploy env file
    env_target = dest / ".env"
    shutil.copy2(args.env_file, env_target)
    print(f"copied {args.env_file} -> {env_target}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.skip_provision:
        provision(args)

    repo = args.dest
    python = repo / ".venv" / "bin" / "python"
    runner = repo / "scripts" / "operations" / "run_pipeline_local.py"

    cmd = [
        str(python), str(runner),
        "--work-root", str(args.work_root),
        "--device", args.device,
        "--providers", args.providers,
    ]
    if args.collections:
        cmd += ["--collections", args.collections]
    if args.videos:
        cmd += ["--videos", args.videos]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.no_push:
        cmd += ["--no-push"]
    if args.ingest_only:
        cmd += ["--ingest-only"]
    if args.evict_artifacts_after_push:
        cmd += ["--evict-artifacts-after-push"]
    if args.allow_model_downloads:
        cmd += ["--allow-model-downloads"]

    return subprocess.run(cmd, cwd=repo).returncode


if __name__ == "__main__":
    sys.exit(main())
