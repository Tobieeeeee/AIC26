"""DATA-03 `mvr worker` operator entrypoint.

Subcommands:
- `run`   — drive a single sequential or prefetch run over a working set of videos.
- `compare` — run both modes back-to-back and print the metric deltas (the
              measured evidence the implementation promises).
- `status` — dump the current BoundedCache + working-set state without running.

The CLI does not import any model. It builds the working set of object keys
(videos + their declared keyframe image keys) and hands them to the orchestrator,
which does not know what those keys represent.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mvr.operations import cache as cache_mod
from mvr.operations import r2_layout
from mvr.operations.worker_orchestrate import (
    DEFAULT_PREFETCH_AHEAD,
    RunResult,
    Stage,
    StageResult,
    WorkItem,
    WorkerMetrics,
    WorkingSet,
    prefetch_run,
    sequential_run,
)
from mvr.operations.stage_local import R2StageFetcher

LOGGER = logging.getLogger(__name__)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _cache_max_bytes(value: str) -> int:
    """Accept sizes like '4G', '512M', or plain bytes."""
    raw = value.strip()
    if not raw:
        raise argparse.ArgumentTypeError("empty value")
    upper = raw[-1].upper()
    if upper in ("G", "M", "K") and raw[:-1].isdigit():
        n = int(raw[:-1])
        return n * {"K": 1024, "M": 1024 * 1024, "G": 1024 * 1024 * 1024}[upper]
    if raw.isdigit():
        return int(raw)
    raise argparse.ArgumentTypeError(f"cannot parse size: {value!r}")


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run", help="Run stages over a working set with N+1 prefetch or sequentially"
    )
    p.add_argument("--videos", required=True,
                   help="Comma-separated video IDs (with --batch/--collection)")
    p.add_argument("--batch", required=True, help="Batch id, e.g. 1")
    p.add_argument("--collection", required=True, help="Collection id, e.g. L21")
    p.add_argument(
        "--keyframe-keys-file", type=Path, default=None,
        help="JSON file with per-video keyframe key lists; "
             "default is a single video-level key (worker uses keyframe-level reuse only when this is provided)",
    )
    p.add_argument(
        "--video-key-template", default=None,
        help="Template for the video key, with {vid} placeholder. "
             "e.g. 'Videos_L21_a/video/{vid}.mp4'. "
             "If omitted, uses the canonical data/raw layout.",
    )
    p.add_argument(
        "--run-mode", choices=["prefetch", "sequential"], default="prefetch"
    )
    p.add_argument("--prefetch", type=_positive_int, default=DEFAULT_PREFETCH_AHEAD,
                   help="Number of items to prefetch ahead (default 1)")
    p.add_argument(
        "--local-root", default=".", help="Local root for materialized files"
    )
    p.add_argument(
        "--cache-dir", default=cache_mod.DEFAULT_CACHE_DIR,
        help="BoundedCache directory (default .cache/mvr)",
    )
    p.add_argument(
        "--cache-max-gb", type=float, default=cache_mod.DEFAULT_MAX_SIZE_GB,
        help="BoundedCache budget in GB (default 50)",
    )
    p.add_argument(
        "--working-set-mb", type=_cache_max_bytes, default=0,
        help="Optional working-set byte budget (e.g. 512M, 2G). 0 = unbounded.",
    )
    p.add_argument(
        "--resume-completed-file", type=Path, default=None,
        help="File of item_ids already completed in a prior run",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write the run JSON",
    )
    p.add_argument(
        "--per-stage-sleep", type=float, default=0.05,
        help="Seconds the dry-run stages spend simulating work (default 0.05). "
             "Increase to make overlap visible; set 0 for transfer-only timing.",
    )
    p.set_defaults(command="run")


def _add_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "compare",
        help="Run sequential then prefetch on the same working set, print metric deltas",
    )
    p.add_argument("--videos", required=True)
    p.add_argument("--batch", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument(
        "--keyframe-keys-file", type=Path, default=None,
    )
    p.add_argument(
        "--video-key-template", default=None,
        help="Template for the video key, with {vid} placeholder. "
             "e.g. 'Videos_L21_a/video/{vid}.mp4'. "
             "If omitted, uses the canonical data/raw layout.",
    )
    p.add_argument("--prefetch", type=_positive_int, default=DEFAULT_PREFETCH_AHEAD)
    p.add_argument("--local-root", default=".")
    p.add_argument("--cache-dir", default=cache_mod.DEFAULT_CACHE_DIR)
    p.add_argument(
        "--cache-max-gb", type=float, default=cache_mod.DEFAULT_MAX_SIZE_GB,
    )
    p.add_argument(
        "--working-set-mb", type=_cache_max_bytes, default=0,
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write the comparison JSON",
    )
    p.set_defaults(command="compare")


def _add_status_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "status", help="Print current BoundedCache + working-set state"
    )
    p.add_argument("--cache-dir", default=cache_mod.DEFAULT_CACHE_DIR)
    p.add_argument("--cache-max-gb", type=float, default=cache_mod.DEFAULT_MAX_SIZE_GB)
    p.set_defaults(command="status")


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog or "mvr worker")
    sub = parser.add_subparsers(dest="worker_command", required=True)
    _add_run_parser(sub)
    _add_compare_parser(sub)
    _add_status_parser(sub)
    return parser


# ---------------------------------------------------------------------------
# Working set construction
# ---------------------------------------------------------------------------


@dataclass
class StageFactory:
    """Stages that exercise the orchestrator without doing real model work.

    The default mode is `dry-run`: each stage touches its keys (proving reuse
    and refcounting) and records bytes read; it does not run any model. The
    user can pass `--real-stages` to opt into real ingestion/OCR/ASR/etc.
    """

    name: str
    keys_kind: str  # "video" or "keyframe"
    sleep_seconds: float = 0.0

    def __post_init__(self) -> None:
        # Build the Stage instance methods dynamically.
        pass

    def make(self) -> Stage:
        stage = self

        class _DryRunStage:
            _name = stage.name
            _keys_kind = stage.keys_kind
            _sleep = stage.sleep_seconds

            @property
            def name(self) -> str:  # type: ignore[override]
                return self._name

            def keys_for(self, item: WorkItem) -> list[str]:  # type: ignore[override]
                if self._keys_kind == "video":
                    return [item.metadata["video_key"]]
                return list(item.metadata.get("keyframe_keys", []))

            async def run(
                self, item: WorkItem, paths: dict[str, Path]
            ) -> StageResult:  # type: ignore[override]
                started = time.perf_counter()
                for key, p in paths.items():
                    if not p.is_file():
                        return StageResult(
                            stage=self._name, item_id=item.item_id,
                            success=False, error=f"{key} not materialized",
                        )
                if self._sleep > 0:
                    await asyncio.sleep(self._sleep)
                return StageResult(
                    stage=self._name, item_id=item.item_id, success=True,
                    seconds=time.perf_counter() - started,
                )

        return _DryRunStage()


def _build_items(
    video_ids: Sequence[str],
    batch: str,
    collection: str,
    keyframe_keys_file: Path | None,
    video_key_template: str | None,
) -> list[WorkItem]:
    kf_map: dict[str, list[str]] = {}
    if keyframe_keys_file is not None and keyframe_keys_file.is_file():
        payload = json.loads(keyframe_keys_file.read_text(encoding="utf-8"))
        kf_map = {str(k): list(v) for k, v in payload.items()}
    items: list[WorkItem] = []
    for vid in video_ids:
        if video_key_template:
            video_key = video_key_template.format(vid=vid)
        else:
            video_key = r2_layout.raw_video_key(batch, collection, vid)
        kf_keys = kf_map.get(vid, [])
        keys = [video_key, *kf_keys]
        items.append(
            WorkItem(
                item_id=vid,
                keys=keys,
                estimated_bytes=0,
                metadata={
                    "video_key": video_key,
                    "keyframe_keys": kf_keys,
                    "video_id": vid,
                },
            )
        )
    return items


def _build_fetcher(local_root: Path, cache_dir: str, cache_max_gb: float):
    """Construct the real R2 fetcher.

    Defer boto3 import so unit tests + dry-runs don't require credentials.
    """
    from mvr.operations.r2_client import R2Client

    cache = cache_mod.BoundedCache(cache_dir, max_size_gb=cache_max_gb)
    client = R2Client()
    fetcher = R2StageFetcher(client=client, cache=cache)
    return fetcher, cache


def _build_dry_run_fetcher(items: list[WorkItem]):
    """A fake fetcher for `--dry-run` and unit tests.

    Each key is materialized into a real local file in a tmp dir (so the
    orchestrator can exercise eviction and reuse honestly), and the
    FetchOutcome reports a network GET/HEAD exactly once per unique key.
    """
    import tempfile

    class _DryRunFetcher:
        def __init__(self) -> None:
            self.root = Path(tempfile.mkdtemp(prefix="mvr_dryrun_"))
            self.downloaded: set[str] = set()
            self.stats: dict[str, int] = {
                "get_requests": 0, "head_requests": 0, "bytes": 0
            }

        async def fetch(self, object_key: str):
            import asyncio as _aio
            return await _aio.to_thread(self._fetch_sync, object_key)

        def _fetch_sync(self, object_key: str):
            local = self.root / object_key.replace("/", "_")
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.is_file() and local.stat().st_size > 0:
                return _make_outcome(object_key, local, cache_hit=True)
            if object_key in self.downloaded:
                return _make_outcome(object_key, local, cache_hit=True)
            # Simulate GET cost proportional to declared byte count
            local.write_bytes(b"x" * _declared_bytes(object_key))
            self.downloaded.add(object_key)
            self.stats["get_requests"] += 1
            self.stats["bytes"] += local.stat().st_size
            return _make_outcome(object_key, local, cache_hit=False)

    return _DryRunFetcher()


def _declared_bytes(object_key: str) -> int:
    """Deterministic per-key byte count for dry runs (visible transfer wait)."""
    return 1024 * 1024 * (4 + (len(object_key) % 8))  # 4–11 MiB


def _make_outcome(key: str, local: Path, *, cache_hit: bool):
    from mvr.operations.worker_orchestrate import FetchOutcome
    return FetchOutcome(
        object_key=key,
        local_path=local,
        size_bytes=local.stat().st_size,
        cache_hit=cache_hit,
        get_requests=0 if cache_hit else 1,
        head_requests=0,
    )


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------


async def _do_run(args: argparse.Namespace) -> RunResult:
    video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    if not video_ids:
        raise SystemExit("--videos must list at least one video id")
    items = _build_items(
        video_ids, args.batch, args.collection,
        args.keyframe_keys_file, args.video_key_template,
    )
    working_set = WorkingSet(
        max_bytes=(args.working_set_mb or None) or None
    )
    fetcher, cache = _build_fetcher(
        Path(args.local_root), args.cache_dir, args.cache_max_gb,
    )
    # Read per-stage sleep so real canary can run with realistic per-stage cost
    # without invoking the model code (the orchestrator proves transfer overlap
    # either way; the dry-run stages in the canary add a tunable CPU cost).
    per_stage_sleep = float(getattr(args, "per_stage_sleep", 0.05) or 0.0)
    stages = [
        StageFactory("ingest", "video", sleep_seconds=per_stage_sleep).make(),
        StageFactory("ocr", "video", sleep_seconds=per_stage_sleep).make(),
        StageFactory("asr", "video", sleep_seconds=per_stage_sleep).make(),
        StageFactory("objects", "keyframe", sleep_seconds=per_stage_sleep).make(),
    ]
    resume = _read_resume(args.resume_completed_file)
    if args.run_mode == "prefetch":
        result = await prefetch_run(
            items, stages, fetcher,
            prefetch_ahead=args.prefetch,
            working_set=working_set,
            resume_completed=resume,
            on_delete=fetcher.delete_local,
        )
    else:
        result = await sequential_run(
            items, stages, fetcher,
            working_set=working_set,
            resume_completed=resume,
            on_delete=fetcher.delete_local,
        )
    result.metrics.cache_hits = cache.stats.hits
    result.metrics.cache_misses = cache.stats.misses
    result.metrics.evictions += cache.stats.evictions
    return result


def _read_resume(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("completed", []))


async def _do_compare(args: argparse.Namespace) -> dict[str, Any]:
    video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    items = _build_items(
        video_ids, args.batch, args.collection,
        args.keyframe_keys_file, getattr(args, "video_key_template", None),
    )
    ws_budget = (args.working_set_mb or None) or None

    def _make_stages() -> list[Stage]:
        return [
            StageFactory("ingest", "video", sleep_seconds=0.05).make(),
            StageFactory("ocr", "video", sleep_seconds=0.05).make(),
            StageFactory("asr", "video", sleep_seconds=0.05).make(),
            StageFactory("objects", "keyframe", sleep_seconds=0.05).make(),
        ]

    # Each arm gets its own fetcher so the byte/GET accounting is independent.
    # A shared fetcher would let the second arm reuse the first arm's downloads
    # and report zero transfer — an unfair comparison.
    seq = await sequential_run(
        items, _make_stages(), _build_dry_run_fetcher(items),
        working_set=WorkingSet(max_bytes=ws_budget),
    )
    pre = await prefetch_run(
        items, _make_stages(), _build_dry_run_fetcher(items),
        prefetch_ahead=args.prefetch,
        working_set=WorkingSet(max_bytes=ws_budget),
    )
    delta = {
        "wall_seconds_delta": round(seq.metrics.wall_seconds - pre.metrics.wall_seconds, 4),
        "visible_transfer_wait_delta": round(
            seq.metrics.visible_transfer_wait_seconds
            - pre.metrics.visible_transfer_wait_seconds, 4
        ),
        "overlap_ratio_sequential": round(seq.metrics.overlap_ratio, 4),
        "overlap_ratio_prefetch": round(pre.metrics.overlap_ratio, 4),
        "bytes_delta": seq.metrics.bytes_transferred - pre.metrics.bytes_transferred,
        "get_requests_sequential": seq.metrics.get_requests,
        "get_requests_prefetch": pre.metrics.get_requests,
    }
    out = {
        "sequential": _result_to_dict(seq),
        "prefetch": _result_to_dict(pre),
        "delta": delta,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _result_to_dict(result: RunResult) -> dict[str, Any]:
    return {
        "metrics": result.metrics.to_dict(),
        "stages": [
            {
                "stage": r.stage, "item_id": r.item_id, "success": r.success,
                "seconds": round(r.seconds, 4), "error": r.error,
            }
            for r in result.stage_results
        ],
    }


async def _do_status(args: argparse.Namespace) -> dict[str, Any]:
    cache = cache_mod.BoundedCache(args.cache_dir, max_size_gb=args.cache_max_gb)
    return {
        "cache_dir": str(cache.cache_dir),
        "cache_stats": cache.stats.to_dict(),
        "cache_size_bytes": cache._cache_size_bytes(),
        "cache_max_bytes": cache.max_size_bytes,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.worker_command == "run":
            result = asyncio.run(_do_run(args))
            out = {
                "run_mode": result.metrics.run_mode,
                "metrics": result.metrics.to_dict(),
                "stages": [
                    {
                        "stage": r.stage, "item_id": r.item_id, "success": r.success,
                        "seconds": round(r.seconds, 4), "error": r.error,
                    }
                    for r in result.stage_results
                ],
            }
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.worker_command == "compare":
            out = asyncio.run(_do_compare(args))
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.worker_command == "status":
            out = asyncio.run(_do_status(args))
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        parser.error(f"unknown command: {args.worker_command}")
        return 2
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        err = {
            "error": type(exc).__name__,
            "message": str(exc),
            "hint": (
                "Real-R2 fetches need R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, "
                "R2_BUCKET, R2_ENDPOINT in the environment, plus the boto3 package. "
                "The compare/status subcommands can run dry-run with no credentials."
            ),
        }
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
