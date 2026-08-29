"""Run the deterministic Data-02R R2 access canary.

The script deliberately keeps credentials in environment variables and never
serializes them.  Heavy inference is replaced by image/audio I/O-only work;
the production FFmpeg/Pillow access paths still run on a real media fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image

from mvr.enrichment.asr.audio import FfmpegAudioWindowProvider
from mvr.enrichment.ocr.scene_text import generate_sampling_frames
from mvr.ingestion.extract import extract_keyframes
from mvr.ingestion.models import Keyframe
from mvr.ingestion.probe import probe_video


REQUIRED_ENV = (
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ENDPOINT_URL",
)
STAGES = (
    "ingestion",
    "ocr",
    "asr",
    "beit3",
    "metaclip2",
    "perception_encoder",
    "objects",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_environment() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError("missing required environment variables: " + ", ".join(missing))
    for executable in ("aws", "ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable is unavailable: {executable}")


def run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=capture,
        text=True,
        env=os.environ.copy(),
    )


def aws(*arguments: str) -> None:
    run(
        [
            "aws",
            *arguments,
            "--endpoint-url",
            os.environ["AWS_ENDPOINT_URL"],
            "--no-cli-pager",
        ]
    )


def graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        "https://api.cloudflare.com/client/v4/graphql",
        headers={"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}"},
        json={"query": query, "variables": variables},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        messages = " | ".join(str(item.get("message")) for item in payload["errors"])
        raise RuntimeError(f"Cloudflare GraphQL query failed: {messages}")
    return payload["data"]


def query_operations(start: str, end: str, bucket: str) -> dict[str, int]:
    query = """
    query R2Ops($accountTag: string!, $startDate: Time, $endDate: Time, $bucketName: string) {
      viewer { accounts(filter: {accountTag: $accountTag}) {
        rows: r2OperationsAdaptiveGroups(
          limit: 10000
          filter: {datetime_geq: $startDate, datetime_leq: $endDate, bucketName: $bucketName}
        ) { sum { requests } dimensions { actionType } }
      } }
    }
    """
    data = graphql(
        query,
        {
            "accountTag": os.environ["CLOUDFLARE_ACCOUNT_ID"],
            "startDate": start,
            "endDate": end,
            "bucketName": bucket,
        },
    )
    accounts = data["viewer"]["accounts"]
    if not accounts:
        raise RuntimeError("Cloudflare GraphQL returned no matching account")
    result: dict[str, int] = {}
    for row in accounts[0]["rows"]:
        action = row["dimensions"]["actionType"]
        result[action] = result.get(action, 0) + int(row["sum"]["requests"])
    return dict(sorted(result.items()))


def query_storage(bucket: str) -> dict[str, Any] | None:
    end = utc_now()
    start = end - timedelta(days=2)
    query = """
    query R2Storage($accountTag: string!, $startDate: Time, $endDate: Time, $bucketName: string) {
      viewer { accounts(filter: {accountTag: $accountTag}) {
        rows: r2StorageAdaptiveGroups(
          limit: 1
          filter: {datetime_geq: $startDate, datetime_leq: $endDate, bucketName: $bucketName}
          orderBy: [datetime_DESC]
        ) {
          max { objectCount uploadCount payloadSize metadataSize }
          dimensions { datetime }
        }
      } }
    }
    """
    data = graphql(
        query,
        {
            "accountTag": os.environ["CLOUDFLARE_ACCOUNT_ID"],
            "startDate": iso(start),
            "endDate": iso(end),
            "bucketName": bucket,
        },
    )
    rows = data["viewer"]["accounts"][0]["rows"]
    return rows[0] if rows else None


def safe_recreate(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"refusing to recreate path outside canary root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def derive_fixture(source_video: Path, source_image: Path, fixture_root: Path) -> tuple[Path, Path]:
    fixture_root.mkdir(parents=True, exist_ok=True)
    video = fixture_root / "L21_V010_12s.mp4"
    image = fixture_root / "L21_V010_S0001_K01.jpg"
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source_video),
            "-t",
            "12",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            str(video),
        ]
    )
    shutil.copy2(source_image, image)
    return video, image


def run_stage(stage: str, video: Path, image: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if stage == "ingestion":
        info = probe_video(video)
        frame = Keyframe("canary-ingestion", "L21_V010", 1, 1, 125, 5.0, "frame.jpg")
        extract_keyframes(video, [frame], output / "ingestion")
        detail = {"frame_count": info.frame_count, "duration": info.duration}
    elif stage == "ocr":
        row = {
            "keyframe_id": "canary-ocr",
            "video_id": "L21_V010",
            "shot_id": 1,
            "frame_idx": 125,
            "timestamp": 5.0,
        }
        frames, _ = generate_sampling_frames([row], {1: (0, 299)})
        extract_keyframes(video, frames, output / "ocr")
        detail = {"sample_frames": len(frames), "inference": "mocked"}
    elif stage == "asr":
        provider = FfmpegAudioWindowProvider()
        has_audio = provider.has_audio_stream(video)
        audio = provider.decode(video, 2.0, 8.0, 16000) if has_audio else []
        detail = {"has_audio": has_audio, "audio_samples": len(audio), "inference": "mocked"}
    else:
        with Image.open(image) as opened:
            opened.convert("RGB").load()
        detail = {"image_decodes": 1, "inference": "mocked"}
    return {"wall_seconds": time.perf_counter() - started, **detail}


def network_window(label: str, action: Callable[[], None], windows: list[dict[str, Any]]) -> None:
    start = utc_now() - timedelta(seconds=0.5)
    started = time.perf_counter()
    action()
    elapsed = time.perf_counter() - started
    end = utc_now() + timedelta(seconds=0.5)
    windows.append({"label": label, "start": iso(start), "end": iso(end), "wall_seconds": elapsed})
    time.sleep(1.25)


def local_stage_window(
    strategy: str,
    stage: str,
    video: Path,
    image: Path,
    output: Path,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    start = utc_now()
    result = run_stage(stage, video, image, output)
    end = utc_now()
    windows.append(
        {
            "label": f"{strategy}_{stage}",
            "start": iso(start),
            "end": iso(end),
            "wall_seconds": result["wall_seconds"],
            "local_only": True,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="aic-test")
    parser.add_argument(
        "--source-video",
        type=Path,
        default=Path("artifacts/real_task04_test/videos/L21_V010.mp4"),
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        default=Path(
            "artifacts/real_task04_test/ingest/keyframes/L21_V010/"
            "L21_V010_S0001_K01.jpg"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/verification/r2_access_audit"),
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Refresh Cloudflare operation rows in an existing canary result.",
    )
    return parser.parse_args()


def refresh_existing_result(root: Path, bucket: str) -> None:
    result_path = root / "canary_results.json"
    cleanup_path = root / "cleanup_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for window in result["windows"]:
        window["cloudflare_operations"] = query_operations(
            window["start"], window["end"], bucket
        )
    result["storage_after_cleanup"] = query_storage(bucket)
    result["analytics_refreshed_at"] = iso(utc_now())
    add_client_summary(result)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if cleanup_path.is_file():
        cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
        cleanup["operations"] = query_operations(cleanup["start"], cleanup["end"], bucket)
        cleanup["analytics_refreshed_at"] = iso(utc_now())
        cleanup_path.write_text(
            json.dumps(cleanup, indent=2, sort_keys=True), encoding="utf-8"
        )


def add_client_summary(result: dict[str, Any]) -> None:
    video_bytes = int(result["fixture"]["video_bytes"])
    image_bytes = int(result["fixture"]["image_bytes"])
    stage_times = {
        name: sum(float(stage["wall_seconds"]) for stage in strategy["stages"].values())
        for name, strategy in result["strategies"].items()
    }
    window_times = {window["label"]: float(window["wall_seconds"]) for window in result["windows"]}
    result["client_access_summary"] = {
        "A_full_presync_local": {
            "object_downloads": 2,
            "downloaded_bytes": video_bytes + image_bytes,
            "wall_seconds": window_times["A_full_presync"] + stage_times["A_full_presync_local"],
        },
        "B_unit_prefetch_cache": {
            "object_downloads": 2,
            "downloaded_bytes": video_bytes + image_bytes,
            "wall_seconds": (
                window_times["B_video_prefetch"]
                + window_times["B_keyframe_prefetch"]
                + stage_times["B_unit_prefetch_cache"]
            ),
        },
        "C_minimal_cache_direct": {
            "object_downloads": 7,
            "downloaded_bytes": 3 * video_bytes + 4 * image_bytes,
            "wall_seconds": sum(
                float(stage["wall_seconds_with_acquisition"])
                for stage in result["strategies"]["C_minimal_cache_direct"]["stages"].values()
            ),
        },
    }


def main() -> int:
    args = parse_args()
    require_environment()
    root = args.output.resolve()
    if args.refresh_only:
        refresh_existing_result(root, args.bucket)
        print(f"refreshed_result={root / 'canary_results.json'}")
        return 0
    source_video = args.source_video.resolve()
    source_image = args.source_image.resolve()
    if not source_video.is_file() or not source_image.is_file():
        raise RuntimeError("the configured real video/keyframe fixture is unavailable")

    root.mkdir(parents=True, exist_ok=True)
    run_root = root / "runtime"
    safe_recreate(run_root, root)
    fixture_video, fixture_image = derive_fixture(source_video, source_image, run_root / "fixture")

    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    prefix = f"data02r/canary-{stamp}"
    video_uri = f"s3://{args.bucket}/{prefix}/fixture/video.mp4"
    image_uri = f"s3://{args.bucket}/{prefix}/fixture/keyframe.jpg"
    windows: list[dict[str, Any]] = []
    storage_before = query_storage(args.bucket)

    try:
        network_window(
            "fixture_upload",
            lambda: (
                aws("s3", "cp", str(fixture_video), video_uri, "--only-show-errors"),
                aws("s3", "cp", str(fixture_image), image_uri, "--only-show-errors"),
            ),
            windows,
        )
        storage_after_upload = query_storage(args.bucket)

        strategies: dict[str, Any] = {}

        # A: one full fixture sync, followed by local processing only.
        a_root = run_root / "strategy_a"
        safe_recreate(a_root, run_root)
        network_window(
            "A_full_presync",
            lambda: aws("s3", "sync", f"s3://{args.bucket}/{prefix}/fixture", str(a_root), "--only-show-errors"),
            windows,
        )
        a_results = {}
        for stage in STAGES:
            a_results[stage] = local_stage_window(
                "A", stage, a_root / "video.mp4", a_root / "keyframe.jpg", a_root / "outputs", windows
            )
        strategies["A_full_presync_local"] = {
            "stages": a_results,
            "local_disk_bytes": sum(path.stat().st_size for path in a_root.rglob("*") if path.is_file()),
        }

        # B: cache/prefetch by the repository's natural access unit.
        b_root = run_root / "strategy_b"
        safe_recreate(b_root, run_root)
        b_video = b_root / "video.mp4"
        b_image = b_root / "keyframe.jpg"
        network_window(
            "B_video_prefetch",
            lambda: aws("s3", "cp", video_uri, str(b_video), "--only-show-errors"),
            windows,
        )
        b_results = {}
        for stage in ("ingestion", "ocr", "asr"):
            b_results[stage] = local_stage_window(
                "B", stage, b_video, fixture_image, b_root / "outputs", windows
            )
        network_window(
            "B_keyframe_prefetch",
            lambda: aws("s3", "cp", image_uri, str(b_image), "--only-show-errors"),
            windows,
        )
        for stage in ("beit3", "metaclip2", "perception_encoder", "objects"):
            b_results[stage] = local_stage_window(
                "B", stage, b_video, b_image, b_root / "outputs", windows
            )
        strategies["B_unit_prefetch_cache"] = {
            "stages": b_results,
            "local_disk_bytes": sum(path.stat().st_size for path in b_root.rglob("*") if path.is_file()),
        }

        # C: acquire a fresh object for every stage and retain no shared cache.
        c_root = run_root / "strategy_c"
        safe_recreate(c_root, run_root)
        c_results = {}
        for stage in STAGES:
            stage_root = c_root / stage
            stage_root.mkdir(parents=True)
            local_video = stage_root / "video.mp4"
            local_image = stage_root / "keyframe.jpg"
            uri = video_uri if stage in ("ingestion", "ocr", "asr") else image_uri
            target = local_video if stage in ("ingestion", "ocr", "asr") else local_image
            start = utc_now() - timedelta(seconds=0.5)
            started = time.perf_counter()
            aws("s3", "cp", uri, str(target), "--only-show-errors")
            stage_result = run_stage(stage, local_video, local_image, stage_root / "outputs")
            elapsed = time.perf_counter() - started
            end = utc_now() + timedelta(seconds=0.5)
            windows.append(
                {
                    "label": f"C_{stage}",
                    "start": iso(start),
                    "end": iso(end),
                    "wall_seconds": elapsed,
                }
            )
            stage_result["wall_seconds_with_acquisition"] = elapsed
            c_results[stage] = stage_result
            target.unlink()
            time.sleep(1.25)
        strategies["C_minimal_cache_direct"] = {
            "stages": c_results,
            "local_disk_bytes": sum(path.stat().st_size for path in c_root.rglob("*") if path.is_file()),
        }

        # R2 operations generally appear within seconds, but allow one bounded retry.
        time.sleep(5)
        for attempt in range(2):
            for window in windows:
                window["cloudflare_operations"] = query_operations(
                    window["start"], window["end"], args.bucket
                )
            if any(window["cloudflare_operations"] for window in windows if window["label"].startswith("C_")):
                break
            if attempt == 0:
                time.sleep(10)

        result = {
            "schema_version": "data02r-r2-canary-v1",
            "created_at": iso(utc_now()),
            "bucket": args.bucket,
            "prefix": prefix,
            "fixture": {
                "source_video": source_video.as_posix(),
                "source_image": source_image.as_posix(),
                "derivation": "first 12 seconds, stream copy, video plus optional first audio stream",
                "video_bytes": fixture_video.stat().st_size,
                "video_sha256": sha256(fixture_video),
                "image_bytes": fixture_image.stat().st_size,
                "image_sha256": sha256(fixture_image),
                "keyframe_frame_idx": 125,
                "keyframe_timestamp": 5.0,
            },
            "inference": "mocked; production media decode and access paths executed",
            "storage_before": storage_before,
            "storage_after_upload": storage_after_upload,
            "windows": windows,
            "strategies": strategies,
        }
        add_client_summary(result)
        (root / "canary_results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    finally:
        cleanup_start = utc_now() - timedelta(seconds=0.5)
        aws("s3", "rm", f"s3://{args.bucket}/{prefix}", "--recursive", "--only-show-errors")
        cleanup_end = utc_now() + timedelta(seconds=0.5)
        cleanup = {
            "start": iso(cleanup_start),
            "end": iso(cleanup_end),
            "operations": query_operations(iso(cleanup_start), iso(cleanup_end), args.bucket),
        }
        (root / "cleanup_results.json").write_text(
            json.dumps(cleanup, indent=2, sort_keys=True), encoding="utf-8"
        )

    print(f"canary_result={root / 'canary_results.json'}")
    print(f"cleanup_result={root / 'cleanup_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
