"""One-command R2 corpus runner: pull -> ingest -> objects + 3 CLIP providers -> push -> evict.

Processes videos one at a time so peak disk stays bounded: each video is
downloaded from R2, ingested (TransNetV2 keyframes), enriched (RF-DETR objects +
3 production visual-embedding providers), its artifacts pushed to R2 staging,
and then the local video is deleted before the next one starts.

Resume is automatic: a video whose per-video status file already reads DONE is
skipped without any R2 GET.

Canonical defaults are preserved: TransNetV2 @ 0.5 + periodic 12s/8s,
RF-DETR Large @ 0.4 pinned, and the 3-provider visual-embedding stack
(BEiT-3 + MetaCLIP-2 + PE-Core-L). Encoders are built lazily and unloaded
after each video to keep VRAM bounded.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger("run_r2_corpus")

CANONICAL_TRANSNET_THRESHOLD = 0.5
CANONICAL_LONG_SHOT_SECONDS = 12.0
CANONICAL_SAMPLE_EVERY_SECONDS = 8.0
CANONICAL_DETECTOR_THRESHOLD = 0.4
CANONICAL_PROVIDERS = ("beit3", "metaclip2", "perception_encoder")


@dataclass
class VideoResult:
    video_id: str
    collection: str
    status: str = "PENDING"
    error: str | None = None
    keyframe_count: int = 0
    bytes_downloaded: int = 0
    seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "collection": self.collection,
            "status": self.status,
            "error": self.error,
            "keyframe_count": self.keyframe_count,
            "bytes_downloaded": self.bytes_downloaded,
            "seconds": {k: round(v, 3) for k, v in self.seconds.items()},
        }


def _r2_client():
    from mvr.operations.r2_client import R2Client
    return R2Client()


def discover_videos(client, collections: list[str]) -> list[tuple[str, str, int]]:
    """Return [(collection_prefix, video_id, size_bytes)] sorted deterministically."""
    found: list[tuple[str, str, int]] = []
    for prefix in collections:
        for obj in client.list_objects(f"{prefix}/video/"):
            key = obj["Key"]
            if not key.endswith(".mp4"):
                continue
            video_id = Path(key).stem
            found.append((prefix, video_id, int(obj.get("Size", 0))))
    found.sort(key=lambda t: (t[0], t[1]))
    return found


def list_collections(client, start_from: str) -> list[str]:
    """Every Videos_Lxx_y prefix at or after start_from, in sorted order."""
    seen: set[str] = set()
    for obj in client.list_objects(""):
        top = obj["Key"].split("/", 1)[0]
        if top.startswith("Videos_L"):
            seen.add(top)
    ordered = sorted(seen)
    if start_from:
        ordered = [p for p in ordered if _collection_sort_key(p) >= _collection_sort_key(start_from)]
    return ordered


def _collection_sort_key(prefix: str) -> tuple[int, str]:
    """Sort Videos_L25_a before Videos_L26_a, and L26_a before L26_b."""
    body = prefix.replace("Videos_L", "")
    num, _, suffix = body.partition("_")
    try:
        return (int(num), suffix)
    except ValueError:
        return (9999, prefix)


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------


def download_video(client, prefix: str, video_id: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    key = f"{prefix}/video/{video_id}.mp4"
    client.download_file(key, dest)
    return dest.stat().st_size


def run_ingest(video_path: Path, out_root: Path, device: str) -> Path:
    """TransNetV2 @0.5 + periodic 12s/8s on a single video."""
    from mvr.ingestion.pipeline import run_pipeline

    run_pipeline(
        input_path=video_path,
        output_root=out_root,
        methods=["transnetv2"],
        primary="transnetv2",
        transnet_threshold=CANONICAL_TRANSNET_THRESHOLD,
        selector_mode="periodic",
        long_shot_seconds=CANONICAL_LONG_SHOT_SECONDS,
        sample_every_seconds=CANONICAL_SAMPLE_EVERY_SECONDS,
        device=device,
    )
    mapping = out_root / "keyframe_mapping.csv"
    if not mapping.is_file():
        raise FileNotFoundError(f"ingest produced no keyframe_mapping.csv in {out_root}")
    return mapping


def run_objects_and_embeddings(
    mapping_csv: Path,
    keyframe_root: Path,
    out_dir: Path,
    detector,
    dataset_name: str,
    provider_ids: tuple[str, ...],
    provider_specs_path: Path,
    beit3_checkpoint: Path | None,
    pe_checkpoint: Path | None,
    metaclip2_model: str | None,
    device: str,
    batch_size: int,
) -> tuple[int, dict[str, float]]:
    """RF-DETR objects + each production embedding provider, one at a time.

    Providers are built and torn down sequentially so only one encoder occupies
    VRAM at any moment — required on a 6 GB card.
    """
    from mvr.enrichment.embeddings.offline import extract_embedding_artifacts
    from mvr.enrichment.embeddings.provider_specs import load_provider_specs
    from mvr.enrichment.feature_extraction import (
        extract_objects,
        sort_keyframes_deterministically,
    )

    keyframes = sort_keyframes_deterministically(mapping_csv)
    if not keyframes:
        return 0, {}
    timings: dict[str, float] = {}

    started = time.perf_counter()
    extract_objects(
        dataset_name, keyframes, detector, out_dir, keyframe_root, mapping_csv
    )
    timings["objects"] = time.perf_counter() - started

    specs = load_provider_specs(provider_specs_path)
    embeddings_root = out_dir / "embeddings"
    for provider_id in provider_ids:
        spec = specs.get(provider_id)
        if spec is None:
            raise RuntimeError(f"provider {provider_id} is absent from {provider_specs_path}")
        started = time.perf_counter()
        encoder = _build_provider_encoder(
            provider_id, spec, device,
            beit3_checkpoint=beit3_checkpoint,
            pe_checkpoint=pe_checkpoint,
            metaclip2_model=metaclip2_model,
        )
        try:
            extract_embedding_artifacts(
                keyframes,
                keyframe_root=keyframe_root,
                embeddings_root=embeddings_root,
                spec=spec,
                encoder=encoder,
                batch_size=batch_size,
            )
        finally:
            del encoder
            _free_vram()
        timings[provider_id] = time.perf_counter() - started
    return len(keyframes), timings


def _build_provider_encoder(
    provider_id: str,
    spec,
    device: str,
    *,
    beit3_checkpoint: Path | None,
    pe_checkpoint: Path | None,
    metaclip2_model: str | None,
):
    if provider_id == "beit3":
        from mvr.enrichment.embeddings.artifacts import verify_file_sha256
        from mvr.enrichment.embeddings.beit3 import Beit3ImageEncoder

        if beit3_checkpoint is None:
            raise RuntimeError("--beit3-checkpoint is required for the beit3 provider")
        verify_file_sha256(beit3_checkpoint, spec.checkpoint_sha256, "BEiT-3 checkpoint")
        return Beit3ImageEncoder(
            spec=spec, checkpoint_path=beit3_checkpoint, device=device
        )
    if provider_id == "metaclip2":
        from mvr.enrichment.embeddings.metaclip2 import MetaClip2ImageEncoder

        return MetaClip2ImageEncoder(
            spec=spec, model_location=metaclip2_model, device=device,
            allow_download=False,
        )
    if provider_id == "perception_encoder":
        from mvr.enrichment.embeddings.perception_encoder import (
            PerceptionEncoderImageEncoder,
        )

        if pe_checkpoint is None:
            raise RuntimeError(
                "--pe-checkpoint is required for the perception_encoder provider"
            )
        return PerceptionEncoderImageEncoder(
            spec=spec, checkpoint_path=pe_checkpoint, device=device
        )
    raise RuntimeError(f"unknown provider id: {provider_id}")


def _free_vram() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def push_artifacts(sync, out_dir: Path, video_id: str, run_id: str) -> int:
    """Push this video's object JSONs + per-provider embedding artifacts."""
    pushed = 0
    obj_dir = out_dir / "objects" / video_id
    if obj_dir.is_dir():
        files = sorted(p for p in obj_dir.iterdir() if p.is_file())
        pushed += sync.push_staging(f"objects/{video_id}", run_id, files).uploaded
    embeddings_root = out_dir / "embeddings"
    if embeddings_root.is_dir():
        for provider_dir in sorted(p for p in embeddings_root.iterdir() if p.is_dir()):
            files = sorted(p for p in provider_dir.rglob("*") if p.is_file())
            if files:
                pushed += sync.push_staging(
                    f"embeddings/{provider_dir.name}/{video_id}", run_id, files
                ).uploaded
    return pushed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_detector(checkpoint: Path, device: str, threshold: float):
    from mvr.enrichment.objects.rfdetr import RFDETRConfig, RFDETRDetector

    config = RFDETRConfig(
        checkpoint_path=checkpoint, device=device, score_threshold=threshold
    )
    if not config.is_pinned:
        raise RuntimeError(
            "RF-DETR runtime is not pinned; refusing to run a production extraction"
        )
    return RFDETRDetector(config)


def _default_pe_checkpoint() -> Path | None:
    """The PE-Core-L checkpoint inside the local HF cache, if it is there."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    root = hub / "models--facebook--PE-Core-L14-336" / "snapshots"
    if not root.is_dir():
        return None
    for snapshot in sorted(root.iterdir()):
        candidate = snapshot / "PE-Core-L14-336.pt"
        if candidate.is_file():
            return candidate
    return None


def load_state(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("state file %s is corrupt; starting fresh", path)
    return {"completed": {}, "failed": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python scripts/run_r2_corpus.py",
        description="Pull from R2, ingest, run RF-DETR objects + OpenCLIP, push, evict.",
    )
    p.add_argument(
        "--start-from", default="Videos_L25_a",
        help="First collection prefix to process (default Videos_L25_a)",
    )
    p.add_argument(
        "--collections", default=None,
        help="Explicit comma-separated collection prefixes; overrides --start-from",
    )
    p.add_argument(
        "--videos", default=None,
        help="Explicit comma-separated video IDs (requires a single --collections)",
    )
    p.add_argument("--limit", type=int, default=0, help="Process at most N videos (0 = all)")
    p.add_argument(
        "--work-root", type=Path, default=Path("_corpus_work"),
        help="Local scratch root for videos + artifacts",
    )
    p.add_argument(
        "--detector-checkpoint", type=Path,
        default=Path("artifacts/verification/rfdetr-production-pin/models/rf-detr-large-2026.pth"),
    )
    p.add_argument("--detector-threshold", type=float, default=CANONICAL_DETECTOR_THRESHOLD)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument(
        "--providers", default=",".join(CANONICAL_PROVIDERS),
        help="Comma-separated embedding providers (default: the 3-provider production stack)",
    )
    p.add_argument(
        "--provider-specs", type=Path, default=Path("configs/embedding_providers.json"),
    )
    p.add_argument(
        "--beit3-checkpoint", type=Path,
        default=Path("artifacts/_embedding_models/beit3/beit3_base_itc_patch16_224.pth"),
    )
    p.add_argument(
        "--pe-checkpoint", type=Path, default=None,
        help="PE-Core-L checkpoint; defaults to the HF cache copy when present",
    )
    p.add_argument(
        "--metaclip2-model", default=None,
        help="Pinned local MetaCLIP-2 model path; defaults to the locked HF id in the cache",
    )
    p.add_argument("--embedding-batch-size", type=int, default=8)
    p.add_argument("--run-id", default=None, help="Staging run id (default: timestamp)")
    p.add_argument(
        "--no-push", action="store_true",
        help="Skip pushing artifacts to R2 (local-only run)",
    )
    p.add_argument(
        "--keep-video", action="store_true",
        help="Do not delete the local video after processing (uses far more disk)",
    )
    p.add_argument(
        "--state-file", type=Path, default=None,
        help="Resume state JSON (default <work-root>/state.json)",
    )
    p.add_argument(
        "--report", type=Path, default=None,
        help="Where to write the final run report (default <work-root>/report.json)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S")
    work = args.work_root.resolve()
    state_file = (args.state_file or work / "state.json").resolve()
    report_file = (args.report or work / "report.json").resolve()

    client = _r2_client()
    sync = None if args.no_push else __import__("mvr.operations.r2_sync", fromlist=["R2Sync"]).R2Sync(client=client)

    provider_ids = tuple(p.strip() for p in args.providers.split(",") if p.strip())
    if not provider_ids:
        raise SystemExit("--providers must name at least one provider")
    pe_checkpoint = args.pe_checkpoint or _default_pe_checkpoint()
    detector = build_detector(args.detector_checkpoint, args.device, args.detector_threshold)

    if args.collections:
        collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    else:
        collections = list_collections(client, args.start_from)
    if not collections:
        LOGGER.error("no collections found at/after %s", args.start_from)
        return 1

    if args.videos:
        if len(collections) != 1:
            raise SystemExit("--videos requires exactly one --collections")
        video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
        queue = [(collections[0], vid, 0) for vid in video_ids]
    else:
        queue = discover_videos(client, collections)
    if args.limit:
        queue = queue[: args.limit]

    state = load_state(state_file)
    results: list[VideoResult] = []
    total = len(queue)
    LOGGER.info(
        "run_id=%s collections=%d videos=%d device=%s checkpoint=%s",
        run_id, len(collections), total, args.device, args.detector_checkpoint,
    )

    for idx, (prefix, video_id, size) in enumerate(queue, 1):
        res = VideoResult(video_id=video_id, collection=prefix)
        if state["completed"].get(video_id):
            LOGGER.info("[%d/%d] SKIP (already done) %s/%s", idx, total, prefix, video_id)
            res.status = "SKIPPED"
            results.append(res)
            continue

        video_dir = work / "videos" / prefix
        video_path = video_dir / f"{video_id}.mp4"
        out_dir = work / "artifacts" / prefix / video_id
        status_file = out_dir / f"{video_id}.status"
        if status_file.is_file() and status_file.read_text().strip() == "DONE":
            state["completed"][video_id] = True
            save_state(state_file, state)
            res.status = "SKIPPED"
            results.append(res)
            continue

        LOGGER.info("[%d/%d] %s/%s (%.1f MiB)", idx, total, prefix, video_id, size / 1e6)
        t0 = time.perf_counter()
        try:
            # 1. pull
            ta = time.perf_counter()
            if not video_path.is_file():
                res.bytes_downloaded = download_video(client, prefix, video_id, video_path)
            res.seconds["pull"] = time.perf_counter() - ta

            # 2. ingest
            ta = time.perf_counter()
            out_dir.mkdir(parents=True, exist_ok=True)
            mapping = run_ingest(video_path, out_dir, args.device)
            res.seconds["ingest"] = time.perf_counter() - ta

            # 3. objects + embeddings
            ta = time.perf_counter()
            keyframe_root = out_dir
            res.keyframe_count, provider_timings = run_objects_and_embeddings(
                mapping, keyframe_root, out_dir, detector, video_id,
                provider_ids, args.provider_specs,
                args.beit3_checkpoint, pe_checkpoint, args.metaclip2_model,
                args.device, args.embedding_batch_size,
            )
            res.seconds["enrich"] = time.perf_counter() - ta
            res.seconds.update(provider_timings)

            # 4. push
            if sync is not None:
                ta = time.perf_counter()
                pushed = push_artifacts(sync, out_dir, video_id, run_id)
                res.seconds["push"] = time.perf_counter() - ta
                res.seconds["pushed_files"] = pushed

            # 5. mark done + evict local video
            status_file.write_text("DONE\n", encoding="utf-8")
            state["completed"][video_id] = True
            save_state(state_file, state)
            if not args.keep_video and video_path.is_file():
                video_path.unlink()
            res.status = "DONE"
        except Exception as exc:
            LOGGER.error("[%d/%d] FAILED %s/%s: %s", idx, total, prefix, video_id, exc)
            res.status = "FAILED"
            res.error = f"{type(exc).__name__}: {exc}"
            state["failed"][video_id] = res.error
            save_state(state_file, state)
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
        res.seconds["total"] = time.perf_counter() - t0
        results.append(res)

    # report
    done = sum(1 for r in results if r.status == "DONE")
    failed = sum(1 for r in results if r.status == "FAILED")
    skipped = sum(1 for r in results if r.status == "SKIPPED")
    report = {
        "run_id": run_id,
        "device": args.device,
        "checkpoint": str(args.detector_checkpoint),
        "collections": collections,
        "summary": {"total": total, "done": done, "failed": failed, "skipped": skipped},
        "results": [r.to_dict() for r in results],
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info(
        "DONE run_id=%s done=%d failed=%d skipped=%d report=%s",
        run_id, done, failed, skipped, report_file,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())




