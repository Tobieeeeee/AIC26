"""End-to-end offline pipeline runner (local workstation).

Autopilot that replaces the manual multi-command flow:

    pull video -> ingest -> RF-DETR objects + 3 embedding providers
        -> consolidate Parquet -> promote to canonical -> push to R2

Reads videos from the canonical R2 layout (data/raw/batch1/videos/...), so it
works correctly after the R2 reorg that moves Videos_* into data/raw. Processes
one video at a time to bound peak disk/VRAM, with resume via a state file.
Ingestion inherits the canonical detector and selector directly from
``mvr.ingestion.pipeline`` instead of freezing a second copy in this runner.

Usage:
    python scripts/operations/run_pipeline_local.py --device cuda
    python scripts/operations/run_pipeline_local.py --collections L21,L24 --videos L21_V001
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


LOGGER = logging.getLogger("run_pipeline_local")

CANONICAL_DETECTOR_THRESHOLD = 0.4
CANONICAL_PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
CANONICAL_BATCH = "1"
DEFAULT_RFDETR_CHECKPOINT = Path(
    "artifacts/verification/rfdetr-production-pin/models/rf-detr-large-2026.pth"
)
DEFAULT_BEIT3_CHECKPOINT = Path(
    "artifacts/_embedding_models/beit3/beit3_base_itc_patch16_224.pth"
)


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


@dataclass(frozen=True)
class RuntimeSettings:
    detector_checkpoint: Path
    beit3_checkpoint: Path
    pe_checkpoint: Path | None
    metaclip2_model: str | None
    allow_model_downloads: bool


def _r2_client():
    from mvr.operations.r2_client import R2Client
    return R2Client()


def discover_videos(
    client, collections: list[str], batch_id: str
) -> list[tuple[str, str, int]]:
    """List videos from the canonical layout: data/raw/batch1/videos/{collection}/*."""
    from mvr.operations.r2_layout import raw_video_prefix

    found: list[tuple[str, str, int]] = []
    for collection in collections:
        prefix = raw_video_prefix(batch_id, collection)
        for obj in client.list_objects(prefix):
            key = obj["Key"]
            if not key.endswith(".mp4"):
                continue
            video_id = Path(key).stem
            found.append((collection, video_id, int(obj.get("Size", 0))))
    found.sort(key=lambda t: (t[0], t[1]))
    return found


def download_video(
    client, collection: str, video_id: str, dest: Path, batch_id: str
) -> int:
    from mvr.operations.r2_layout import raw_video_key

    dest.parent.mkdir(parents=True, exist_ok=True)
    key = raw_video_key(batch_id, collection, video_id)
    client.download_file(key, dest)
    return dest.stat().st_size


def run_ingest(
    video_path: Path,
    out_root: Path,
    device: str,
    *,
    provider_specs_path: Path,
    metaclip2_model: str | None,
    allow_model_downloads: bool,
    embedding_batch_size: int,
) -> Path:
    from mvr.ingestion.pipeline import run_pipeline

    run_pipeline(
        input_path=video_path,
        output_root=out_root,
        device=device,
        provider_specs_path=provider_specs_path,
        metaclip2_model=metaclip2_model,
        allow_model_downloads=allow_model_downloads,
        embedding_batch_size=embedding_batch_size,
    )
    mapping = out_root / "keyframe_mapping.csv"
    if not mapping.is_file():
        raise FileNotFoundError(f"ingest produced no keyframe_mapping.csv in {out_root}")
    return mapping


def count_keyframes(mapping_csv: Path) -> int:
    with Path(mapping_csv).open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _build_provider_encoder(
    provider_id: str, spec, device: str, *,
    beit3_checkpoint, pe_checkpoint, metaclip2_model, allow_model_downloads,
):
    if provider_id == "beit3":
        from mvr.enrichment.embeddings.artifacts import verify_file_sha256
        from mvr.enrichment.embeddings.beit3 import Beit3ImageEncoder
        if beit3_checkpoint is None:
            raise RuntimeError("--beit3-checkpoint is required for beit3")
        verify_file_sha256(beit3_checkpoint, spec.checkpoint_sha256, "BEiT-3 checkpoint")
        return Beit3ImageEncoder(spec=spec, checkpoint_path=beit3_checkpoint, device=device)
    if provider_id == "metaclip2":
        from mvr.enrichment.embeddings.metaclip2 import MetaClip2ImageEncoder
        return MetaClip2ImageEncoder(
            spec=spec,
            model_location=metaclip2_model,
            device=device,
            allow_download=allow_model_downloads,
        )
    if provider_id == "perception_encoder":
        from mvr.enrichment.embeddings.perception_encoder import PerceptionEncoderImageEncoder
        if pe_checkpoint is None:
            raise RuntimeError("--pe-checkpoint is required for perception_encoder")
        return PerceptionEncoderImageEncoder(spec=spec, checkpoint_path=pe_checkpoint, device=device)
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


def run_objects_and_embeddings(
    mapping_csv, keyframe_root, out_dir, detector, video_id,
    provider_ids, provider_specs_path, beit3_checkpoint, pe_checkpoint,
    metaclip2_model, device, batch_size, allow_model_downloads,
):
    from mvr.enrichment.embeddings.offline import (
        extract_embedding_artifacts,
        reuse_embedding_artifact,
    )
    from mvr.enrichment.embeddings.provider_specs import load_provider_specs
    from mvr.enrichment.feature_extraction import extract_objects, sort_keyframes_deterministically

    keyframes = sort_keyframes_deterministically(mapping_csv)
    if not keyframes:
        return 0, {}
    timings: dict[str, float] = {}

    started = time.perf_counter()
    extract_objects(video_id, keyframes, detector, out_dir, keyframe_root, mapping_csv)
    timings["objects"] = time.perf_counter() - started

    specs = load_provider_specs(provider_specs_path)
    embeddings_root = out_dir / "embeddings"
    for provider_id in provider_ids:
        spec = specs.get(provider_id)
        if spec is None:
            raise RuntimeError(f"provider {provider_id} absent from {provider_specs_path}")
        started = time.perf_counter()
        if provider_id == "metaclip2":
            selector_artifact = (
                Path(mapping_csv).parent / "selector_embeddings" / "metaclip2"
            )
            if selector_artifact.is_dir():
                reuse_embedding_artifact(
                    selector_artifact,
                    embeddings_root=embeddings_root,
                    rows=keyframes,
                    spec=spec,
                )
                timings[provider_id] = time.perf_counter() - started
                LOGGER.info(
                    "reused validated selector MetaCLIP-2 artifact: %s",
                    selector_artifact,
                )
                continue
        encoder = _build_provider_encoder(
            provider_id, spec, device,
            beit3_checkpoint=beit3_checkpoint, pe_checkpoint=pe_checkpoint,
            metaclip2_model=metaclip2_model,
            allow_model_downloads=allow_model_downloads,
        )
        try:
            extract_embedding_artifacts(
                keyframes, keyframe_root=keyframe_root,
                embeddings_root=embeddings_root, spec=spec,
                encoder=encoder, batch_size=batch_size,
            )
        finally:
            del encoder
            _free_vram()
        timings[provider_id] = time.perf_counter() - started
    return len(keyframes), timings


def consolidate_feature(out_dir: Path, feature: str, batch_id: str, collection_id: str) -> dict:
    """Consolidate a feature's staging JSON into a canonical Parquet."""
    from mvr.operations.consolidate import consolidate_partitioned
    source_dir = out_dir / feature
    if not source_dir.is_dir():
        return {"schema": feature, "skipped": True, "reason": "no staging dir"}
    output_root = out_dir / "corpus" / "metadata" / feature
    return consolidate_partitioned(
        source_dir, output_root, feature,
        batch_id=batch_id, collection_id=collection_id,
    )


def promote_feature(out_dir: Path, feature: str, batch_id: str, collection_id: str) -> dict:
    """Promote a consolidated Parquet to the canonical corpus path."""
    from mvr.operations.promote import promote_file
    part_dir = out_dir / "corpus" / "metadata" / feature / f"batch={batch_id}" / f"collection={collection_id}"
    staging = part_dir / "part-000.parquet"
    corpus_root = out_dir / "corpus_canonical" / "metadata" / feature
    corpus_path = corpus_root / f"batch={batch_id}" / f"collection={collection_id}" / "part-000.parquet"
    result = promote_file(staging, corpus_path, schema_name=feature)
    return result.to_dict()


def push_artifacts(sync, out_dir: Path, video_id: str, collection: str, run_id: str,
                   batch_id: str) -> int:
    """Push staging embeddings/objects and canonical Parquet metadata to R2."""
    from mvr.operations.r2_layout import corpus_metadata_key
    pushed = 0
    # staging objects
    obj_dir = out_dir / "objects" / video_id
    if obj_dir.is_dir():
        files = sorted(p for p in obj_dir.iterdir() if p.is_file())
        pushed += sync.push_staging(f"objects/{video_id}", run_id, files).uploaded
    # staging embeddings
    embeddings_root = out_dir / "embeddings"
    if embeddings_root.is_dir():
        for provider_dir in sorted(p for p in embeddings_root.iterdir() if p.is_dir()):
            files = sorted(p for p in provider_dir.rglob("*") if p.is_file())
            if files:
                pushed += sync.push_staging(
                    f"embeddings/{provider_dir.name}/{video_id}", run_id, files,
                ).uploaded
    # canonical Parquet metadata -> corpus tree
    corpus_meta_root = out_dir / "corpus_canonical" / "metadata"
    if corpus_meta_root.is_dir():
        for feature_dir in sorted(p for p in corpus_meta_root.iterdir() if p.is_dir()):
            for parquet in sorted(feature_dir.rglob("*.parquet")):
                feature = feature_dir.name
                key = corpus_metadata_key(feature, batch_id, collection)
                if sync.push_file(parquet, key, content_type="application/octet-stream"):
                    pushed += 1
    return pushed


def push_ingestion_artifacts(
    sync,
    out_dir: Path,
    *,
    video_id: str,
    collection: str,
    batch_id: str,
    run_id: str,
) -> int:
    """Push a publishable per-video ingestion tree and require its success marker."""
    stats = sync.push_ingestion_run(
        out_dir,
        run_id=run_id,
        batch_id=batch_id,
        collection_id=collection,
        video_id=video_id,
    )
    if stats.errors:
        raise RuntimeError(
            f"R2 ingestion upload incomplete for {video_id}: "
            f"uploaded={stats.uploaded}, errors={stats.errors}"
        )
    return stats.uploaded


def build_detector(checkpoint: Path, device: str, threshold: float):
    from mvr.enrichment.objects.rfdetr import RFDETRConfig, RFDETRDetector
    config = RFDETRConfig(checkpoint_path=checkpoint, device=device, score_threshold=threshold)
    if not config.is_pinned:
        raise RuntimeError("RF-DETR runtime is not pinned; refusing production extraction")
    return RFDETRDetector(config)


def _default_pe_checkpoint() -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    root = hub / "models--facebook--PE-Core-L14-336" / "snapshots"
    if not root.is_dir():
        return None
    for snapshot in sorted(root.iterdir()):
        candidate = snapshot / "PE-Core-L14-336.pt"
        if candidate.is_file():
            return candidate
    return None


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_runtime_settings(args: argparse.Namespace) -> RuntimeSettings:
    return RuntimeSettings(
        detector_checkpoint=(
            args.detector_checkpoint
            or _env_path("MVR_RFDETR_CHECKPOINT")
            or DEFAULT_RFDETR_CHECKPOINT
        ),
        beit3_checkpoint=(
            args.beit3_checkpoint
            or _env_path("MVR_BEIT3_CHECKPOINT")
            or DEFAULT_BEIT3_CHECKPOINT
        ),
        pe_checkpoint=(
            args.pe_checkpoint
            or _env_path("MVR_PE_CHECKPOINT")
            or _default_pe_checkpoint()
        ),
        metaclip2_model=(
            args.metaclip2_model
            or os.environ.get("MVR_METACLIP2_MODEL", "").strip()
            or None
        ),
        allow_model_downloads=(
            args.allow_model_downloads
            or _env_flag("MVR_METACLIP2_ALLOW_DOWNLOAD")
        ),
    )


def load_state(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("state file %s corrupt; starting fresh", path)
    return {"completed": {}, "failed": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python scripts/operations/run_pipeline_local.py",
        description="Local autopilot: pull -> ingest -> objects+embeddings -> Parquet -> promote -> push.",
    )
    p.add_argument("--collections", default=None, help="Comma-separated collection IDs (e.g. L21,L24)")
    p.add_argument("--videos", default=None, help="Comma-separated video IDs (requires a single --collections)")
    p.add_argument("--limit", type=int, default=0, help="Process at most N videos (0 = all)")
    p.add_argument("--work-root", type=Path, default=Path("_corpus_work"))
    p.add_argument("--batch-id", default=CANONICAL_BATCH)
    p.add_argument("--detector-checkpoint", type=Path, default=None)
    p.add_argument("--detector-threshold", type=float, default=CANONICAL_DETECTOR_THRESHOLD)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--providers", default=",".join(CANONICAL_PROVIDERS))
    p.add_argument("--provider-specs", type=Path, default=Path("configs/embedding_providers.json"))
    p.add_argument("--beit3-checkpoint", type=Path, default=None)
    p.add_argument("--pe-checkpoint", type=Path, default=None)
    p.add_argument("--metaclip2-model", default=None)
    p.add_argument(
        "--allow-model-downloads",
        action="store_true",
        help="Explicitly allow pinned model downloads (disabled by default)",
    )
    p.add_argument("--embedding-batch-size", type=int, default=8)
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--ingest-only",
        action="store_true",
        help="Stop after canonical ingestion and upload its publishable artifacts",
    )
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--keep-video", action="store_true")
    p.add_argument(
        "--evict-artifacts-after-push",
        action="store_true",
        help=(
            "Delete each local ingestion output only after its R2 _SUCCESS marker "
            "is uploaded (requires --ingest-only with push enabled)"
        ),
    )
    p.add_argument("--state-file", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)
    if args.evict_artifacts_after_push and not args.ingest_only:
        p.error("--evict-artifacts-after-push requires --ingest-only")
    if args.evict_artifacts_after_push and args.no_push:
        p.error("--evict-artifacts-after-push requires R2 push to be enabled")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    except ImportError:
        # Explicit process environment variables remain fully supported.
        pass
    run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S")
    work = args.work_root.resolve()
    state_name = "ingest_state.json" if args.ingest_only else "state.json"
    report_name = "ingest_report.json" if args.ingest_only else "report.json"
    state_file = (args.state_file or work / state_name).resolve()
    report_file = (args.report or work / report_name).resolve()
    runtime = resolve_runtime_settings(args)

    client = _r2_client()
    sync = None if args.no_push else __import__("mvr.operations.r2_sync", fromlist=["R2Sync"]).R2Sync(client=client)

    provider_ids = tuple(p.strip() for p in args.providers.split(",") if p.strip())
    if not args.ingest_only and not provider_ids:
        raise SystemExit("--providers must name at least one provider")
    detector = None
    if not args.ingest_only:
        detector = build_detector(
            runtime.detector_checkpoint, args.device, args.detector_threshold
        )

    if args.collections:
        collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    else:
        collections = [f"L{n}" for n in range(21, 31)]
    if args.videos:
        if len(collections) != 1:
            raise SystemExit("--videos requires exactly one --collections")
        video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
        queue = [(collections[0], vid, 0) for vid in video_ids]
    else:
        queue = discover_videos(client, collections, args.batch_id)
    if args.limit:
        queue = queue[: args.limit]

    state = load_state(state_file)
    results: list[VideoResult] = []
    total = len(queue)
    mode = "ingest-only" if args.ingest_only else "full"
    LOGGER.info(
        "run_id=%s mode=%s collections=%s videos=%d device=%s",
        run_id,
        mode,
        collections,
        total,
        args.device,
    )
    completion_token = "INGEST_DONE" if args.ingest_only else "DONE"

    for idx, (collection, video_id, size) in enumerate(queue, 1):
        res = VideoResult(video_id=video_id, collection=collection)
        if state["completed"].get(video_id):
            LOGGER.info("[%d/%d] SKIP (already done) %s/%s", idx, total, collection, video_id)
            res.status = "SKIPPED"
            results.append(res)
            continue

        video_dir = work / "videos" / collection
        video_path = video_dir / f"{video_id}.mp4"
        output_namespace = "ingestion" if args.ingest_only else "artifacts"
        out_dir = work / output_namespace / collection / video_id
        status_file = out_dir / f"{video_id}.status"
        if (
            status_file.is_file()
            and status_file.read_text().strip() == completion_token
        ):
            state["completed"][video_id] = True
            save_state(state_file, state)
            res.status = "SKIPPED"
            results.append(res)
            continue

        LOGGER.info("[%d/%d] %s/%s (%.1f MiB)", idx, total, collection, video_id, size / 1e6)
        t0 = time.perf_counter()
        try:
            # 1. pull
            ta = time.perf_counter()
            if not video_path.is_file():
                res.bytes_downloaded = download_video(
                    client, collection, video_id, video_path, args.batch_id
                )
            res.seconds["pull"] = time.perf_counter() - ta

            # 2. ingest
            ta = time.perf_counter()
            out_dir.mkdir(parents=True, exist_ok=True)
            mapping = run_ingest(
                video_path,
                out_dir,
                args.device,
                provider_specs_path=args.provider_specs,
                metaclip2_model=runtime.metaclip2_model,
                allow_model_downloads=runtime.allow_model_downloads,
                embedding_batch_size=args.embedding_batch_size,
            )
            res.seconds["ingest"] = time.perf_counter() - ta
            res.keyframe_count = count_keyframes(mapping)

            if args.ingest_only:
                if sync is not None:
                    ta = time.perf_counter()
                    pushed = push_ingestion_artifacts(
                        sync,
                        out_dir,
                        video_id=video_id,
                        collection=collection,
                        batch_id=args.batch_id,
                        run_id=run_id,
                    )
                    res.seconds["push"] = time.perf_counter() - ta
                    res.seconds["pushed_files"] = pushed
            else:
                # 3. objects + embeddings
                ta = time.perf_counter()
                res.keyframe_count, provider_timings = run_objects_and_embeddings(
                    mapping, out_dir, out_dir, detector, video_id,
                    provider_ids, args.provider_specs,
                    runtime.beit3_checkpoint, runtime.pe_checkpoint,
                    runtime.metaclip2_model, args.device, args.embedding_batch_size,
                    runtime.allow_model_downloads,
                )
                res.seconds["enrich"] = time.perf_counter() - ta
                res.seconds.update(provider_timings)

                # 4. consolidate Parquet (objects + extraction_status)
                ta = time.perf_counter()
                for feature in ("objects", "extraction_status"):
                    consolidate_feature(out_dir, feature, args.batch_id, collection)
                res.seconds["consolidate"] = time.perf_counter() - ta

                # 5. promote to canonical
                ta = time.perf_counter()
                for feature in ("objects", "extraction_status"):
                    promote_feature(out_dir, feature, args.batch_id, collection)
                res.seconds["promote"] = time.perf_counter() - ta

                # 6. push
                if sync is not None:
                    ta = time.perf_counter()
                    pushed = push_artifacts(
                        sync, out_dir, video_id, collection, run_id, args.batch_id
                    )
                    res.seconds["push"] = time.perf_counter() - ta
                    res.seconds["pushed_files"] = pushed

            status_file.write_text(f"{completion_token}\n", encoding="utf-8")
            state["completed"][video_id] = True
            state["failed"].pop(video_id, None)
            save_state(state_file, state)
            if not args.keep_video and video_path.is_file():
                try:
                    video_path.unlink()
                except OSError as exc:
                    LOGGER.warning("could not evict raw video %s: %s", video_path, exc)
            if args.evict_artifacts_after_push:
                shutil.rmtree(out_dir, ignore_errors=True)
                if out_dir.exists():
                    LOGGER.warning("could not evict ingestion artifacts: %s", out_dir)
                else:
                    LOGGER.info("evicted uploaded ingestion artifacts: %s", out_dir)
            res.status = "DONE"
        except Exception as exc:
            LOGGER.error("[%d/%d] FAILED %s/%s: %s", idx, total, collection, video_id, exc)
            res.status = "FAILED"
            res.error = f"{type(exc).__name__}: {exc}"
            state["failed"][video_id] = res.error
            save_state(state_file, state)
            if out_dir.exists() and not args.ingest_only:
                shutil.rmtree(out_dir, ignore_errors=True)
            elif out_dir.exists():
                LOGGER.info("preserving incomplete ingestion output for retry: %s", out_dir)
        res.seconds["total"] = time.perf_counter() - t0
        results.append(res)

    done = sum(1 for r in results if r.status == "DONE")
    failed = sum(1 for r in results if r.status == "FAILED")
    skipped = sum(1 for r in results if r.status == "SKIPPED")
    report = {
        "run_id": run_id, "mode": mode, "device": args.device,
        "batch_id": args.batch_id,
        "checkpoint": str(runtime.detector_checkpoint),
        "collections": collections,
        "summary": {"total": total, "done": done, "failed": failed, "skipped": skipped},
        "results": [r.to_dict() for r in results],
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("DONE run_id=%s done=%d failed=%d skipped=%d report=%s", run_id, done, failed, skipped, report_file)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
