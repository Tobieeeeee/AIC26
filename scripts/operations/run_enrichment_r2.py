"""Optimized enrichment-only runner for an already-ingested R2 corpus."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.operations.r2_client import R2Client
from mvr.operations.r2_enrichment import (
    EnrichmentRunConfig,
    R2EnrichmentRunner,
)
from mvr.operations.r2_sync import R2Sync


LOGGER = logging.getLogger("run_enrichment_r2")
DEFAULT_RFDETR_CHECKPOINT = Path(
    "artifacts/verification/rfdetr-production-pin/models/rf-detr-large-2026.pth"
)
DEFAULT_BEIT3_CHECKPOINT = Path(
    "artifacts/_embedding_models/beit3/beit3_base_itc_patch16_224.pth"
)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _default_pe_checkpoint() -> Path | None:
    configured = _env_path("MVR_PE_CHECKPOINT")
    if configured is not None:
        return configured
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = hub / "models--facebook--PE-Core-L14-336" / "snapshots"
    if snapshots.is_dir():
        for snapshot in sorted(snapshots.iterdir()):
            candidate = snapshot / "PE-Core-L14-336.pt"
            if candidate.is_file():
                return candidate
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "R2 canonical keyframes -> RF-DETR + BEiT-3 + selector MetaCLIP-2 "
            "+ PE-Core-L -> run-scoped R2 staging"
        )
    )
    parser.add_argument(
        "--collections",
        default=",".join(f"L{value}" for value in range(21, 31)),
        help="Comma-separated canonical collections",
    )
    parser.add_argument("--videos", default=None, help="Optional comma-separated canary videos")
    parser.add_argument("--batch-id", default="1")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--work-root", type=Path, default=Path("_enrichment_work"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download-workers", type=int, default=24)
    parser.add_argument("--upload-workers", type=int, default=12)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument(
        "--provider-specs",
        type=Path,
        default=Path("configs/embedding_providers.json"),
    )
    parser.add_argument("--detector-checkpoint", type=Path, default=None)
    parser.add_argument("--beit3-checkpoint", type=Path, default=None)
    parser.add_argument("--pe-checkpoint", type=Path, default=None)
    parser.add_argument("--detector-threshold", type=float, default=0.4)
    parser.add_argument(
        "--evict-after-push",
        action="store_true",
        help="Remove local collection artifacts/keyframes only after its R2 success marker",
    )
    args = parser.parse_args(argv)
    if args.download_workers < 1 or args.upload_workers < 1:
        parser.error("worker counts must be positive")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be positive")
    return args


def _build_detector(checkpoint: Path, device: str, threshold: float):
    from mvr.enrichment.objects.rfdetr import RFDETRConfig, RFDETRDetector

    config = RFDETRConfig(
        checkpoint_path=checkpoint,
        device=device,
        score_threshold=threshold,
    )
    if not config.is_pinned:
        raise RuntimeError("RF-DETR runtime is not pinned")
    return RFDETRDetector(config)


def _build_encoder(provider_id: str, spec, *, device: str, beit3: Path, pe: Path):
    if provider_id == "beit3":
        from mvr.enrichment.embeddings.beit3 import Beit3ImageEncoder

        return Beit3ImageEncoder(spec=spec, checkpoint_path=beit3, device=device)
    if provider_id == "perception_encoder":
        from mvr.enrichment.embeddings.perception_encoder import (
            PerceptionEncoderImageEncoder,
        )

        return PerceptionEncoderImageEncoder(
            spec=spec,
            checkpoint_path=pe,
            device=device,
        )
    raise RuntimeError(
        f"{provider_id} should not construct an encoder; MetaCLIP-2 is reused from ingestion"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    except ImportError:
        pass

    collections = tuple(
        value.strip() for value in args.collections.split(",") if value.strip()
    )
    videos = (
        {value.strip() for value in args.videos.split(",") if value.strip()}
        if args.videos
        else None
    )
    run_id = args.run_id or time.strftime("enrich-%Y%m%d-%H%M%S")
    detector_checkpoint = (
        args.detector_checkpoint
        or _env_path("MVR_RFDETR_CHECKPOINT")
        or DEFAULT_RFDETR_CHECKPOINT
    )
    beit3_checkpoint = (
        args.beit3_checkpoint
        or _env_path("MVR_BEIT3_CHECKPOINT")
        or DEFAULT_BEIT3_CHECKPOINT
    )
    pe_checkpoint = args.pe_checkpoint or _default_pe_checkpoint()
    if pe_checkpoint is None:
        LOGGER.error(
            "PE-Core-L14-336 checkpoint not found; set MVR_PE_CHECKPOINT or pass --pe-checkpoint"
        )
        return 2

    try:
        client = R2Client(
            max_pool_connections=max(args.download_workers, args.upload_workers)
        )
        sync = R2Sync(client=client, local_root=args.work_root)
        specs = load_provider_specs(args.provider_specs)
        runner = R2EnrichmentRunner(
            sync=sync,
            config=EnrichmentRunConfig(
                run_id=run_id,
                batch_id=args.batch_id,
                collections=collections,
                work_root=args.work_root,
                keyframe_download_workers=args.download_workers,
                upload_workers=args.upload_workers,
                embedding_batch_size=args.embedding_batch_size,
                evict_after_push=args.evict_after_push,
            ),
            provider_specs=specs,
            detector_factory=lambda: _build_detector(
                detector_checkpoint, args.device, args.detector_threshold
            ),
            encoder_factory=lambda provider_id, spec: _build_encoder(
                provider_id,
                spec,
                device=args.device,
                beit3=beit3_checkpoint,
                pe=pe_checkpoint,
            ),
            video_ids=videos,
        )
        report = runner.run()
    except Exception as exc:
        LOGGER.exception("enrichment failed: %s", exc)
        return 2

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
