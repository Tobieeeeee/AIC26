from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .enrichment.asr.audio import FfmpegAudioWindowProvider
from .enrichment.asr.chunkformer import (
    DEFAULT_MODEL_REVISION,
    ChunkFormerASRProvider,
)
from .enrichment.asr.pipeline import ASRPipelineConfig, build_asr_features
from .enrichment.asr.vad import SileroVADProvider
from .enrichment.embeddings.openclip import OpenClipEncoder
from .evaluation.benchmark import (
    KisBenchmarkQuery,
    KisBenchmarkRecord,
    build_aic_keyframe_mapping_csv,
    run_kis_benchmark_query,
    summarize_kis_benchmark,
)
from .indexing.faiss_index import FaissIndex
from .indexing.semantic_builder import build_semantic_index
from .ingestion.aic_dataset import AicDatasetRegistry
from .ingestion.pipeline import (
    CANONICAL_DETECTOR,
    CANONICAL_LONG_SHOT_SECONDS,
    CANONICAL_SAMPLE_EVERY_SECONDS,
    CANONICAL_SELECTOR_MODE,
    CANONICAL_TRANSNET_THRESHOLD,
    run_pipeline,
)
from .retrieval.frame_refinement import RefinementWindowConfig, refine_top_candidates
from .retrieval.kis import KisCandidate, search_kis
from .retrieval.semantic import SemanticRetriever


KIS_BENCHMARK_TOP_K = 100


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _required_manifest_string(manifest: dict, key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"semantic index manifest requires a non-empty {key}")
    return value


def _add_ingest_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("ingest", help="Probe videos and extract keyframes")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["pyscenedetect", "transnetv2"],
        default=[CANONICAL_DETECTOR],
    )
    parser.add_argument(
        "--primary",
        choices=["pyscenedetect", "transnetv2"],
        default=CANONICAL_DETECTOR,
    )
    parser.add_argument("--pyscene-threshold", type=float, default=27.0)
    parser.add_argument(
        "--transnet-threshold", type=float, default=CANONICAL_TRANSNET_THRESHOLD
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--selector-mode",
        choices=["midpoint", "periodic", "hybrid_temporal_content"],
        default=CANONICAL_SELECTOR_MODE,
    )
    parser.add_argument(
        "--long-shot-seconds", type=float, default=CANONICAL_LONG_SHOT_SECONDS,
        help="Periodic-mode long-shot threshold",
    )
    parser.add_argument(
        "--sample-every-seconds", type=float, default=CANONICAL_SAMPLE_EVERY_SECONDS,
        help="Periodic-mode interval",
    )
    parser.add_argument("--candidate-step-sec", type=float, default=1.0)
    parser.add_argument("--max-temporal-gap-sec", type=float, default=8.0)
    parser.add_argument("--clip-knn-k", type=_positive_int, default=3)
    parser.add_argument("--clip-neighbor-window-sec", type=float, default=4.0)
    parser.add_argument("--clip-novelty-threshold", type=float, default=0.15)
    parser.add_argument("--clip-rare-threshold", type=float, default=0.25)
    parser.add_argument("--min-event-gap-sec", type=float, default=2.0)
    parser.add_argument("--edge-change-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--edge-change-min-score", type=float, default=0.08)
    parser.add_argument(
        "--midpoint-mandatory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--provider-specs-path",
        type=Path,
        default=Path("configs/embedding_providers.json"),
    )
    parser.add_argument("--metaclip2-model")
    parser.add_argument("--allow-model-downloads", action="store_true")
    parser.add_argument("--embedding-batch-size", type=_positive_int, default=16)
    parser.set_defaults(command="ingest")


def _add_build_index_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "build-index",
        help="Encode Phase 02 keyframes and build an exact FAISS index",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.set_defaults(command="build-index")


def _add_search_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "search",
        help="Retrieve keyframes with a text query",
    )
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.set_defaults(command="search")


def _add_build_aic_index_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "build-aic-index",
        help="Build a semantic index directly from an AIC organizer dataset root",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Video IDs to index; defaults to every video with local keyframes.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.set_defaults(command="build-aic-index")


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _add_build_asr_features_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "build-asr-features",
        help="Build VAD-gated Vietnamese ASR JSON independently per keyframe",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-seconds", type=_non_negative_float, default=3.0)
    parser.add_argument("--post-seconds", type=_non_negative_float, default=3.0)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-duration-ms", type=_positive_int, default=250)
    parser.add_argument("--min-silence-duration-ms", type=_positive_int, default=150)
    parser.add_argument("--merge-gap-ms", type=int, default=250)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Optional video IDs (space- or comma-separated).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face model download; default is local-files-only.",
    )
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.set_defaults(command="build-asr-features")


def _normalize_video_args(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    video_ids = [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ]
    return video_ids or None


def _run_build_asr_features(
    *,
    dataset_root: Path,
    output: Path,
    pre_seconds: float,
    post_seconds: float,
    vad_threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    merge_gap_ms: int,
    device: str,
    videos: Sequence[str] | None,
    force: bool,
    allow_download: bool,
    model_revision: str | None,
) -> None:
    if merge_gap_ms < 0:
        raise ValueError("merge_gap_ms must be non-negative")
    vad = SileroVADProvider(
        threshold=vad_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    asr = ChunkFormerASRProvider(
        device=device,
        revision=model_revision,
        allow_download=allow_download,
    )
    manifest = build_asr_features(
        dataset_root,
        output,
        audio_provider=FfmpegAudioWindowProvider(),
        vad_provider=vad,
        asr_provider=asr,
        config=ASRPipelineConfig(
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
            merge_gap_ms=merge_gap_ms,
            force=force,
        ),
        videos=_normalize_video_args(videos),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def _add_enrich_ocr_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "enrich-ocr",
        help="Run temporal OCR enrichment on extracted keyframes",
    )
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--ocr-conf-threshold", type=float, default=0.50)
    parser.add_argument("--text-similarity-threshold", type=float, default=0.80)
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.30)
    parser.set_defaults(command="enrich-ocr")


def _discover_aic_video_ids(dataset_root: Path) -> list[str]:
    """Every video that actually has local keyframes, in stable order."""
    keyframe_root = Path(dataset_root) / "keyframes"
    if not keyframe_root.is_dir():
        raise FileNotFoundError(f"no keyframes directory under {dataset_root}")
    return sorted(
        directory.name
        for directory in keyframe_root.iterdir()
        if directory.is_dir() and (Path(dataset_root) / "map_keyframes" / f"{directory.name}.csv").is_file()
    )


def _run_build_aic_index(
    *, dataset_root: Path, output: Path, videos: Sequence[str] | None,
    batch_size: int, device: str,
) -> None:
    video_ids = list(videos) if videos else _discover_aic_video_ids(dataset_root)
    if not video_ids:
        raise ValueError(
            f"no indexable videos found under {dataset_root}: each needs both "
            "keyframes/<video_id>/ and map_keyframes/<video_id>.csv"
        )
    staging_root = Path(output).with_name(Path(output).name + "_ingest")
    build_aic_keyframe_mapping_csv(
        dataset_root, staging_root / "keyframe_mapping.csv", video_ids
    )
    encoder = OpenClipEncoder(device=device)
    index = build_semantic_index(staging_root, output, encoder, batch_size=batch_size)
    print(json.dumps(index.manifest, indent=2, ensure_ascii=False))


def _add_benchmark_kis_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "benchmark-kis",
        help="Run the real-data KIS benchmark (semantic vs semantic+refinement)",
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refine-top-n", type=_positive_int, default=3)
    parser.add_argument("--refine-frame-radius", type=_positive_int, default=45)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.set_defaults(command="benchmark-kis")


def _load_kis_benchmark_queries(gt_path: Path) -> list[KisBenchmarkQuery]:
    payload = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    label = payload.get("label", "internal")
    return [
        KisBenchmarkQuery(
            query_id=row["query_id"], query_text=row["query_text"],
            video_id=row["video_id"], start_frame=row["start_frame"],
            end_frame=row["end_frame"], label=label,
        )
        for row in payload["queries"]
    ]


def _run_benchmark_kis(
    *, index: Path, gt: Path, dataset_root: Path, output: Path,
    refine_top_n: int, refine_frame_radius: int, device: str,
) -> None:
    faiss_index = FaissIndex.load(index)
    encoder = OpenClipEncoder(
        model_name=_required_manifest_string(faiss_index.manifest, "embedding_model"),
        pretrained=_required_manifest_string(faiss_index.manifest, "pretrained"),
        device=device,
    )
    retriever = SemanticRetriever(encoder, faiss_index)
    registry = AicDatasetRegistry(dataset_root)
    window = RefinementWindowConfig(frame_radius=refine_frame_radius)

    def coarse_search(text: str) -> list[KisCandidate]:
        return search_kis(
            text, top_k=KIS_BENCHMARK_TOP_K, semantic_service=retriever,
            multimodal_service=None, aic_registry=registry,
        )

    def refined_search(text: str) -> list[KisCandidate]:
        return refine_top_candidates(
            coarse_search(text),
            query_vector=encoder.encode_text([text.strip()])[0],
            encoder=encoder,
            video_path_for=registry.video_path,
            video_info_for=registry.video_info,
            window=window,
            top_n=refine_top_n,
            enabled=True,
        )

    records: list[KisBenchmarkRecord] = []
    for query in _load_kis_benchmark_queries(gt):
        records.append(
            run_kis_benchmark_query(query, config_name="semantic", search_fn=coarse_search)
        )
        records.append(
            run_kis_benchmark_query(
                query, config_name="semantic+refine", search_fn=refined_search
            )
        )

    summary = summarize_kis_benchmark(records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.json").write_text(
        json.dumps(
            [
                {
                    "query_id": r.query_id, "config_name": r.config_name,
                    "r_at_k": r.r_at_k, "final_score": r.final_score,
                    "latency_seconds": r.latency_seconds,
                }
                for r in records
            ],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# Data-02 CLI commands
# --------------------------------------------------------------------------

def _add_show_defaults_parser(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser(
        "show-defaults",
        help="Print canonical production system defaults",
    ).set_defaults(command="show-defaults")


def _run_show_defaults() -> None:
    """Print the canonical production defaults summary."""
    try:
        import yaml
        config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "production.yaml"
        if config_path.is_file():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            config = None
    except ImportError:
        config = None

    lines = [
        "Canonical Production Defaults",
        "-----------------------------",
    ]
    if config:
        ing = config.get("ingestion", {})
        lines.append(
            f"Ingestion: {ing.get('detector', '?')} @ {ing.get('detector_threshold', '?')} "
            f"+ {ing.get('selector', '?')} {ing.get('long_threshold_sec', '?')}s/{ing.get('interval_sec', '?')}s"
        )
        obj = config.get("objects", {})
        pin = obj.get("pin_status", "unknown")
        lines.append(f"Objects:   {obj.get('provider', '?')} (pin: {pin})")
        emb = config.get("embeddings", {})
        providers = ", ".join(emb.get("providers", ["?"]))
        lines.append(f"Visual:    {providers}")
        ocr = config.get("ocr", {})
        lines.append(
            f"OCR:       {ocr.get('detector', '?')} + {ocr.get('recognizer', '?')} "
            f"+ {ocr.get('temporal_context', '?')}"
        )
        asr = config.get("asr", {})
        vad = asr.get("vad", {})
        lines.append(f"ASR:       {vad.get('provider', '?')} VAD + {asr.get('recognizer', '?')}")
    else:
        lines.extend([
            "Ingestion: TransNetV2 @ 0.5 + periodic 12s/8s",
            "Objects:   RFDETRLarge / rf-detr-large-2026.pth (pin: pinned)",
            "Visual:    beit3, metaclip2, perception_encoder",
            "OCR:       DeepSolo + Vietnamese PARSeq + F-1/F/F+1",
            "ASR:       Silero VAD + ChunkFormer CTC Large Vie",
        ])
    print("\n".join(lines))


def _add_audit_status_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "audit-status",
        help="Show extraction status summary from a status Parquet file",
    )
    parser.add_argument("--status-file", type=Path, required=True)
    parser.set_defaults(command="audit-status")


def _run_audit_status(status_file: Path) -> None:
    from .operations.extraction_status import summary
    result = summary(status_file)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _add_consolidate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "consolidate-parquet",
        help="Consolidate staging JSON into canonical Parquet",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", required=True,
                        choices=["ocr", "asr", "objects", "keyframe_mapping", "extraction_status"])
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--collection-id", default=None)
    parser.set_defaults(command="consolidate-parquet")


def _run_consolidate(source: Path, output: Path, schema: str,
                     batch_id: str | None, collection_id: str | None) -> None:
    from .operations.consolidate import consolidate_to_parquet
    result = consolidate_to_parquet(
        source, output, schema,
        batch_id=batch_id, collection_id=collection_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _add_validate_staging_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "validate-staging",
        help="Validate a staging Parquet artifact",
    )
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--schema", default=None,
                        choices=["ocr", "asr", "objects", "keyframe_mapping", "extraction_status"])
    parser.add_argument("--min-rows", type=int, default=0)
    parser.set_defaults(command="validate-staging")


def _run_validate_staging(file: Path, schema: str | None, min_rows: int) -> None:
    from .operations.promote import validate_parquet
    checks = validate_parquet(file, schema_name=schema, min_rows=min_rows)
    print(json.dumps(checks, indent=2, ensure_ascii=False))


def _add_promote_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "promote",
        help="Validate and promote a staging artifact to corpus",
    )
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.set_defaults(command="promote")


def _run_promote(staging: Path, corpus: Path, schema: str | None,
                 manifest: Path | None) -> None:
    from .operations.promote import promote_file
    result = promote_file(
        staging, corpus,
        schema_name=schema,
        manifest_path=manifest,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if not result.success:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Milvus commands (TASK05-MILVUS)
# ---------------------------------------------------------------------------

def _add_milvus_parser(subparsers: "argparse._SubParsersAction") -> None:
    milvus_parser = subparsers.add_parser("milvus", help="Milvus semantic backend operations")
    milvus_subs = milvus_parser.add_subparsers(dest="milvus_command", required=True)

    # milvus status
    p = milvus_subs.add_parser("status", help="Check Milvus connection and collection status")
    p.add_argument("--collection", default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.set_defaults(command="milvus", milvus_command="status")

    # milvus init-schema
    p = milvus_subs.add_parser("init-schema", help="Create aic26_keyframes collection schema")
    p.add_argument("--collection", default=None)
    p.add_argument("--drop-existing", action="store_true")
    p.set_defaults(command="milvus", milvus_command="init-schema")

    # milvus ingest
    p = milvus_subs.add_parser(
        "ingest",
        help="Ingest eligible keyframes from canonical Data-02 Parquet into Milvus",
    )
    p.add_argument("--status-parquet", type=Path, required=True,
                   help="Path to extraction_status.parquet")
    p.add_argument("--keyframe-mapping", type=Path, required=True,
                   help="Path to canonical keyframe_mapping.csv")
    p.add_argument("--beit3-vectors", type=Path, required=True)
    p.add_argument("--beit3-mapping", type=Path, default=None)
    p.add_argument("--beit3-manifest", type=Path, default=None)
    p.add_argument("--metaclip2-vectors", type=Path, required=True)
    p.add_argument("--metaclip2-mapping", type=Path, default=None)
    p.add_argument("--metaclip2-manifest", type=Path, default=None)
    p.add_argument("--pe-vectors", type=Path, required=True)
    p.add_argument("--pe-mapping", type=Path, default=None)
    p.add_argument("--pe-manifest", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=Path("data/milvus_index_manifest.json"))
    p.add_argument("--index-type", choices=["FLAT", "HNSW", "IVF_FLAT"], default="FLAT")
    p.add_argument("--collection", default=None)
    p.add_argument("--batch-size", type=_positive_int, default=1000)
    p.set_defaults(command="milvus", milvus_command="ingest")

    # milvus build-index
    p = milvus_subs.add_parser("build-index", help="Create ANN indexes on vector fields")
    p.add_argument("--index-type", choices=["FLAT", "HNSW", "IVF_FLAT"], default="HNSW")
    p.add_argument("--collection", default=None)
    p.set_defaults(command="milvus", milvus_command="build-index")

    # milvus verify
    p = milvus_subs.add_parser("verify", help="Verify collection schema, indexes, and manifest")
    p.add_argument("--collection", default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.set_defaults(command="milvus", milvus_command="verify")

    # milvus show-manifest
    p = milvus_subs.add_parser("show-manifest", help="Print the current semantic index manifest")
    p.add_argument("--manifest", type=Path, default=Path("data/milvus_index_manifest.json"))
    p.set_defaults(command="milvus", milvus_command="show-manifest")

    # milvus stale-status
    p = milvus_subs.add_parser("stale-status", help="Check if semantic index manifest is stale")
    p.add_argument("--manifest", type=Path, default=Path("data/milvus_index_manifest.json"))
    p.add_argument("--keyframe-mapping", type=Path, default=None)
    p.add_argument("--beit3-vectors", type=Path, default=None)
    p.add_argument("--metaclip2-vectors", type=Path, default=None)
    p.add_argument("--pe-vectors", type=Path, default=None)
    p.set_defaults(command="milvus", milvus_command="stale-status")


def _run_milvus(values: dict) -> None:
    milvus_command = values.pop("milvus_command")

    from .indexing.milvus_client import MilvusClient, MilvusConfig
    from .indexing.milvus_schema import COLLECTION_NAME

    collection_name = values.pop("collection", None) or COLLECTION_NAME

    if milvus_command == "status":
        manifest_path = values.pop("manifest", None)
        client = MilvusClient()
        try:
            health = client.health()
            if client.collection_exists(collection_name):
                info = client.collection_info(collection_name)
                health["collection_info"] = info
            else:
                health["collection_exists"] = False
            from .indexing.milvus_manifest import load_manifest
            if manifest_path:
                m = load_manifest(Path(manifest_path))
                health["manifest_state"] = m.state if m else "NOT_FOUND"
            print(json.dumps(health, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
            raise SystemExit(1)

    elif milvus_command == "init-schema":
        from .indexing.milvus_schema import build_collection_schema
        drop_existing = values.pop("drop_existing", False)
        client = MilvusClient()
        schema = build_collection_schema()
        client.create_collection(schema, collection_name, drop_existing=drop_existing)
        print(json.dumps({"status": "ok", "collection": collection_name}, indent=2))

    elif milvus_command == "ingest":
        from .indexing.milvus_ingest import ingest_to_milvus
        from .indexing.milvus_client import MilvusClient
        provider_artifacts = {
            "beit3": {
                "vectors": values.pop("beit3_vectors"),
                "mapping": values.pop("beit3_mapping"),
                "manifest": values.pop("beit3_manifest"),
            },
            "metaclip2": {
                "vectors": values.pop("metaclip2_vectors"),
                "mapping": values.pop("metaclip2_mapping"),
                "manifest": values.pop("metaclip2_manifest"),
            },
            "perception_encoder": {
                "vectors": values.pop("pe_vectors"),
                "mapping": values.pop("pe_mapping"),
                "manifest": values.pop("pe_manifest"),
            },
        }
        # Remove None paths
        for pid, art in provider_artifacts.items():
            provider_artifacts[pid] = {k: v for k, v in art.items() if v is not None}
        result = ingest_to_milvus(
            status_parquet=values.pop("status_parquet"),
            keyframe_mapping_csv=values.pop("keyframe_mapping"),
            provider_artifacts=provider_artifacts,
            manifest_path=values.pop("manifest"),
            index_type=values.pop("index_type"),
            collection_name=collection_name,
            batch_size=values.pop("batch_size"),
        )
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2, ensure_ascii=False))

    elif milvus_command == "build-index":
        from .indexing.milvus_client import MilvusClient
        from .indexing.milvus_schema import VECTOR_FIELDS, INDEX_TYPE_PARAMS, CANONICAL_METRIC
        from .indexing.milvus_client import CANONICAL_METRIC as _METRIC
        index_type = values.pop("index_type", "HNSW")
        client = MilvusClient()
        client.connect()
        for vf in VECTOR_FIELDS:
            client.create_index(
                vf,
                index_type=index_type,
                metric_type=_METRIC,
                params=INDEX_TYPE_PARAMS.get(index_type, {}),
                collection_name=collection_name,
            )
        client.load_collection(collection_name)
        print(json.dumps({
            "status": "ok", "collection": collection_name,
            "index_type": index_type, "fields": list(VECTOR_FIELDS),
        }, indent=2))

    elif milvus_command == "verify":
        from .indexing.milvus_client import MilvusClient
        from .indexing.milvus_schema import VECTOR_FIELDS, SCALAR_FIELDS
        from .indexing.milvus_manifest import load_manifest
        client = MilvusClient()
        result: dict = {}
        try:
            info = client.validate_collection(
                collection_name,
                expected_fields=list(VECTOR_FIELDS) + list(SCALAR_FIELDS),
            )
            result.update(info)
        except Exception as exc:
            result["error"] = str(exc)
        manifest_path = values.pop("manifest", None)
        if manifest_path:
            m = load_manifest(Path(manifest_path))
            result["manifest_state"] = m.state if m else "NOT_FOUND"
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif milvus_command == "show-manifest":
        from .indexing.milvus_manifest import load_manifest
        manifest_path = values.pop("manifest")
        m = load_manifest(Path(manifest_path))
        if m is None:
            print(json.dumps({"error": f"Manifest not found: {manifest_path}"}, indent=2))
            raise SystemExit(1)
        print(json.dumps(m.to_dict(), indent=2, ensure_ascii=False))

    elif milvus_command == "stale-status":
        from .indexing.milvus_manifest import load_manifest, check_staleness
        from .indexing.milvus_ingest import _sha256_path
        manifest_path = values.pop("manifest")
        m = load_manifest(Path(manifest_path))
        if m is None:
            print(json.dumps({"state": "NOT_FOUND"}, indent=2))
            raise SystemExit(1)
        kf_path = values.pop("keyframe_mapping", None)
        b3_path = values.pop("beit3_vectors", None)
        m2_path = values.pop("metaclip2_vectors", None)
        pe_path = values.pop("pe_vectors", None)
        checksums = {}
        if b3_path and Path(b3_path).is_file():
            checksums["beit3"] = _sha256_path(Path(b3_path))
        if m2_path and Path(m2_path).is_file():
            checksums["metaclip2"] = _sha256_path(Path(m2_path))
        if pe_path and Path(pe_path).is_file():
            checksums["perception_encoder"] = _sha256_path(Path(pe_path))
        kf_checksum = _sha256_path(Path(kf_path)) if kf_path and Path(kf_path).is_file() else ""
        state, reasons = check_staleness(
            m,
            current_keyframe_checksum=kf_checksum,
            current_provider_checksums=checksums,
            current_dimensions={},
        )
        print(json.dumps({"state": state, "reasons": reasons}, indent=2))
        if state == "STALE":
            raise SystemExit(1)

    else:
        raise SystemExit(f"Unknown milvus command: {milvus_command}")






def _add_r2_parser(subparsers: "argparse._SubParsersAction") -> None:
    """Add the `r2` command group (status / list / pull / push)."""
    parser = subparsers.add_parser(
        "r2",
        help="Cloudflare R2 object storage sync operations",
    )
    r2_subs = parser.add_subparsers(dest="r2_command", required=True)

    # r2 status
    p = r2_subs.add_parser("status", help="Check R2 connectivity and credential config")
    p.add_argument("--prefix", default=None, help="Prefix to probe (default: data/raw)")
    p.set_defaults(command="r2", r2_command="status")

    # r2 list
    p = r2_subs.add_parser("list", help="List objects under a prefix")
    p.add_argument("prefix", help="R2 object key prefix, e.g. data/corpus/keyframes")
    p.add_argument("--max-keys", type=int, default=100, help="Maximum objects to list (default 100; -1 for all)")
    p.set_defaults(command="r2", r2_command="list")

    # r2 pull
    p = r2_subs.add_parser("pull", help="Pull objects from R2 into the local tree")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prefix", default=None, help="Pull every object under this prefix")
    group.add_argument("--videos", default=None, help="Comma-separated video IDs (requires --batch and --collection)")
    group.add_argument("--corpus", action="store_true", help="Pull the whole canonical corpus tree")
    group.add_argument("--manifest", action="store_true", help="Pull only the dataset manifest")
    p.add_argument("--batch", default=None, help="Batch id for raw videos, e.g. 1")
    p.add_argument("--collection", default=None, help="Collection id for raw videos, e.g. L21")
    p.add_argument("--local-root", default=".", help="Local root directory (default: current dir)")
    p.add_argument("--force", action="store_true", help="Re-download even if a local file exists")
    p.set_defaults(command="r2", r2_command="pull")

    # r2 push
    p = r2_subs.add_parser("push", help="Push local artifacts to R2")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staging", default=None, help="Push directory of files under data/staging/<feature>/<run_id>")
    mode.add_argument("--parquet", default=None, help="Push a canonical metadata parquet to data/corpus/metadata")
    mode.add_argument("--embeddings", default=None, help="Push a provider embedding file to data/corpus/features")
    mode.add_argument("--index", default=None, help="Push an index artifact to data/corpus/indexes")
    mode.add_argument("--report", default=None, help="Push a report JSON to data/corpus/reports")
    p.add_argument("--feature", default=None, help="Staging feature name (ocr/asr/objects/embeddings)")
    p.add_argument("--run-id", default=None, help="Staging run id")
    p.add_argument("--provider", default=None, help="Provider id for --embeddings (beit3/metaclip2/perception_encoder)")
    p.add_argument("--batch", default=None, help="Batch id for parquet metadata")
    p.add_argument("--collection", default=None, help="Collection id for parquet metadata")
    p.add_argument("--name", default=None, help="Artifact name/parts for indexes/reports (comma-separated path parts)")
    p.set_defaults(command="r2", r2_command="push")


def _load_dotenv() -> None:
    """Load .env into os.environ if present and not already set.

    Lets `python -m mvr.cli r2 ...` work without manually exporting
    credentials. Existing environment variables always win.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(".env")
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def _run_r2(values: dict) -> None:
    _load_dotenv()
    r2_command = values.pop("r2_command")

    from .operations import cache as cache_mod
    from .operations.r2_client import R2Client
    from .operations.r2_sync import R2Sync

    local_root = values.pop("local_root", None) or "."
    cache_dir = values.pop("cache_dir", None) or cache_mod.DEFAULT_CACHE_DIR
    max_gb = values.pop("cache_max_gb", None) or cache_mod.DEFAULT_MAX_SIZE_GB

    client = R2Client()
    cache = cache_mod.BoundedCache(cache_dir, max_size_gb=max_gb)
    sync = R2Sync(client=client, local_root=local_root, cache=cache)

    if r2_command == "status":
        prefix = values.pop("prefix", None) or sync.raw_prefix
        print(json.dumps({
            "status": "checking",
            "prefix": prefix,
            "warning": "use r2 list to confirm access",
        }, indent=2, ensure_ascii=False))
        return

    if r2_command == "list":
        prefix = values.pop("prefix")
        max_keys = values.pop("max_keys")
        objects = sync.list_remote(prefix)
        if max_keys is not None and max_keys >= 0:
            objects = objects[:max_keys]
        print(json.dumps({
            "prefix": prefix,
            "count": len(objects),
            "objects": [
                {"key": o.get("Key"), "size": o.get("Size")} for o in objects
            ],
        }, indent=2, ensure_ascii=False))
        return

    if r2_command == "pull":
        force = values.pop("force", False)
        if values.pop("manifest", False):
            path = sync.pull_dataset_manifest(force=force)
            print(json.dumps({
                "mode": "manifest",
                "local_path": str(path) if path else None,
            }, indent=2, ensure_ascii=False))
            return
        if values.pop("corpus", False):
            stats = sync.pull_corpus(force=force)
            print(json.dumps({"mode": "corpus", **stats.to_dict()}, indent=2, ensure_ascii=False))
            return
        prefix = values.pop("prefix", None)
        if prefix:
            stats = sync.pull_prefix(prefix, force=force)
            print(json.dumps({"mode": "prefix", "prefix": prefix, **stats.to_dict()}, indent=2, ensure_ascii=False))
            return
        videos = values.pop("videos", None)
        if videos:
            batch = values.pop("batch", None)
            collection = values.pop("collection", None)
            if not batch or not collection:
                raise SystemExit("--videos requires --batch and --collection")
            video_ids = [v.strip() for v in videos.split(",") if v.strip()]
            stats = sync.pull_videos(batch, collection, video_ids, force=force)
            print(json.dumps({"mode": "videos", "batch": batch, "collection": collection, **stats.to_dict()}, indent=2, ensure_ascii=False))
            return
        raise SystemExit("r2 pull requires one of --prefix/--videos/--corpus/--manifest")

    if r2_command == "push":
        from pathlib import Path as _Path
        staging = values.pop("staging", None)
        parquet = values.pop("parquet", None)
        embeddings = values.pop("embeddings", None)
        index = values.pop("index", None)
        report = values.pop("report", None)
        feature = values.pop("feature", None)
        run_id = values.pop("run_id", None)
        provider = values.pop("provider", None)
        batch = values.pop("batch", None)
        collection = values.pop("collection", None)
        name = values.pop("name", None)
        parts = [p.strip() for p in name.split(",") if p.strip()] if name else []

        if staging:
            if not feature or not run_id:
                raise SystemExit("--staging requires --feature and --run-id")
            files = sorted(_Path(staging).iterdir()) if _Path(staging).is_dir() else [_Path(staging)]
            stats = sync.push_staging(feature, run_id, files)
            print(json.dumps({"mode": "staging", "feature": feature, "run_id": run_id, **stats.to_dict()}, indent=2, ensure_ascii=False))
            return
        if parquet:
            if not feature or not batch or not collection:
                raise SystemExit("--parquet requires --feature, --batch, --collection")
            stats = sync.push_corpus_metadata(feature, batch, collection, _Path(parquet))
            print(json.dumps({"mode": "parquet", "feature": feature, **stats.to_dict()}, indent=2, ensure_ascii=False))
            return
        if embeddings:
            if not provider:
                raise SystemExit("--embeddings requires --provider")
            stats = sync.push_corpus_embeddings(provider, _Path(embeddings))
            print(json.dumps({"mode": "embeddings", "provider": provider, **stats.to_dict()}, indent=2, ensure_ascii=False))
            return
        if index:
            stats = sync.push_corpus_index(_Path(index), *parts) if parts else sync.push_corpus_index(_Path(index))
            print(json.dumps({"mode": "index", "parts": parts, **stats.to_dict()}, indent=2, ensure_ascii=False))
            return
        if report:
            stats = sync.push_corpus_report(_Path(report), *parts) if parts else sync.push_corpus_report(_Path(report))
            print(json.dumps({"mode": "report", "parts": parts, **stats.to_dict()}, indent=2, ensure_ascii=False))
            return
        raise SystemExit("r2 push requires one of --staging/--parquet/--embeddings/--index/--report")

def _add_worker_parser(subparsers: "argparse._SubParsersAction") -> None:
    """Add the `worker` command group (DATA-03 prefetch orchestration).

    The worker CLI owns its own subcommand parser (run/compare/status), so we
    capture everything after `worker` and hand it off intact.
    """
    parser = subparsers.add_parser(
        "worker",
        help="DATA-03 worker prefetch orchestration (run / compare / status)",
    )
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    parser.set_defaults(command="worker")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mvr.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_ingest_parser(subparsers)
    _add_build_index_parser(subparsers)
    _add_search_parser(subparsers)
    _add_build_aic_index_parser(subparsers)
    _add_build_asr_features_parser(subparsers)
    _add_benchmark_kis_parser(subparsers)
    _add_enrich_ocr_parser(subparsers)
    # Data-02 commands
    _add_show_defaults_parser(subparsers)
    _add_audit_status_parser(subparsers)
    _add_consolidate_parser(subparsers)
    _add_validate_staging_parser(subparsers)
    _add_promote_parser(subparsers)
    # Task05-Milvus commands
    _add_milvus_parser(subparsers)
    # R2 sync commands
    _add_r2_parser(subparsers)
    # DATA-03 worker prefetch commands
    _add_worker_parser(subparsers)
    values = vars(parser.parse_args(argv))
    command = values.pop("command")
    if command == "worker":
        from .worker.cli import main as worker_main
        raise SystemExit(worker_main(values.pop("worker_args")))
    if command == "ingest":
        values["input_path"] = values.pop("input")
        values["output_root"] = values.pop("output")
        run_pipeline(**values)
        return
    if command == "build-index":
        encoder = OpenClipEncoder(device=values.pop("device"))
        index = build_semantic_index(
            values.pop("input"),
            values.pop("output"),
            encoder,
            batch_size=values.pop("batch_size"),
        )
        print(json.dumps(index.manifest, indent=2, ensure_ascii=False))
        return
    if command == "build-aic-index":
        _run_build_aic_index(**values)
        return
    if command == "build-asr-features":
        _run_build_asr_features(**values)
        return
    if command == "benchmark-kis":
        _run_benchmark_kis(**values)
        return
    if command == "enrich-ocr":
        mapping_csv = values.pop("mapping_csv")
        video_root = values.pop("video_root")
        output = values.pop("output")
        device = values.pop("device")
        ocr_conf = values.pop("ocr_conf_threshold")
        text_sim = values.pop("text_similarity_threshold")
        bbox_iou = values.pop("bbox_iou_threshold")

        # We process each video separately. A naive way is to loop over videos based on the mapping-csv.
        # However, mapping-csv contains all keyframes. We can group them by video, then find the shots.json.
        # But wait, the instruction says "Run temporal OCR enrichment". The input contract:
        # artifacts/ingest_<dataset>/
        # ├── video_metadata.csv
        # ├── keyframes/
        # ├── keyframe_mapping.csv
        # └── shots.json
        # So we process the entire dataset ingest dir!
        # wait, video-root is supposed to be the root of the raw videos?
        # The prompt says: --video-root
        # Yes, extract_keyframes needs the original video file.

        # Let's write the handler here or call a main function in scene_text.py
        from .enrichment.ocr.scene_text import process_dataset_ocr
        manifest = process_dataset_ocr(
            mapping_csv=mapping_csv,
            video_root=video_root,
            output_dir=output,
            device=device,
            ocr_conf_threshold=ocr_conf,
            text_similarity_threshold=text_sim,
            bbox_iou_threshold=bbox_iou
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    # Data-02 commands
    if command == "show-defaults":
        _run_show_defaults()
        return
    if command == "audit-status":
        _run_audit_status(values.pop("status_file"))
        return
    if command == "consolidate-parquet":
        _run_consolidate(
            values.pop("source"), values.pop("output"),
            values.pop("schema"), values.pop("batch_id"),
            values.pop("collection_id"),
        )
        return
    if command == "validate-staging":
        _run_validate_staging(
            values.pop("file"), values.pop("schema"), values.pop("min_rows"),
        )
        return
    if command == "promote":
        _run_promote(
            values.pop("staging"), values.pop("corpus"),
            values.pop("schema"), values.pop("manifest"),
        )
        return
    if command == "milvus":
        _run_milvus(values)
        return
    if command == "r2":
        _run_r2(values)
        return

    index = FaissIndex.load(values.pop("index"))
    encoder = OpenClipEncoder(
        model_name=_required_manifest_string(index.manifest, "embedding_model"),
        pretrained=_required_manifest_string(index.manifest, "pretrained"),
        device=values.pop("device"),
    )
    results = SemanticRetriever(encoder, index).search(
        values.pop("query"),
        top_k=values.pop("top_k"),
    )
    print(
        json.dumps(
            [result.to_dict() for result in results],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
