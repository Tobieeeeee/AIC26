from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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
from .ingestion.pipeline import run_pipeline
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
        default=["pyscenedetect"],
    )
    parser.add_argument(
        "--primary",
        choices=["pyscenedetect", "transnetv2"],
        default="pyscenedetect",
    )
    parser.add_argument("--pyscene-threshold", type=float, default=27.0)
    parser.add_argument("--transnet-threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--long-shot-seconds", type=float, default=12.0)
    parser.add_argument("--sample-every-seconds", type=float, default=8.0)
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mvr.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_ingest_parser(subparsers)
    _add_build_index_parser(subparsers)
    _add_search_parser(subparsers)
    _add_build_aic_index_parser(subparsers)
    _add_benchmark_kis_parser(subparsers)
    values = vars(parser.parse_args(argv))
    command = values.pop("command")
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
    if command == "benchmark-kis":
        _run_benchmark_kis(**values)
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
