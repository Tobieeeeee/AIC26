from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .enrichment.embeddings.openclip import OpenClipEncoder
from .indexing.faiss_index import FaissIndex
from .indexing.semantic_builder import build_semantic_index
from .ingestion.pipeline import run_pipeline
from .retrieval.semantic import SemanticRetriever


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mvr.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_ingest_parser(subparsers)
    _add_build_index_parser(subparsers)
    _add_search_parser(subparsers)
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
