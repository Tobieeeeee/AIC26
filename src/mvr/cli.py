from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .ingestion.pipeline import run_pipeline


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mvr.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_ingest_parser(subparsers)
    values = vars(parser.parse_args(argv))
    values.pop("command")
    values["input_path"] = values.pop("input")
    values["output_root"] = values.pop("output")
    run_pipeline(**values)


if __name__ == "__main__":
    main()
