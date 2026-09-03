"""The sampling a run used has to travel with the run.

OCR reads each keyframe at several temporal offsets, and that choice sets both
the cost of the run and how much scene text it finds. It reaches the pipeline
from the command line, so it has to reach the resume fingerprint too: a run
resumed under different offsets would leave half its collections sampled one
way and half the other, with nothing in the artifacts to say which.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mvr.operations.r2_enrichment import EnrichmentRunConfig

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "operations" / "run_enrichment_r2.py"
)
DIGEST = "f2c468219be9752713c195aa56e6840a97a91512d0dcd9e72f3fafcff857392d"


def _load():
    spec = importlib.util.spec_from_file_location("_run_enrichment_r2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(**overrides) -> EnrichmentRunConfig:
    return EnrichmentRunConfig(
        run_id="enrich-b1-m1",
        batch_id="1",
        collections=("L26",),
        work_root=Path("work"),
        ocr_model_lock_sha256=DIGEST,
        **overrides,
    )


def test_the_default_sampling_is_the_one_the_pipeline_has_always_used() -> None:
    assert tuple(_config().ocr_temporal_offsets) == (-1, 0, 1)


def test_the_sampling_is_part_of_the_run_identity() -> None:
    """Two samplings are two different runs, not one run resumed."""
    assert _config(ocr_temporal_offsets=(0, 1)) != _config()


@pytest.mark.parametrize("offsets", [(), (-1, 1), (0, 0)])
def test_a_sampling_without_exactly_one_anchor_is_refused(offsets: tuple[int, ...]) -> None:
    """Every sample is measured against the keyframe itself, read exactly once."""
    with pytest.raises(ValueError):
        _config(ocr_temporal_offsets=offsets)


def test_the_command_line_reads_the_sampling_it_is_given() -> None:
    args = _load().parse_args(
        ["--collections", "L26", "--ocr-temporal-offsets", "0,1"]
    )

    assert args.ocr_temporal_offsets == (0, 1)


def test_the_command_line_default_changes_nothing() -> None:
    args = _load().parse_args(["--collections", "L26"])

    assert args.ocr_temporal_offsets == (-1, 0, 1)


def test_a_command_line_sampling_without_the_anchor_is_refused() -> None:
    with pytest.raises(SystemExit):
        _load().parse_args(["--collections", "L26", "--ocr-temporal-offsets", "-1,1"])


def test_the_ocr_stage_hands_the_sampling_to_the_pipeline(monkeypatch) -> None:
    module = _load()
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return {"backend": "docker"}

    monkeypatch.setattr(
        "mvr.enrichment.ocr.docker_ocr.process_dataset_docker_ocr", fake
    )

    module._run_ocr_stage(
        Path("m.csv"), Path("videos"), Path("out"), Path("tmp"),
        model_lock=Path("lock.json"),
        weights_root=Path("weights"),
        temporal_offsets=(0, 1),
    )

    assert seen["temporal_offsets"] == (0, 1)
