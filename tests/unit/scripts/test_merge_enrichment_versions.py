"""Contract checks for the enrichment merge entry point.

The command exists so a batch can be split across machines, so its argument
surface must not encode how many there are.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "operations"
    / "merge_enrichment_versions.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_merge_enrichment_versions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_target_and_at_least_one_source_are_required() -> None:
    module = _load()

    with pytest.raises(SystemExit):
        module.parse_args([])
    with pytest.raises(SystemExit):
        module.parse_args(["--target", "batch1-v1"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--source", "m1", "--source", "m2", "--source", "m3"],
        ["--source", "m1,m2,m3"],
        ["--source", "m1, m2", "--source", "m3"],
    ],
)
def test_sources_are_counted_by_neither_the_flag_nor_the_separator(argv: list[str]) -> None:
    """Three machines named three ways must reach the merge identically.

    Whoever runs this is coordinating machines by hand, so the command has to
    accept whichever shape their notes are already in.
    """
    args = _load().parse_args([*argv, "--target", "batch1-v1"])

    assert args.source == ["m1", "m2", "m3"]


def test_a_repeated_source_is_named_rather_than_silently_collapsed() -> None:
    """Listing one machine twice means the operator lost track of the split."""
    module = _load()

    with pytest.raises(SystemExit):
        module.parse_args(["--source", "m1", "--source", "m1", "--target", "v1"])


def test_main_reports_the_refusal_and_exits_non_zero(monkeypatch, capsys) -> None:
    module = _load()
    from mvr.operations.enrichment_version import EnrichmentVersionError

    monkeypatch.setattr(module, "R2Client", lambda *a, **k: object())

    def _refuse(**kwargs):
        raise EnrichmentVersionError("L22 is claimed by m1, m2")

    monkeypatch.setattr(module, "merge_enrichment_versions", _refuse)

    exit_code = module.main(
        ["--source", "m1", "--source", "m2", "--target", "batch1-v1"]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "FAILED"
    assert "L22" in payload["error"]
