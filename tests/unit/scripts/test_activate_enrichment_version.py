"""Contract checks for the enrichment activation entry point."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "operations"
    / "activate_enrichment_version.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_activate_enrichment_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_required() -> None:
    with pytest.raises(SystemExit):
        _load().parse_args([])


def test_parse_args_exposes_the_version_and_work_root() -> None:
    args = _load().parse_args(["--version", "enrich-v1", "--work-root", "out"])

    assert args.version == "enrich-v1"
    assert args.work_root == Path("out")


def test_main_reports_a_failed_status_without_activating(monkeypatch, capsys) -> None:
    module = _load()
    from mvr.operations.enrichment_version import EnrichmentVersionError

    monkeypatch.setattr(module, "R2Client", lambda *a, **k: object())

    def _refuse(**kwargs):
        raise EnrichmentVersionError("missing _READY marker")

    monkeypatch.setattr(module, "activate_enrichment_version", _refuse)

    assert module.main(["--version", "enrich-v1"]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "FAILED"
