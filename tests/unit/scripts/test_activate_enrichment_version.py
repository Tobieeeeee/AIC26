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


def test_validate_only_checks_the_version_without_activating(monkeypatch, capsys) -> None:
    module = _load()
    monkeypatch.setattr(module, "R2Client", lambda *a, **k: object())
    monkeypatch.setattr(
        module,
        "validate_enrichment_version",
        lambda **kwargs: {"version": kwargs["version"], "collections": ["L21"]},
    )

    def _must_not_run(**kwargs):
        raise AssertionError("activation ran during --validate-only")

    monkeypatch.setattr(module, "activate_enrichment_version", _must_not_run)

    assert module.main(["--version", "enrich-v1", "--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "VALIDATED"
    assert report["version"] == "enrich-v1"


def test_validate_only_reports_failure_without_activating(monkeypatch, capsys) -> None:
    module = _load()
    from mvr.operations.enrichment_version import EnrichmentVersionError

    monkeypatch.setattr(module, "R2Client", lambda *a, **k: object())

    def _refuse(**kwargs):
        raise EnrichmentVersionError("L21: missing task05a/ocr.tar.gz")

    monkeypatch.setattr(module, "validate_enrichment_version", _refuse)

    assert module.main(["--version", "enrich-v1", "--validate-only"]) == 2
    assert "ocr.tar.gz" in json.loads(capsys.readouterr().err)["error"]
