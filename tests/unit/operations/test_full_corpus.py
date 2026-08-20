from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from mvr.operations.full_corpus import (
    PROVIDERS,
    STAGES,
    FullCorpusError,
    dry_run_plan,
    load_full_corpus_config,
    run_full_corpus,
)


def _archives(dataset: Path) -> None:
    dataset.mkdir(parents=True)
    with zipfile.ZipFile(dataset / "Keyframes_L21.zip", "w") as archive:
        for index in range(1, 6):
            archive.writestr(f"keyframes/L21_V001/{index:03d}.jpg", b"fixture")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["n", "pts_time", "fps", "frame_idx"])
    writer.writeheader()
    for index in range(1, 6):
        writer.writerow(
            {"n": index, "pts_time": index / 2, "fps": 25, "frame_idx": index * 10}
        )
    with zipfile.ZipFile(dataset / "maps.zip", "w") as archive:
        archive.writestr("map-keyframes/L21_V001.csv", buffer.getvalue())


def _config(tmp_path: Path, *, output_inside_repo=False, index_policy="create"):
    repository = tmp_path / "repo"
    repository.mkdir(parents=True)
    dataset = tmp_path / "dataset"
    _archives(dataset)
    output = repository / "artifacts" if output_inside_repo else tmp_path / "output"
    payload = {
        "dataset_root": str(dataset),
        "mapping_archive": "maps.zip",
        "keyframes_archive": "Keyframes_L21.zip",
        "canonical_mapping": "full/keyframe_mapping.csv",
        "keyframe_root": "full/keyframes",
        "output_root": str(output),
        "dataset_prefix": "L21_",
        "providers": list(PROVIDERS),
        "artifact_policy": "resume",
        "index_policy": index_policy,
        "resource_estimates": {"gpu": "test", "ram": "test", "disk": "test", "time": "test"},
        "commands": {stage: ["command", stage] for stage in STAGES},
    }
    config_path = repository / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path, repository


def test_dry_run_resolves_five_unique_identities_and_exact_order(tmp_path: Path) -> None:
    path, repository = _config(tmp_path)
    config = load_full_corpus_config(path, repository)
    plan = dry_run_plan(config)
    assert plan["source_audit"] == {
        "expected_keyframes": 5,
        "unique_keyframe_ids": 5,
        "video_count": 1,
        "duplicates": 0,
        "missing": 0,
        "extra": 0,
        "mapping_archive_sha256": plan["source_audit"]["mapping_archive_sha256"],
        "keyframes_archive_sha256": plan["source_audit"]["keyframes_archive_sha256"],
    }
    assert [row["stage"] for row in plan["stages"]] == list(STAGES)
    assert plan["approval_required"].startswith("Explicit owner approval")
    assert not config.output_root.is_relative_to(repository)


def test_execution_requires_approval_and_audits_before_next_provider(tmp_path: Path) -> None:
    path, repository = _config(tmp_path)
    config = load_full_corpus_config(path, repository)
    with pytest.raises(FullCorpusError, match="explicit approval"):
        run_full_corpus(config, approved=False, execute_stage=lambda *args: {})
    calls = []

    def execute(stage, command):
        calls.append(stage)
        if stage.startswith("audit_"):
            return {
                "status": "pass",
                "valid": 5,
                "missing": 0,
                "extra": 0,
                "duplicates": 0,
                "invalid": 0,
            }
        return {"status": "pass"}

    state = run_full_corpus(config, approved=True, execute_stage=execute)
    assert calls == list(STAGES)
    assert state["status"] == "complete"
    for provider in PROVIDERS:
        assert calls.index(f"audit_{provider}") == calls.index(f"extract_{provider}") + 1


def test_failed_audit_stops_before_next_extraction_and_resume_skips_completed(
    tmp_path: Path,
) -> None:
    path, repository = _config(tmp_path)
    config = load_full_corpus_config(path, repository)
    state_path = tmp_path / "status.json"
    first_calls = []

    def fail_beit(stage, command):
        first_calls.append(stage)
        if stage == "audit_beit3":
            return {
                "status": "pass",
                "valid": 4,
                "missing": 1,
                "extra": 0,
                "duplicates": 0,
                "invalid": 0,
            }
        return {"status": "pass"}

    with pytest.raises(FullCorpusError, match="identity audit"):
        run_full_corpus(
            config,
            approved=True,
            execute_stage=fail_beit,
            state_path=state_path,
        )
    assert first_calls == ["canonical_mapping", "extract_beit3", "audit_beit3"]
    second_calls = []

    def pass_all(stage, command):
        second_calls.append(stage)
        if stage.startswith("audit_"):
            return {
                "status": "pass",
                "valid": 5,
                "missing": 0,
                "extra": 0,
                "duplicates": 0,
                "invalid": 0,
            }
        return {"status": "pass"}

    run_full_corpus(
        config,
        approved=True,
        execute_stage=pass_all,
        state_path=state_path,
    )
    assert second_calls[0] == "audit_beit3"
    assert "extract_beit3" not in second_calls


def test_output_inside_repository_and_rebuild_execution_are_rejected(tmp_path: Path) -> None:
    path, repository = _config(tmp_path, output_inside_repo=True)
    with pytest.raises(FullCorpusError, match="outside"):
        load_full_corpus_config(path, repository)

    path, repository = _config(tmp_path / "second", index_policy="rebuild")
    config = load_full_corpus_config(path, repository)
    with pytest.raises(FullCorpusError, match="destructive-target approval"):
        run_full_corpus(config, approved=True, execute_stage=lambda *args: {"status": "pass"})


def test_resume_rejects_changed_config_hash(tmp_path: Path) -> None:
    path, repository = _config(tmp_path)
    config = load_full_corpus_config(path, repository)
    state_path = tmp_path / "status.json"
    state_path.write_text(
        json.dumps({"config_sha256": "wrong", "completed_stages": [], "stage_results": {}}),
        encoding="utf-8",
    )
    with pytest.raises(FullCorpusError, match="config hash mismatch"):
        run_full_corpus(
            config,
            approved=True,
            execute_stage=lambda *args: {"status": "pass"},
            state_path=state_path,
        )
