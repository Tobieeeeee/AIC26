from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mvr.ingestion import pipeline as ingestion_pipeline
from scripts.operations import run_pipeline_cloud, run_pipeline_local


def test_local_r2_ingest_inherits_canonical_selector(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_pipeline(**kwargs) -> None:
        captured.update(kwargs)
        (kwargs["output_root"] / "keyframe_mapping.csv").parent.mkdir(
            parents=True, exist_ok=True
        )
        (kwargs["output_root"] / "keyframe_mapping.csv").write_text(
            "keyframe_id\n", encoding="utf-8"
        )

    monkeypatch.setattr(ingestion_pipeline, "run_pipeline", fake_run_pipeline)
    output = tmp_path / "out"
    mapping = run_pipeline_local.run_ingest(
        tmp_path / "video.mp4",
        output,
        "cuda",
        provider_specs_path=Path("configs/embedding_providers.json"),
        metaclip2_model="locked-model",
        allow_model_downloads=False,
        embedding_batch_size=8,
    )

    assert ingestion_pipeline.CANONICAL_SELECTOR_MODE == "hybrid_temporal_content"
    assert "selector_mode" not in captured
    assert "methods" not in captured
    assert captured["metaclip2_model"] == "locked-model"
    assert captured["allow_model_downloads"] is False
    assert mapping == output / "keyframe_mapping.csv"


def test_local_runner_model_downloads_are_explicit() -> None:
    assert not run_pipeline_local.parse_args([]).allow_model_downloads
    assert run_pipeline_local.parse_args(["--allow-model-downloads"]).allow_model_downloads


def test_ingest_only_eviction_requires_successful_push() -> None:
    with pytest.raises(SystemExit):
        run_pipeline_local.parse_args(["--evict-artifacts-after-push"])
    with pytest.raises(SystemExit):
        run_pipeline_local.parse_args(
            ["--ingest-only", "--no-push", "--evict-artifacts-after-push"]
        )


def test_local_runner_resolves_model_settings_from_environment(
    monkeypatch, tmp_path: Path
) -> None:
    detector = tmp_path / "rf-detr-large-2026.pth"
    beit3 = tmp_path / "beit3.pth"
    pe = tmp_path / "PE-Core-L14-336.pt"
    metaclip = tmp_path / "metaclip"
    monkeypatch.setenv("MVR_RFDETR_CHECKPOINT", str(detector))
    monkeypatch.setenv("MVR_BEIT3_CHECKPOINT", str(beit3))
    monkeypatch.setenv("MVR_PE_CHECKPOINT", str(pe))
    monkeypatch.setenv("MVR_METACLIP2_MODEL", str(metaclip))
    monkeypatch.setenv("MVR_METACLIP2_ALLOW_DOWNLOAD", "true")

    settings = run_pipeline_local.resolve_runtime_settings(
        run_pipeline_local.parse_args([])
    )

    assert settings == run_pipeline_local.RuntimeSettings(
        detector_checkpoint=detector,
        beit3_checkpoint=beit3,
        pe_checkpoint=pe,
        metaclip2_model=str(metaclip),
        allow_model_downloads=True,
    )


def test_local_runner_reuses_canonical_selector_metaclip_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    from mvr.enrichment import feature_extraction
    from mvr.enrichment.embeddings import offline, provider_specs

    output = tmp_path / "ingest"
    selector_artifact = output / "selector_embeddings" / "metaclip2"
    selector_artifact.mkdir(parents=True)
    rows = [{"video_id": "L21_V001", "keyframe_id": "kf-1", "frame_idx": "1"}]
    spec = SimpleNamespace(model_id="metaclip2")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        feature_extraction, "sort_keyframes_deterministically", lambda _: rows
    )
    monkeypatch.setattr(feature_extraction, "extract_objects", lambda *args: None)
    monkeypatch.setattr(
        provider_specs,
        "load_provider_specs",
        lambda _: SimpleNamespace(get=lambda provider_id: spec),
    )

    def fake_reuse(source, **kwargs) -> None:
        captured["source"] = source
        captured.update(kwargs)

    monkeypatch.setattr(offline, "reuse_embedding_artifact", fake_reuse)
    monkeypatch.setattr(
        run_pipeline_local,
        "_build_provider_encoder",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("MetaCLIP-2 encoder must not be constructed")
        ),
    )

    count, timings = run_pipeline_local.run_objects_and_embeddings(
        output / "keyframe_mapping.csv",
        output,
        output,
        object(),
        "L21_V001",
        ("metaclip2",),
        Path("configs/embedding_providers.json"),
        None,
        None,
        None,
        "cuda",
        8,
        False,
    )

    assert count == 1
    assert "metaclip2" in timings
    assert captured == {
        "source": selector_artifact,
        "embeddings_root": output / "embeddings",
        "rows": rows,
        "spec": spec,
    }


def test_cloud_runner_forwards_canonical_runtime_options(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess_run(cmd, **kwargs):
        captured["cmd"] = [str(item) for item in cmd]
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(run_pipeline_cloud.subprocess, "run", fake_subprocess_run)
    result = run_pipeline_cloud.main(
        [
            "--skip-provision",
            "--dest",
            str(tmp_path),
            "--env-file",
            str(tmp_path / ".env"),
            "--allow-model-downloads",
            "--ingest-only",
            "--evict-artifacts-after-push",
        ]
    )

    assert result == 0
    assert captured["cwd"] == tmp_path
    assert "--allow-model-downloads" in captured["cmd"]
    assert "--ingest-only" in captured["cmd"]
    assert "--evict-artifacts-after-push" in captured["cmd"]


def test_ingest_only_skips_enrichment_and_uses_separate_resume_state(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def download_file(self, key, dest) -> None:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"video")

    def fake_ingest(video_path, out_root, device, **kwargs):
        out_root.mkdir(parents=True, exist_ok=True)
        mapping = out_root / "keyframe_mapping.csv"
        mapping.write_text(
            "keyframe_id,video_id,frame_idx\nkf-1,L21_V001,1\n",
            encoding="utf-8",
        )
        return mapping

    monkeypatch.setattr(run_pipeline_local, "_r2_client", lambda: FakeClient())
    monkeypatch.setattr(run_pipeline_local, "run_ingest", fake_ingest)
    monkeypatch.setattr(
        run_pipeline_local,
        "build_detector",
        lambda *args, **kwargs: pytest.fail("ingest-only must not build RF-DETR"),
    )
    monkeypatch.setattr(
        run_pipeline_local,
        "run_objects_and_embeddings",
        lambda *args, **kwargs: pytest.fail("ingest-only must not run enrichment"),
    )

    result = run_pipeline_local.main(
        [
            "--ingest-only",
            "--no-push",
            "--keep-video",
            "--collections",
            "L21",
            "--videos",
            "L21_V001",
            "--work-root",
            str(tmp_path),
        ]
    )

    assert result == 0
    state = json.loads((tmp_path / "ingest_state.json").read_text(encoding="utf-8"))
    assert state["completed"] == {"L21_V001": True}
    status = tmp_path / "ingestion/L21/L21_V001/L21_V001.status"
    assert status.read_text(encoding="utf-8").strip() == "INGEST_DONE"
    report = json.loads((tmp_path / "ingest_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "ingest-only"
    assert report["results"][0]["keyframe_count"] == 1


def test_ingest_only_does_not_mark_done_when_r2_upload_fails(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def download_file(self, key, dest) -> None:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"video")

    def fake_ingest(video_path, out_root, device, **kwargs):
        out_root.mkdir(parents=True, exist_ok=True)
        mapping = out_root / "keyframe_mapping.csv"
        mapping.write_text(
            "keyframe_id,video_id,frame_idx\nkf-1,L21_V001,1\n",
            encoding="utf-8",
        )
        return mapping

    monkeypatch.setattr(run_pipeline_local, "_r2_client", lambda: FakeClient())
    monkeypatch.setattr(run_pipeline_local, "run_ingest", fake_ingest)
    monkeypatch.setattr(
        run_pipeline_local,
        "push_ingestion_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("R2 failed")),
    )

    result = run_pipeline_local.main(
        [
            "--ingest-only",
            "--collections",
            "L21",
            "--videos",
            "L21_V001",
            "--work-root",
            str(tmp_path),
        ]
    )

    assert result == 2
    state = json.loads((tmp_path / "ingest_state.json").read_text(encoding="utf-8"))
    assert state["completed"] == {}
    assert "L21_V001" in state["failed"]
    output = tmp_path / "ingestion/L21/L21_V001"
    assert output.is_dir()
    assert not (output / "L21_V001.status").exists()
    assert (tmp_path / "videos/L21/L21_V001.mp4").is_file()


def test_ingest_only_can_evict_local_files_after_successful_push(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def download_file(self, key, dest) -> None:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"video")

    def fake_ingest(video_path, out_root, device, **kwargs):
        out_root.mkdir(parents=True, exist_ok=True)
        mapping = out_root / "keyframe_mapping.csv"
        mapping.write_text(
            "keyframe_id,video_id,frame_idx\nkf-1,L21_V001,1\n",
            encoding="utf-8",
        )
        return mapping

    pushed: list[dict] = []
    monkeypatch.setattr(run_pipeline_local, "_r2_client", lambda: FakeClient())
    monkeypatch.setattr(run_pipeline_local, "run_ingest", fake_ingest)
    monkeypatch.setattr(
        run_pipeline_local,
        "push_ingestion_artifacts",
        lambda *args, **kwargs: pushed.append(kwargs) or 3,
    )

    result = run_pipeline_local.main(
        [
            "--ingest-only",
            "--evict-artifacts-after-push",
            "--collections",
            "L21",
            "--videos",
            "L21_V001",
            "--work-root",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert pushed[0]["video_id"] == "L21_V001"
    assert not (tmp_path / "ingestion/L21/L21_V001").exists()
    assert not (tmp_path / "videos/L21/L21_V001.mp4").exists()
    state = json.loads((tmp_path / "ingest_state.json").read_text(encoding="utf-8"))
    assert state["completed"] == {"L21_V001": True}
