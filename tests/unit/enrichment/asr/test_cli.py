from pathlib import Path


def test_build_asr_features_cli_wires_required_options(monkeypatch, tmp_path: Path) -> None:
    from mvr import cli

    captured: dict = {}
    monkeypatch.setattr(cli, "_run_build_asr_features", lambda **values: captured.update(values))

    cli.main(
        [
            "build-asr-features",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output",
            str(tmp_path / "output"),
            "--pre-seconds",
            "2",
            "--post-seconds",
            "4",
            "--device",
            "cuda",
            "--videos",
            "L21_V010,L21_V025",
            "--force",
        ]
    )

    assert captured["dataset_root"] == tmp_path / "dataset"
    assert captured["output"] == tmp_path / "output"
    assert captured["pre_seconds"] == 2.0
    assert captured["post_seconds"] == 4.0
    assert captured["device"] == "cuda"
    assert captured["videos"] == ["L21_V010,L21_V025"]
    assert captured["force"] is True
    assert captured["allow_download"] is False
