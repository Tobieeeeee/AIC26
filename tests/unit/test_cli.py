from pathlib import Path


def test_ingest_command_preserves_lab_defaults(monkeypatch, tmp_path: Path) -> None:
    from mvr import cli

    calls: list[dict] = []
    monkeypatch.setattr(cli, "run_pipeline", lambda **values: calls.append(values))

    input_path = tmp_path / "video.mp4"
    output_path = tmp_path / "output"
    cli.main(["ingest", str(input_path), "--output", str(output_path)])

    assert calls == [
        {
            "input_path": input_path,
            "output_root": output_path,
            "methods": ["pyscenedetect"],
            "primary": "pyscenedetect",
            "pyscene_threshold": 27.0,
            "transnet_threshold": 0.5,
            "device": "auto",
            "long_shot_seconds": 12.0,
            "sample_every_seconds": 8.0,
        }
    ]
