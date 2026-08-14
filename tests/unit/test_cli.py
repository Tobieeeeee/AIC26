import json
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


def test_build_index_command_uses_lab3_defaults(monkeypatch, tmp_path: Path) -> None:
    from mvr import cli

    calls: list[tuple] = []

    class FakeEncoder:
        def __init__(self, device: str) -> None:
            calls.append(("encoder", device))

    def fake_build(input_root, output_root, encoder, batch_size):
        calls.append(("build", input_root, output_root, encoder, batch_size))
        return type("BuiltIndex", (), {"manifest": {"vector_count": 2}})()

    monkeypatch.setattr(cli, "OpenClipEncoder", FakeEncoder)
    monkeypatch.setattr(cli, "build_semantic_index", fake_build)
    input_root = tmp_path / "keyframes"
    output_root = tmp_path / "index"

    cli.main(
        [
            "build-index",
            "--input",
            str(input_root),
            "--output",
            str(output_root),
            "--device",
            "cpu",
        ]
    )

    assert calls[0] == ("encoder", "cpu")
    assert calls[1][0:3] == ("build", input_root, output_root)
    assert calls[1][3] is not None
    assert calls[1][4] == 32


def test_search_command_prints_required_fields(monkeypatch, tmp_path: Path, capsys) -> None:
    from mvr import cli
    from mvr.retrieval.semantic import SearchResult

    calls: list[tuple] = []

    class FakeIndex:
        manifest = {
            "embedding_model": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
        }

        @classmethod
        def load(cls, index_root):
            calls.append(("load", index_root))
            return cls()

    class FakeEncoder:
        def __init__(self, model_name: str, pretrained: str, device: str) -> None:
            calls.append(("encoder", model_name, pretrained, device))

    class FakeRetriever:
        def __init__(self, encoder, index) -> None:
            calls.append(("retriever", encoder, index))

        def search(self, query: str, top_k: int):
            calls.append(("search", query, top_k))
            return [
                SearchResult(
                    rank=1,
                    frame_id="frame-a",
                    video_id="video-1",
                    timestamp=1.25,
                    score=0.75,
                    image_path="keyframes/video-1/frame-a.jpg",
                )
            ]

    monkeypatch.setattr(cli, "FaissIndex", FakeIndex)
    monkeypatch.setattr(cli, "OpenClipEncoder", FakeEncoder)
    monkeypatch.setattr(cli, "SemanticRetriever", FakeRetriever)
    index_root = tmp_path / "index"

    cli.main(
        [
            "search",
            "a man riding a motorcycle",
            "--index",
            str(index_root),
            "--top-k",
            "3",
            "--device",
            "cpu",
        ]
    )

    assert json.loads(capsys.readouterr().out) == [
        {
            "rank": 1,
            "frame_id": "frame-a",
            "video_id": "video-1",
            "timestamp": 1.25,
            "score": 0.75,
            "image_path": "keyframes/video-1/frame-a.jpg",
        }
    ]
    assert calls[0] == ("load", index_root)
    assert calls[1] == (
        "encoder",
        "ViT-B-32",
        "laion2b_s34b_b79k",
        "cpu",
    )
    assert calls[-1] == ("search", "a man riding a motorcycle", 3)


def test_semantic_commands_reject_non_positive_values_before_model_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import pytest

    from mvr import cli

    monkeypatch.setattr(
        cli,
        "OpenClipEncoder",
        lambda **_: pytest.fail("model must not load for invalid CLI arguments"),
    )
    with pytest.raises(SystemExit):
        cli.main(
            [
                "build-index",
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "index"),
                "--batch-size",
                "0",
            ]
        )
    with pytest.raises(SystemExit):
        cli.main(
            [
                "search",
                "query",
                "--index",
                str(tmp_path / "index"),
                "--top-k",
                "-1",
            ]
        )


def test_search_requires_embedding_identity_in_manifest(monkeypatch, tmp_path: Path) -> None:
    import pytest

    from mvr import cli

    class FakeIndex:
        manifest = {"embedding_model": "ViT-B-32"}

        @classmethod
        def load(cls, _index_root):
            return cls()

    monkeypatch.setattr(cli, "FaissIndex", FakeIndex)
    monkeypatch.setattr(
        cli,
        "OpenClipEncoder",
        lambda **_: pytest.fail("model must not load without its pretrained ID"),
    )

    with pytest.raises(ValueError, match="pretrained"):
        cli.main(["search", "query", "--index", str(tmp_path)])
