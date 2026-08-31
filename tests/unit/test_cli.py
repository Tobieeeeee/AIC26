import json
from pathlib import Path


def test_ingest_command_uses_canonical_hybrid_defaults(monkeypatch, tmp_path: Path) -> None:
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
            "methods": ["transnetv2"],
            "primary": "transnetv2",
            "pyscene_threshold": 27.0,
            "transnet_threshold": 0.5,
            "device": "auto",
            "selector_mode": "hybrid_temporal_content",
            "long_shot_seconds": 12.0,
            "sample_every_seconds": 8.0,
            "candidate_step_sec": 1.0,
            "max_temporal_gap_sec": 8.0,
            "clip_knn_k": 3,
            "clip_neighbor_window_sec": 4.0,
            "clip_novelty_threshold": 0.15,
            "clip_rare_threshold": 0.25,
            "min_event_gap_sec": 2.0,
            "edge_change_mad_multiplier": 3.0,
            "edge_change_min_score": 0.08,
            "midpoint_mandatory": True,
            "semantic_dedup_candidate_fps": 2.0,
            "semantic_dedup_similarity_threshold": 0.85,
            "semantic_dedup_max_gap_seconds": 8.0,
            "semantic_dedup_force_first_frame_per_shot": True,
            "provider_specs_path": Path("configs/embedding_providers.json"),
            "metaclip2_model": None,
            "allow_model_downloads": False,
            "embedding_batch_size": 16,
        }
    ]


def test_show_defaults_reports_canonical_hybrid_profile(capsys) -> None:
    from mvr import cli

    cli.main(["show-defaults"])

    output = capsys.readouterr().out
    assert (
        "Ingestion: transnetv2 @ 0.5 + hybrid_temporal_content "
        "(candidate 1.0s, max gap 8.0s)"
    ) in output
    assert "?s" not in output


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


def test_build_aic_index_command_discovers_videos_and_reuses_semantic_builder(
    monkeypatch, tmp_path: Path
) -> None:
    """The AIC dataset -> index path must be reproducible from the CLI alone,
    otherwise a teammate cannot rebuild the index the README points at."""
    from mvr import cli

    dataset_root = tmp_path / "dataset"
    for video_id in ("L21_V001", "L21_V002"):
        keyframe_dir = dataset_root / "keyframes" / video_id
        keyframe_dir.mkdir(parents=True)
        (keyframe_dir / "001.jpg").write_bytes(b"")
        map_dir = dataset_root / "map_keyframes"
        map_dir.mkdir(exist_ok=True)
        (map_dir / f"{video_id}.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n", encoding="utf-8"
        )
    # A video with keyframes but no mapping CSV must be skipped, not guessed at.
    (dataset_root / "keyframes" / "L21_V999").mkdir(parents=True)

    calls: list[tuple] = []

    class FakeEncoder:
        def __init__(self, device: str) -> None:
            calls.append(("encoder", device))

    def fake_build(input_root, output_root, encoder, batch_size):
        calls.append(("build", Path(input_root), Path(output_root), batch_size))
        return type("BuiltIndex", (), {"manifest": {"vector_count": 2}})()

    monkeypatch.setattr(cli, "OpenClipEncoder", FakeEncoder)
    monkeypatch.setattr(cli, "build_semantic_index", fake_build)
    output_root = tmp_path / "index"

    cli.main(
        [
            "build-aic-index",
            "--dataset-root", str(dataset_root),
            "--output", str(output_root),
            "--device", "cpu",
        ]
    )

    assert calls[0] == ("encoder", "cpu")
    staging_root = calls[1][1]
    assert calls[1][2] == output_root
    assert calls[1][3] == 32

    import csv as csv_module

    with (staging_root / "keyframe_mapping.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    assert {row["video_id"] for row in rows} == {"L21_V001", "L21_V002"}


def test_build_aic_index_command_rejects_dataset_with_no_indexable_videos(
    monkeypatch, tmp_path: Path
) -> None:
    import pytest

    from mvr import cli

    dataset_root = tmp_path / "dataset"
    (dataset_root / "keyframes").mkdir(parents=True)
    monkeypatch.setattr(
        cli, "OpenClipEncoder", lambda **_: pytest.fail("model must not load with no videos")
    )

    with pytest.raises(ValueError, match="no indexable videos"):
        cli.main(
            [
                "build-aic-index",
                "--dataset-root", str(dataset_root),
                "--output", str(tmp_path / "index"),
            ]
        )


def test_benchmark_kis_command_writes_summary_and_records(monkeypatch, tmp_path: Path) -> None:
    from mvr import cli
    from mvr.retrieval.semantic import SearchResult

    class FakeIndex:
        manifest = {"embedding_model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}

        @classmethod
        def load(cls, index_root):
            return cls()

    class FakeEncoder:
        def __init__(self, model_name: str, pretrained: str, device: str) -> None:
            pass

        def encode_text(self, texts):
            return [[1.0]]

    class FakeRetriever:
        def __init__(self, encoder, index) -> None:
            pass

        def search(self, query: str, top_k: int):
            return [
                SearchResult(
                    rank=1, frame_id="003", video_id="L21_V001",
                    timestamp=8.7, score=0.9, image_path="unused.jpg",
                )
            ]

    class FakeRegistry:
        def __init__(self, root) -> None:
            pass

        def video_path(self, video_id: str):
            return Path(f"{video_id}.mp4")

        def video_info(self, video_id: str):
            from mvr.ingestion.models import VideoInfo

            return VideoInfo(video_id, Path(f"{video_id}.mp4"), 30.0, 100000, 3000.0)

        def resolve_frame(self, video_id: str, keyframe_ref: str):
            from mvr.ingestion.aic_dataset import AicFrameMapping

            return AicFrameMapping(
                video_id=video_id, keyframe_ref=keyframe_ref, frame_idx=261,
                timestamp=8.7, fps=30.0,
            )

    refine_calls: list[tuple] = []

    def fake_refine_top_candidates(candidates, **kwargs):
        refine_calls.append((candidates, kwargs))
        return candidates  # pass-through: no real FFmpeg/model needed for this wiring test

    monkeypatch.setattr(cli, "FaissIndex", FakeIndex)
    monkeypatch.setattr(cli, "OpenClipEncoder", FakeEncoder)
    monkeypatch.setattr(cli, "SemanticRetriever", FakeRetriever)
    monkeypatch.setattr(cli, "AicDatasetRegistry", FakeRegistry)
    monkeypatch.setattr(cli, "refine_top_candidates", fake_refine_top_candidates)

    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "label": "internal",
                "queries": [
                    {
                        "query_id": "q1", "query_text": "a red car",
                        "video_id": "L21_V001", "start_frame": 250, "end_frame": 270,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "bench_out"

    cli.main(
        [
            "benchmark-kis",
            "--index", str(tmp_path / "index"),
            "--gt", str(gt_path),
            "--dataset-root", str(tmp_path / "dataset"),
            "--output", str(output_dir),
            "--device", "cpu",
        ]
    )

    records = json.loads((output_dir / "records.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert {r["config_name"] for r in records} == {"semantic", "semantic+refine"}
    assert all(r["final_score"] == 1.0 for r in records)  # frame 261 is inside [250,270]
    assert set(summary.keys()) == {"semantic", "semantic+refine"}
    assert refine_calls  # the refine branch was actually invoked, not skipped


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
