import json
from pathlib import Path

import pytest

from mvr.ingestion.models import Keyframe, Shot, VideoInfo


def test_pipeline_uses_primary_shots_for_keyframes_and_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from mvr.ingestion import pipeline

    video_path = tmp_path / "video.mp4"
    video_path.touch()
    video = VideoInfo("video", video_path, 25.0, 25, 1.0)
    monkeypatch.setattr(pipeline, "probe_video", lambda path: video)

    class Detector:
        def __init__(self, method: str) -> None:
            self.method = method

        def detect(self, current: VideoInfo) -> list[Shot]:
            return [Shot(current.video_id, self.method, 1, 0, 24, current.fps)]

    monkeypatch.setattr(
        pipeline,
        "create_detector",
        lambda method, threshold, device: Detector(method),
    )
    keyframe = Keyframe(
        "video_S0001_K01",
        "video",
        1,
        1,
        12,
        0.48,
        "keyframes/video/video_S0001_K01.jpg",
    )
    selected: list[list[Shot]] = []

    def fake_make_keyframes(shots, long_shot_seconds, sample_every_seconds):
        selected.append(shots)
        return [keyframe]

    monkeypatch.setattr(pipeline, "make_keyframes", fake_make_keyframes)
    monkeypatch.setattr(pipeline, "extract_keyframes", lambda *args: None)
    metadata: list[tuple[list[Shot], list[Keyframe]]] = []
    monkeypatch.setattr(
        pipeline,
        "write_metadata",
        lambda output, videos, shots, frames: metadata.append((shots, frames)),
    )
    monkeypatch.setattr(pipeline, "write_benchmark", lambda *args: None)

    pipeline.run_pipeline(
        video_path,
        tmp_path / "output",
        ["pyscenedetect", "transnetv2"],
        "transnetv2",
        27.0,
        0.5,
        "cpu",
        12.0,
        8.0,
        selector_mode="periodic",
    )

    assert [shot.method for shot in selected[0]] == ["transnetv2"]
    assert metadata == [(selected[0], [keyframe])]


def test_probe_error_closes_progress_and_propagates_original_exception(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from mvr.ingestion import pipeline

    video_path = tmp_path / "broken.mp4"
    video_path.touch()

    class RecordingProgress:
        def __init__(self, iterable, **kwargs) -> None:
            self.iterable = iterable
            self.closed = False
            self.postfixes: list[dict[str, str]] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.closed = True

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **values: str) -> None:
            self.postfixes.append(values)

    bars: list[RecordingProgress] = []

    def fake_tqdm(iterable, **kwargs):
        bar = RecordingProgress(iterable, **kwargs)
        bars.append(bar)
        return bar

    failure = RuntimeError("probe failed")
    monkeypatch.setattr(pipeline, "tqdm", fake_tqdm)
    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda path: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(RuntimeError) as captured:
        pipeline.run_pipeline(
            video_path,
            tmp_path / "output",
            ["pyscenedetect"],
            "pyscenedetect",
            27.0,
            0.5,
            "cpu",
            12.0,
            8.0,
            selector_mode="periodic",
        )

    assert captured.value is failure
    assert bars[0].postfixes == [{"file": "broken.mp4"}]
    assert bars[0].closed is True


def test_hybrid_mode_propagates_metaclip_unavailability_without_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    from mvr.enrichment.embeddings import metaclip2
    from mvr.ingestion import pipeline

    video_path = tmp_path / "video.mp4"
    video_path.touch()
    video = VideoInfo("video", video_path, 25.0, 25, 1.0)
    monkeypatch.setattr(pipeline, "probe_video", lambda path: video)

    class Detector:
        def detect(self, current: VideoInfo) -> list[Shot]:
            return [Shot(current.video_id, "pyscenedetect", 1, 0, 24, current.fps)]

    monkeypatch.setattr(
        pipeline, "create_detector", lambda method, threshold, device: Detector()
    )

    def unavailable(**kwargs):
        raise metaclip2.MetaClip2DependencyError("pinned MetaCLIP unavailable")

    monkeypatch.setattr(metaclip2, "MetaClip2ImageEncoder", unavailable)
    with pytest.raises(
        metaclip2.MetaClip2DependencyError, match="pinned MetaCLIP unavailable"
    ):
        pipeline.run_pipeline(
            video_path,
            tmp_path / "output",
            ["pyscenedetect"],
            "pyscenedetect",
            27.0,
            0.5,
            "cpu",
            selector_mode="hybrid_temporal_content",
        )


def test_library_defaults_use_transnet_hybrid_with_content_encoder(
    monkeypatch, tmp_path: Path
) -> None:
    """The canonical library default is hybrid_temporal_content, which always
    needs MetaCLIP-2 to score candidate novelty -- unlike the retired
    periodic default, a bare run_pipeline() call is expected to load it."""
    from mvr.enrichment.embeddings import metaclip2
    from mvr.ingestion import pipeline

    video_path = tmp_path / "video.mp4"
    video_path.touch()
    video = VideoInfo("video", video_path, 25.0, 25, 1.0)
    monkeypatch.setattr(pipeline, "probe_video", lambda path: video)

    detector_calls: list[tuple[str, float, str]] = []

    class Detector:
        def detect(self, current: VideoInfo) -> list[Shot]:
            return [Shot(current.video_id, "transnetv2", 1, 0, 24, current.fps)]

    def make_detector(method: str, threshold: float, device: str):
        detector_calls.append((method, threshold, device))
        return Detector()

    monkeypatch.setattr(pipeline, "create_detector", make_detector)

    encoder_calls: list[dict] = []

    class FakeEncoder:
        pass

    def fake_encoder(**kwargs):
        encoder_calls.append(kwargs)
        return FakeEncoder()

    monkeypatch.setattr(metaclip2, "MetaClip2ImageEncoder", fake_encoder)

    keyframe = Keyframe(
        "video_S0001_K01",
        "video",
        1,
        1,
        12,
        0.48,
        "keyframes/video/video_S0001_K01.jpg",
    )

    def fake_run_hybrid_selector(
        *, video, shots, output_root, config, spec, encoder, embedding_batch_size
    ):
        manifest = {
            "video_id": video.video_id,
            "candidate_count": 1,
            "candidate_extraction_seconds": 0.0,
            "candidate_embedding_seconds": 0.0,
            "selector_only_seconds": 0.0,
            "content_event_selection_count": 0,
            "temporal_repair_selection_count": 0,
            "max_observed_temporal_gap_sec": 0.0,
        }
        return [keyframe], manifest

    monkeypatch.setattr(pipeline, "run_hybrid_selector", fake_run_hybrid_selector)
    monkeypatch.setattr(pipeline, "write_metadata", lambda *args: None)
    monkeypatch.setattr(pipeline, "write_benchmark", lambda *args: None)

    output = tmp_path / "output"
    pipeline.run_pipeline(video_path, output)

    assert detector_calls == [("transnetv2", 0.5, "auto")]
    assert len(encoder_calls) == 1
    manifest = json.loads((output / "selector_manifest.json").read_text())
    assert manifest["selector_mode"] == "hybrid_temporal_content"
    assert manifest["selected_keyframe_count"] == 1


def test_default_transnet_initialization_failure_never_falls_back(
    monkeypatch, tmp_path: Path
) -> None:
    from mvr.ingestion import pipeline

    video_path = tmp_path / "video.mp4"
    video_path.touch()
    video = VideoInfo("video", video_path, 25.0, 25, 1.0)
    monkeypatch.setattr(pipeline, "probe_video", lambda path: video)
    detector_calls: list[tuple[str, float, str]] = []
    failure = RuntimeError("TransNetV2 weights unavailable")

    def unavailable(method: str, threshold: float, device: str):
        detector_calls.append((method, threshold, device))
        raise failure

    monkeypatch.setattr(pipeline, "create_detector", unavailable)

    with pytest.raises(RuntimeError) as captured:
        pipeline.run_pipeline(video_path, tmp_path / "output")

    assert captured.value is failure
    assert detector_calls == [("transnetv2", 0.5, "auto")]
