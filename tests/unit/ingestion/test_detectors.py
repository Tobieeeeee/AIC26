from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from mvr.ingestion.models import VideoInfo


def test_detector_factory_rejects_unknown_method() -> None:
    from mvr.ingestion.shot_detection import create_detector

    with pytest.raises(ValueError, match="Unknown detector: other"):
        create_detector("other", 1.0)


def test_pyscenedetect_converts_exclusive_ends_to_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_scenedetect = ModuleType("scenedetect")

    class ContentDetector:
        def __init__(self, threshold: float) -> None:
            self.threshold = threshold

    class Timecode:
        def __init__(self, frame: int) -> None:
            self.frame = frame

        def get_frames(self) -> int:
            return self.frame

    def detect(path: str, detector: ContentDetector, start_in_scene: bool):
        assert path == "video.mp4"
        assert detector.threshold == 27.0
        assert start_in_scene is True
        return [(Timecode(0), Timecode(10)), (Timecode(10), Timecode(25))]

    fake_scenedetect.ContentDetector = ContentDetector
    fake_scenedetect.detect = detect
    monkeypatch.setitem(sys.modules, "scenedetect", fake_scenedetect)

    from mvr.ingestion.shot_detection.pyscenedetect import PySceneDetector

    video = VideoInfo("video", Path("video.mp4"), 25.0, 25, 1.0)
    shots = PySceneDetector().detect(video)

    assert [(shot.start_frame, shot.end_frame) for shot in shots] == [
        (0, 9),
        (10, 24),
    ]


def test_pyscenedetect_falls_back_to_whole_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_scenedetect = ModuleType("scenedetect")
    fake_scenedetect.ContentDetector = lambda threshold: object()
    fake_scenedetect.detect = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "scenedetect", fake_scenedetect)

    sys.modules.pop("mvr.ingestion.shot_detection.pyscenedetect", None)
    from mvr.ingestion.shot_detection.pyscenedetect import PySceneDetector

    video = VideoInfo("video", Path("video.mp4"), 25.0, 25, 1.0)
    shots = PySceneDetector().detect(video)

    assert [(shot.start_frame, shot.end_frame) for shot in shots] == [(0, 24)]


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transnetv2_pytorch") is None,
    reason="optional TransNetV2 runtime dependencies are not installed",
)
def test_transnetv2_backend_is_importable_when_dependencies_are_available() -> None:
    from mvr.ingestion.shot_detection.transnetv2 import TransNetV2Detector

    assert TransNetV2Detector.name == "transnetv2"
